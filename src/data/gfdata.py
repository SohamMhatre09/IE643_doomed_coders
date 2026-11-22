"""
gfdata.py -- data layer for GAN_FINETUNING (Step 1).

One entry per ICBHI **train** cycle (the official test split is never cached or read here):
  * raw AST log-mel fbank, computed exactly like ASTFeatureExtractor (Kaldi fbank, 128 mel, 25 ms Hann,
    10 ms hop, no dither) but WITHOUT padding / normalisation -> variable length, stored float16 flat
  * metadata: class flags (crackle, wheeze), device, chest location + region, patient, acquisition mode
  * gan_split: "train" or "val" -- a patient-level hold-out (~15% of train patients) that the GAN never sees,
    used by the evaluation suite as the real reference set.

Waveforms are not duplicated: get_wave() reads the existing per-recording cache of train_optimal_pipeline.

Build:   $PY GAN_FINETUNING/gfdata.py            (writes GAN_FINETUNING/data/, then data_stats.md)
"""
import os
import sys
import csv
import random
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P
DATA = os.path.join(_P.OUT, "gan_data")                  # log-mel cache of ICBHI cycles for the generators
META_CSV = os.path.join(DATA, "meta.csv")
FBANK_NPY = os.path.join(DATA, "fbank.npy")

SR = 16000
N_MELS = 128
MAX_FRAMES = 1024                     # stored cap (~10.2 s); >99.9% of train cycles are shorter
WIN = 512                             # GAN window: 5.12 s, covers >95% of cycles without repeat padding
AST_MEAN, AST_STD = -4.2677393, 4.5689974
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
DEVICES = ["AKGC417L", "Litt3200", "LittC2SE", "Meditron"]
LOCS = ["Tc", "Al", "Ar", "Pl", "Pr", "Ll", "Lr"]
REGION = {"Tc": "trachea", "Al": "anterior", "Ar": "anterior",
          "Pl": "post_lat", "Pr": "post_lat", "Ll": "post_lat", "Lr": "post_lat"}
REGIONS = ["trachea", "anterior", "post_lat"]
VAL_FRAC, VAL_SEED = 0.15, 0


# ----------------------------------------------------------------------------- features
def fbank(wave):
    """Same call as transformers ASTFeatureExtractor._extract_fbank_features (torchaudio path)."""
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi
    return ta_kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)).unsqueeze(0),
                          sample_frequency=SR, window_type="hanning", num_mel_bins=N_MELS).numpy()


def ast_normalize(fb):
    return (fb - AST_MEAN) / (2 * AST_STD)


def ast_denormalize(x):
    return x * (2 * AST_STD) + AST_MEAN


# ----------------------------------------------------------------------------- patient hold-out
def choose_val_patients(rows):
    """~15% of train patients; among 500 random draws keep the one whose minority-class shares in val are
    closest to 15% (so crackle / wheeze / both are all represented in the reference set)."""
    pats = sorted({r["patient"] for r in rows})
    per_pat = defaultdict(Counter)
    for r in rows:
        per_pat[r["patient"]][r["label"]] += 1
    tot = Counter(r["label"] for r in rows)
    rng = random.Random(VAL_SEED)
    k = round(VAL_FRAC * len(pats))
    best, best_cost = None, 1e9
    for _ in range(500):
        cand = rng.sample(pats, k)
        c = Counter()
        for p in cand:
            c.update(per_pat[p])
        cost = sum(abs(c[l] / tot[l] - VAL_FRAC) for l in (1, 2, 3)) + 0.5 * abs(c[0] / tot[0] - VAL_FRAC)
        if cost < best_cost:
            best, best_cost = set(cand), cost
    return best


# ----------------------------------------------------------------------------- build
def build():
    import logging
    logging.disable(logging.CRITICAL)
    import train_optimal_pipeline as tp
    from tqdm import tqdm
    recs, cycles = tp.build_dataset(smoke_stride=1)
    tp.ensure_wave_cache(recs)
    os.makedirs(DATA, exist_ok=True)

    rows = []
    for c in cycles:
        if c["split"] != "train":
            continue
        r = recs[c["ridx"]]
        rows.append(dict(cid=c["cid"], rec_id=r["rec_id"], patient=r["patient"], label=c["label"],
                         crackle=int(c["label"] in (1, 3)), wheeze=int(c["label"] in (2, 3)),
                         device=r["equip"], loc=r["loc"], region=REGION[r["loc"]], mode=r["mode"],
                         start=c["start"], end=c["end"], dur=round(c["end"] - c["start"], 4)))
    val = choose_val_patients(rows)

    feats, off, mm = [], 0, {}
    for r in tqdm(rows, desc="fbank"):
        if r["rec_id"] not in mm:
            mm[r["rec_id"]] = np.load(tp.wave_path(r["rec_id"]), mmap_mode="r")
        w = mm[r["rec_id"]]
        s, e = int(r["start"] * SR), int(r["end"] * SR)
        fb = fbank(np.array(w[s:max(e, s + 1600)], dtype=np.float32))[:MAX_FRAMES]
        r.update(gan_split="val" if r["patient"] in val else "train", fb_off=off, fb_len=len(fb))
        feats.append(fb.astype(np.float16))
        off += len(fb)
    np.save(FBANK_NPY, np.concatenate(feats))
    with open(META_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # sanity: our fbank == ASTFeatureExtractor before padding/normalisation (first 5 cycles)
    from transformers import ASTFeatureExtractor
    fe = ASTFeatureExtractor.from_pretrained(tp.MODEL_NAME, local_files_only=_P.OFFLINE)
    fe.max_length = MAX_FRAMES
    for r in rows[:5]:
        w = mm[r["rec_id"]]
        s, e = int(r["start"] * SR), int(r["end"] * SR)
        ref = fe([np.array(w[s:max(e, s + 1600)], dtype=np.float32)], sampling_rate=SR,
                 return_tensors="np")["input_values"][0][:r["fb_len"]]
        ours = ast_normalize(load_fbank()[r["fb_off"]:r["fb_off"] + r["fb_len"]].astype(np.float32))
        assert np.abs(ref - ours).max() < 2e-3, "fbank mismatch vs ASTFeatureExtractor"
    print("cached %d train cycles, %d frames -> %s (matches ASTFeatureExtractor)" % (len(rows), off, DATA))
    write_stats(load_meta(), cycles, recs)


# ----------------------------------------------------------------------------- GAN training-data augmentation
def remove_gaps(w, min_gap_s=0.2, keep_s=0.04, rel_db=-25.0):
    """Cut internal / edge silences longer than min_gap_s (20 ms frames below rel_db of the 95th-pct RMS)."""
    hop = int(0.02 * SR)
    n = len(w) // hop
    if n < 10:
        return w
    rms = np.sqrt((w[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    loud = rms >= np.percentile(rms, 95) * 10 ** (rel_db / 20)
    keep = loud.copy()
    i, gap, k = 0, int(min_gap_s / 0.02), int(keep_s / 0.02)
    while i < n:
        if not loud[i]:
            j = i
            while j < n and not loud[j]:
                j += 1
            if j - i < gap:
                keep[i:j] = True                                     # short pauses are kept
            else:
                keep[i:i + k] = True; keep[max(i, j - k):j] = True   # long gaps shrink to 2 x keep_s
            i = j
        else:
            i += 1
    out = w[:n * hop].reshape(n, hop)[keep].reshape(-1)
    return out if len(out) >= 0.5 * SR else w


def time_stretch(w, rate):
    import torch
    import torchaudio
    n_fft, hop = 512, 128
    win = torch.hann_window(n_fft)
    S_ = torch.stft(torch.from_numpy(np.ascontiguousarray(w)), n_fft, hop, window=win, return_complex=True)
    adv = torch.linspace(0, np.pi * hop, n_fft // 2 + 1)[:, None]
    return torch.istft(torchaudio.functional.phase_vocoder(S_, rate, adv), n_fft, hop, window=win).numpy()


def augment_wave(w, rng):
    """GAN training-data augmentation, physically plausible ops only (each p=0.5, at least one): gap removal,
    time-stretch 0.85-1.15 (tempo only), speed 0.92-1.08 (tempo + pitch). No noise / gain: the GAN would learn
    to reproduce them (DiffAugment already varies gain on the critic side)."""
    from scipy.signal import resample_poly
    while True:
        ops = rng.rand(3) < 0.5
        if ops.any():
            break
    if ops[0]:
        w = remove_gaps(w)
    if ops[1]:
        w = time_stretch(w, rng.uniform(0.85, 1.15))
    if ops[2]:
        w = resample_poly(w, 100, int(round(100 * rng.uniform(0.92, 1.08))))
    return np.ascontiguousarray(w, dtype=np.float32)


AUG_META_CSV = os.path.join(DATA, "meta_aug.csv")
AUG_FBANK_NPY = os.path.join(DATA, "fbank_aug.npy")


def build_aug(copies=3, seed=0):
    """`copies` augmented versions of every gan_split=train MINORITY cycle -> data/meta_aug.csv + fbank_aug.npy.
    Rows keep the source cycle's metadata (class, device, region, patient) plus aug_of = source cid."""
    from tqdm import tqdm
    rng = np.random.RandomState(seed)
    rows = [r for r in load_meta() if r["gan_split"] == "train" and r["label"] in (1, 2, 3)]
    out, feats, off = [], [], 0
    for r in tqdm(rows, desc="aug x%d" % copies):
        w0 = get_wave(r)
        for k in range(copies):
            fb = fbank(augment_wave(w0, rng))[:MAX_FRAMES]
            out.append(dict(r, aug_of=r["cid"], aug_k=k, fb_off=off, fb_len=len(fb)))
            feats.append(fb.astype(np.float16)); off += len(fb)
    np.save(AUG_FBANK_NPY, np.concatenate(feats))
    with open(AUG_META_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("augmented cache: %d copies of %d minority cycles, %d frames" % (len(out), len(rows), off))


def load_aug():
    """-> (rows, fbank memmap) of the augmented minority cache."""
    with open(AUG_META_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("cid", "patient", "label", "crackle", "wheeze", "fb_off", "fb_len", "aug_of", "aug_k"):
            r[k] = int(r[k])
    return rows, np.load(AUG_FBANK_NPY, mmap_mode="r")


# ----------------------------------------------------------------------------- access
_FB = None


def load_fbank():
    global _FB
    if _FB is None:
        _FB = np.load(FBANK_NPY, mmap_mode="r")
    return _FB


def load_meta():
    with open(META_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("cid", "patient", "label", "crackle", "wheeze", "fb_off", "fb_len"):
            r[k] = int(r[k])
        for k in ("start", "end", "dur"):
            r[k] = float(r[k])
    return rows


def get_fbank(r):
    """Raw (un-normalised) log-mel [T,128] float32 of one cycle."""
    return np.asarray(load_fbank()[r["fb_off"]:r["fb_off"] + r["fb_len"]], dtype=np.float32)


PAD = os.environ.get("GF_PAD", "floor")   # GF_PAD=loop: short cycles repeated to fill the window (user, 2026-10-22)


def crop_pad(fb, n=WIN, rng=None):
    """-> (x [n,128], mask [n] bool). Long cycles: random crop if rng else centre crop. Short cycles:
    right-padded with the cycle's own minimum (a 'silence' floor, not a repeat); mask marks real frames.
    With GF_PAD=loop, short cycles are tiled (repeated) to n frames instead and the mask is all True."""
    T = len(fb)
    if T >= n:
        s = rng.randint(0, T - n) if rng is not None else (T - n) // 2
        return fb[s:s + n], np.ones(n, dtype=bool)
    if PAD == "loop":
        return np.tile(fb, (int(np.ceil(n / T)), 1))[:n].astype(np.float32), np.ones(n, dtype=bool)
    x = np.full((n, fb.shape[1]), fb.min(), dtype=np.float32)
    x[:T] = fb
    m = np.zeros(n, dtype=bool)
    m[:T] = True
    return x, m


def get_wave(r):
    import train_optimal_pipeline as tp
    w = np.load(tp.wave_path(r["rec_id"]), mmap_mode="r")
    s, e = int(r["start"] * SR), int(r["end"] * SR)
    return np.array(w[s:e], dtype=np.float32)


# ----------------------------------------------------------------------------- stats report
def write_stats(rows, cycles=None, recs=None):
    L = []
    def table(title, rowkey, colkey, cols, subset, row_names=None):
        L.append("\n### %s\n" % title)
        L.append("| | " + " | ".join(cols) + " | total |")
        L.append("|---" * (len(cols) + 2) + "|")
        for rk in (row_names or sorted({rowkey(r) for r in subset})):
            vals = [sum(1 for r in subset if rowkey(r) == rk and colkey(r) == c) for c in cols]
            L.append("| %s | " % rk + " | ".join(map(str, vals)) + " | %d |" % sum(vals))
    cname = lambda r: CLASS_NAMES[r["label"]]
    L.append("# GAN_FINETUNING data (train split only)\n")
    L.append("Cache: `data/meta.csv` + `data/fbank.npy` (raw AST log-mel, 128 mel x 10 ms, float16).")
    for gs in ("train", "val"):
        sub = [r for r in rows if r["gan_split"] == gs]
        L.append("\n## gan_split = %s: %d cycles, %d patients" % (gs, len(sub), len({r["patient"] for r in sub})))
        table("class x device", cname, lambda r: r["device"], DEVICES, sub, CLASS_NAMES)
        table("class x region", cname, lambda r: r["region"], REGIONS, sub, CLASS_NAMES)
    L.append("\n## cycle length (frames, 10 ms) and window coverage (WIN=%d)\n" % WIN)
    L.append("| class | mean | p50 | p95 | max | % > WIN |")
    L.append("|---|---|---|---|---|---|")
    for k, n in enumerate(CLASS_NAMES):
        t = np.array([r["fb_len"] for r in rows if r["label"] == k])
        L.append("| %s | %.0f | %.0f | %.0f | %d | %.1f |" % (n, t.mean(), np.median(t), np.percentile(t, 95),
                                                              t.max(), 100 * np.mean(t > WIN)))
    L.append("\n## patient concentration (top-5 share of each minority class)\n")
    for k in (1, 2, 3):
        c = Counter(r["patient"] for r in rows if r["label"] == k)
        L.append("- %s: %d patients, top-5 = %.0f%% (%s)" % (CLASS_NAMES[k], len(c),
                 100 * sum(v for _, v in c.most_common(5)) / sum(c.values()), c.most_common(5)))
    if cycles is not None:
        L.append("\n## official TEST split (counts only, for choosing what to generate)\n")
        test = [dict(label=c["label"], device=recs[c["ridx"]]["equip"]) for c in cycles if c["split"] == "test"]
        table("test class x device", cname, lambda r: r["device"], DEVICES, test, CLASS_NAMES)
    with open(os.path.join(DATA, "data_stats.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "aug":
        build_aug(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        build()
