"""
hfdata.py -- HF_Lung_V1 pretraining cache for the GAN (README E18, design decisions D1-D5 of 2026-10-21).

Source: external/HF_Lung_V1 (Hsu et al., PLOS ONE 2021, CC BY 4.0): 9,765 x 15 s, 4 kHz, Littmann 3200 ("steth_")
and HF-Type-1 ("trunc_") recordings with event labels `type start end` (I, E, D = crackle, Wheeze, Rhonchi, Stridor).
All of it (its own train + test) is used for pretraining only: no patient overlaps ICBHI.

  D4  4 kHz -> 16 kHz (resample_poly x4), then gfdata.fbank: the same raw AST log-mel as the ICBHI cache. Content
      stops at 2 kHz, exactly like ICBHI's own Littmann 3200 files (also 4 kHz).
  D1  two kinds of samples per recording:
        cycle  one breathing cycle = start of an inhalation to the start of the next (the last one: to the end of
               the last exhalation); 0.5-10.24 s kept; variable length like ICBHI cycles (crop_pad at training time)
        event  a 512-frame (5.12 s) window around every crackle / wheeze / rhonchi event, with the event at a RANDOM
               position inside the window (not always centred); windows starting < 1 s from an already kept window
               of the same recording are skipped
  D2  labels from overlapping events: crackle = a D event; wheeze = a Wheeze or Rhonchi event (rhonchi are low-
      pitched continuous sounds, inside ICBHI's "wheeze"); both = both; samples overlapping Stridor are dropped
  D5  patient = recording date (the dataset README: same date = very likely the same subject)

Writes data/hf_meta.csv + data/hf_fbank.npy (float16, flat) and hf_data_stats.md.
Usage: $PY GAN_FINETUNING/hfdata.py
"""
import os
import sys
import csv
import glob
import collections
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import gfdata as G                                  # noqa: E402

import na_paths as _P
HF_ROOT = _P.HF_DIR
META_CSV = os.path.join(G.DATA, "hf_meta.csv")
FBANK_NPY = os.path.join(G.DATA, "hf_fbank.npy")
WIN, MIN_CYC, MAX_CYC, MIN_GAP = 512, 50, 1024, 100          # frames (10 ms)
CLASS_NAMES = G.CLASS_NAMES


def secs(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def patient_of(name):
    """steth_yyyymmdd_HH_MM_ss / trunc_yyyy-mm-dd-HH-MM-ss-LX_N -> 'yyyymmdd' (same date = same subject)."""
    p = name.split("_")
    return p[1] if name.startswith("steth_") else p[1][:10].replace("-", "")


def label_of(ev, a, b):
    """Class of frames [a, b) from the overlapping events; None if it overlaps stridor."""
    ov = lambda k: any(s < b and e > a for s, e in ev[k])
    if ov("Stridor"):
        return None
    c, w = ov("D"), ov("Wheeze") or ov("Rhonchi")
    return 3 if c and w else 1 if c else 2 if w else 0


def _one_thread():
    """One torch thread per worker: torch otherwise starts ~100 threads per process and the workers fight."""
    import torch
    torch.set_num_threads(1)


def process(args):
    wav, split, seed = args
    import soundfile as sf
    from scipy.signal import resample_poly
    name = os.path.basename(wav)[:-4]
    x, sr = sf.read(wav, dtype="float32")
    assert sr == 4000, (wav, sr)
    fb = G.fbank(resample_poly(x, 4, 1).astype(np.float32))                     # [~1498, 128]
    T = len(fb)
    ev = collections.defaultdict(list)
    for line in open(wav[:-4] + "_label.txt", errors="replace"):
        t = line.split()
        if len(t) >= 3:
            ev[t[0]].append((int(secs(t[1]) * 100), int(secs(t[2]) * 100)))
    dev = "Litt3200" if name.startswith("steth_") else "HF-Type-1"
    base = dict(file=name, hf_split=split, patient=patient_of(name), device=dev)
    rows, feats = [], []
    # cycles: inhalation start -> next inhalation start (last: -> end of the last exhalation)
    ins = sorted(s for s, e in ev["I"])
    ends = ins[1:] + [max(e for s, e in ev["E"])] if ev["E"] else ins[1:] + [None]
    for s, e in zip(ins, ends):
        if e is None or not (MIN_CYC <= e - s <= MAX_CYC) or e > T:
            continue
        lab = label_of(ev, s, e)
        if lab is not None:
            rows.append(dict(base, kind="cycle", start=s, end=e, label=lab))
            feats.append(fb[s:e])
    # event windows: 512 frames around each crackle / wheeze / rhonchi event, event at a random position
    rng = np.random.RandomState(seed)
    kept = []
    for k in ("D", "Wheeze", "Rhonchi"):
        for a, b in sorted(ev[k]):
            b = min(b, a + WIN)
            lo, hi = max(0, b - WIN), min(a, T - WIN)
            if hi < lo:
                continue
            st = rng.randint(lo, hi + 1)
            if any(abs(st - q) < MIN_GAP for q in kept):
                continue
            lab = label_of(ev, st, st + WIN)
            if lab is None:
                continue
            kept.append(st)
            rows.append(dict(base, kind="event", start=st, end=st + WIN, label=lab))
            feats.append(fb[st:st + WIN])
    return rows, [f.astype(np.float16) for f in feats]


def build():
    wavs = [(w, sp) for sp in ("train", "test") for w in sorted(glob.glob(os.path.join(HF_ROOT, sp, "*.wav")))]
    jobs = [(w, sp, i) for i, (w, sp) in enumerate(wavs)]
    rows, feats = [], []
    os.nice(10)                                          # shared machine: yield CPU to other users
    with Pool(48, initializer=_one_thread) as pool:      # user: up to 50-60 cores for minutes-long jobs
        for r, f in pool.imap(process, jobs, chunksize=16):
            rows += r
            feats += f
    off = 0
    for i, (r, f) in enumerate(zip(rows, feats)):
        r.update(hid=i, crackle=int(r["label"] in (1, 3)), wheeze=int(r["label"] in (2, 3)), fb_off=off, fb_len=len(f))
        off += len(f)
    np.save(FBANK_NPY, np.concatenate(feats))
    with open(META_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["hid", "file", "hf_split", "patient", "device", "kind", "start", "end",
                                           "label", "crackle", "wheeze", "fb_off", "fb_len"])
        w.writeheader()
        w.writerows(rows)
    print("HF_Lung cache: %d samples from %d recordings, %d frames -> %s" % (len(rows), len(wavs), off, G.DATA))
    write_stats(rows)


HOLDOUT_FRAC, HOLDOUT_SEED = 0.10, 0


def holdout_patients():
    """~10% of HF_Lung patients (recording dates), never used for GAN training: the reference for HF-style samples
    (incl. Littmann 3200, which the ICBHI hold-out lacks) and for the HF_Lung copy check."""
    with open(META_CSV) as f:
        dev_of = collections.defaultdict(collections.Counter)
        for r in csv.DictReader(f):
            dev_of[r["patient"]][r["device"]] += 1
    rng = np.random.RandomState(HOLDOUT_SEED)
    hold = set()
    for d in ("Litt3200", "HF-Type-1"):                    # stratified: ~10% of each device's patients
        pats = sorted(p for p, c in dev_of.items() if c.most_common(1)[0][0] == d)
        hold |= set(rng.choice(pats, max(1, int(round(HOLDOUT_FRAC * len(pats)))), replace=False).tolist())
    return hold


def load_hf():
    with open(META_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("hid", "start", "end", "label", "crackle", "wheeze", "fb_off", "fb_len"):
            r[k] = int(r[k])
    return rows, np.load(FBANK_NPY, mmap_mode="r")


def write_stats(rows):
    L = ["# HF_Lung_V1 pretraining cache (hfdata.py)\n",
         "%d samples, %d patients (recording dates). ICBHI GAN-train minority for comparison: crackle 1065, wheeze 425, "
         "both 310.\n" % (len(rows), len({r["patient"] for r in rows}))]
    for kind in ("cycle", "event"):
        sub = [r for r in rows if r["kind"] == kind]
        L += ["\n## %s samples: %d\n" % (kind, len(sub)), "| class | Litt3200 | HF-Type-1 | total |", "|---|---|---|---|"]
        for k, n in enumerate(CLASS_NAMES):
            a = sum(1 for r in sub if r["label"] == k and r["device"] == "Litt3200")
            b = sum(1 for r in sub if r["label"] == k and r["device"] == "HF-Type-1")
            L.append("| %s | %d | %d | %d |" % (n, a, b, a + b))
        lens = np.array([r["fb_len"] for r in sub])
        if len(lens):
            L.append("\nlength (frames): mean %.0f, p50 %.0f, p95 %.0f" % (lens.mean(), np.median(lens), np.percentile(lens, 95)))
    open(os.path.join(G.DATA, "hf_data_stats.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    build()
