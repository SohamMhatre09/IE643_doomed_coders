"""
hfreal.py -- HF_Lung_V1 breathing cycles as REAL AST training data (2026-11-20; the user lifted the "HF only for the
GAN" rule for this test). Question: does HF help the AST when the AST has the device-adversarial head (S1 / DAT)?
The teammate's Round 24 tested HF only on a plain AST (no DAT).

Selection follows the teammate's Round 24 result (best HF handling on validation):
  - breathing cycles (inhalation onset -> next inhalation onset; hfdata.py), capped to the 8 s AST window
  - DROP every cycle that overlaps Rhonchi or Stridor
  - crackle if >= 0.02 s of D inside the cycle, wheeze if >= 0.10 s of Wheeze; both devices; normals kept
  - N cycles drawn with ICBHI's official-train class ratio (2057 / 1210 / 501 / 363), so the class prior is unchanged
Each cycle: 4 kHz -> 16 kHz (resample_poly x4), tiled to 8 s exactly like sg_exp's real cycles, then the same
ASTFeatureExtractor. Device label for DAT: steth_ files = Litt3200 (same stethoscope as ICBHI's), trunc_ = HF-Type-1.
Cached in HF_GAN/data/hfreal_n<N>_f<frames>.npz.

    $PY HF_GAN/hfreal.py 4000 798
"""
import os
import sys
import csv
import collections
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P
ICBHI_TRAIN = np.array([2057, 1210, 501, 363], dtype=float)
SEG = 8 * 16000


def path(n, frames):
    return os.path.join(_P.sub("hf_data"), "hfreal_n%d_f%d.npz" % (n, frames))


def select(n, seed=0):
    meta = {m["file"]: m for m in csv.DictReader(open(os.path.join(_P.OUT, "hf_data", "full_meta.csv")))}
    lab = np.load(os.path.join(_P.OUT, "hf_data", "full_labels.npy"), mmap_mode="r")      # [frames, D/Wheeze/Rhonchi/Stridor]
    rows = [r for r in csv.DictReader(open(os.path.join(_P.OUT, "gan_data", "hf_meta.csv"))) if r["kind"] == "cycle"]
    by = collections.defaultdict(list)
    n_drop = 0
    for r in rows:
        m = meta[r["file"]]
        o, s, e = int(m["off"]), int(r["start"]), min(int(r["end"]), int(r["start"]) + 800)
        L = np.asarray(lab[o + s:o + e])
        if L[:, 2].any() or L[:, 3].any():
            n_drop += 1
            continue
        c, w = int(L[:, 0].sum() >= 2), int(L[:, 1].sum() >= 10)
        by[c + 2 * w].append((r["file"], m["hf_split"], s, e, "Litt3200" if r["file"].startswith("steth_") else "HF-Type-1"))
    rng = np.random.RandomState(seed)
    want = np.round(n * ICBHI_TRAIN / ICBHI_TRAIN.sum()).astype(int)
    pick, ys = [], []
    for k in range(4):
        idx = rng.choice(len(by[k]), min(want[k], len(by[k])), replace=False)
        pick += [by[k][i] for i in idx]; ys += [k] * len(idx)
    print("HF cycles: %d, dropped (rhonchi/stridor) %d, available per class %s, taken %s"
          % (len(rows), n_drop, [len(by[k]) for k in range(4)], np.bincount(ys, minlength=4).tolist()), flush=True)
    return pick, np.array(ys)


_FE = None


def _feat(args):
    global _FE
    import soundfile as sf
    from scipy.signal import resample_poly
    if _FE is None:
        import torch
        torch.set_num_threads(1)
        from transformers import ASTFeatureExtractor
        _FE = ASTFeatureExtractor.from_pretrained(_P.AST_MODEL, local_files_only=_P.OFFLINE)
    (f, split, s, e, _), frames = args
    _FE.max_length = frames
    x, sr = sf.read(os.path.join(_P.HF_DIR, split, f + ".wav"), dtype="float32")
    x = resample_poly(x, 4, 1).astype(np.float32)
    seg = x[s * 160:max(e * 160, s * 160 + 1600)]
    if len(seg) < SEG:
        seg = np.tile(seg, int(np.ceil(SEG / len(seg))))
    seg = seg[:SEG]
    return _FE(seg, sampling_rate=16000, return_tensors="np")["input_values"][0].astype(np.float16)


def build(n, frames):
    pick, y = select(n)
    os.environ["OMP_NUM_THREADS"] = "1"
    with get_context("spawn").Pool(40) as pool:
        X = pool.map(_feat, [(p, frames) for p in pick], chunksize=8)
    np.savez(path(n, frames), x=np.stack(X), y=y, devs=np.array([p[4] for p in pick]))
    print("wrote", path(n, frames), np.stack(X).shape, flush=True)


def load(n, frames):
    """-> (x float16 [n, frames, 128], y, device names); builds the cache on first use (lock file for parallel seeds)."""
    f = path(n, frames)
    lock = f + ".lock"
    if not os.path.exists(f):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            build(n, frames)
            os.remove(lock)
        except FileExistsError:
            import time
            while not os.path.exists(f) or os.path.exists(lock):
                time.sleep(20)
    z = np.load(f)
    return z["x"], z["y"], [str(d) for d in z["devs"]]


if __name__ == "__main__":
    build(int(sys.argv[1]), int(sys.argv[2]))
