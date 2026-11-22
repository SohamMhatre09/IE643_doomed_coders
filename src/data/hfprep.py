"""
hfprep.py -- HF_Lung_V1 data for the HF -> GAN plan (2026-11-08). HF_Lung is used ONLY to improve the GAN; the AST
classifier never sees it.

Writes two things:
  1. GAN_FINETUNING/data/hf_clean.csv: for every row of the existing GAN cache (hf_meta.csv) whether its "wheeze" flag
     comes from a real Wheeze event or only from Rhonchi (hfdata.py mapped Wheeze OR Rhonchi -> wheeze; 33-34% of HF
     "wheeze" / "both" were rhonchi only = label noise). gan.py --hf-clean drops the rhonchi-only rows.
  2. HF_GAN/data/full_fbank.npy + full_labels.npy + full_meta.csv: the whole 15 s recordings (raw AST log-mel at 16 kHz,
     exactly like hfdata.py / gfdata.fbank) with FRAME labels (10 ms) for the event detector (eventdet.py):
     channels D (crackle), Wheeze, Rhonchi, Stridor.

    $PY HF_GAN/hfprep.py
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
import hfdata as H                                  # noqa: E402

import na_paths as _P
OUT = _P.sub("hf_data")
CH = ["D", "Wheeze", "Rhonchi", "Stridor"]


def events(path):
    ev = collections.defaultdict(list)
    for line in open(path, errors="replace"):
        t = line.split()
        if len(t) >= 3:
            ev[t[0]].append((int(H.secs(t[1]) * 100), int(H.secs(t[2]) * 100)))
    return ev


def process(args):
    import torch
    torch.set_num_threads(1)
    import soundfile as sf
    from scipy.signal import resample_poly
    wav, split = args
    name = os.path.basename(wav)[:-4]
    x, sr = sf.read(wav, dtype="float32")
    assert sr == 4000
    fb = G.fbank(resample_poly(x, 4, 1).astype(np.float32)).astype(np.float16)     # same as hfdata.process
    ev = events(wav[:-4] + "_label.txt")
    lab = np.zeros((len(fb), len(CH)), dtype=np.uint8)
    for j, k in enumerate(CH):
        for s, e in ev[k]:
            lab[max(0, s):min(len(fb), e), j] = 1
    dev = "Litt3200" if name.startswith("steth_") else "HF-Type-1"
    return dict(file=name, hf_split=split, patient=H.patient_of(name), device=dev), fb, lab, ev


def main():
    os.nice(10)
    wavs = [(w, sp) for sp in ("train", "test") for w in sorted(glob.glob(os.path.join(H.HF_ROOT, sp, "*.wav")))]
    metas, fbs, labs, evs = [], [], [], {}
    with Pool(48) as pool:
        for m, fb, lab, ev in pool.imap(process, wavs, chunksize=16):
            metas.append(m); fbs.append(fb); labs.append(lab); evs[m["file"]] = ev
    off = 0
    for m, fb in zip(metas, fbs):
        m.update(off=off, n=len(fb)); off += len(fb)
    np.save(os.path.join(OUT, "full_fbank.npy"), np.concatenate(fbs))
    np.save(os.path.join(OUT, "full_labels.npy"), np.concatenate(labs))
    with open(os.path.join(OUT, "full_meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0]))
        w.writeheader(); w.writerows(metas)
    print("full recordings: %d, %d frames; positive frame share %s" % (
        len(metas), off, dict(zip(CH, np.concatenate(labs).mean(0).round(4).tolist()))))
    # 1. rhonchi-only flag for every GAN-cache row
    rows, _ = H.load_hf()
    out, n_ro = [], collections.Counter()
    for r in rows:
        ev = evs[r["file"]]
        ov = lambda k: any(s < r["end"] and e > r["start"] for s, e in ev[k])
        true_w = int(ov("Wheeze"))
        ro = int(r["wheeze"] == 1 and not true_w)
        n_ro[(r["kind"], r["label"], ro)] += 1
        out.append(dict(hid=r["hid"], wheeze_true=true_w, rhonchi_only=ro))
    with open(os.path.join(G.DATA, "hf_clean.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hid", "wheeze_true", "rhonchi_only"])
        w.writeheader(); w.writerows(out)
    for kind in ("cycle", "event"):
        for k in (2, 3):
            a, b = n_ro[(kind, k, 0)], n_ro[(kind, k, 1)]
            print("%s %s: %d with a real Wheeze, %d rhonchi only (%.0f%%)" % (kind, G.CLASS_NAMES[k], a, b, 100 * b / max(1, a + b)))


if __name__ == "__main__":
    main()
