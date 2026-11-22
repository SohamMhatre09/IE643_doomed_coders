"""
bgnoise.py -- real background noise for synthetic cycles (Round 15 noise_mix.py, ported to the 8 s S1 pipeline).

Why: GAN fakes are too smooth; real recordings carry a noise floor (stethoscope hiss, friction, room, device).
The noise is added AFTER generation (no GAN retraining), to the final AST input windows.

Differences from Round 15 (gan_testing/round15/noise_mix.py), forced by this pipeline:
  * Round 15 windows are 3 s crops (298 frames); here the AST input is 8 s (frames_for(8) = 798 frames) and the
    GAN's 512-frame cycle is tiled to it (ast_gan.to_ast). The noise is mixed into the whole 798-frame window AFTER
    tiling, with 798 freshly drawn quiet frames, so the noise floor does not repeat along with the tiled cycle.
  * Bank source: the RAW cycles of GAN_FINETUNING/data (always data/, also when GF_DATA points at another cache:
    the real AST training data is raw audio), gan_split == train only (official train minus the 12 GAN validation
    patients; the official test split is not in this cache at all). One entry per CYCLE (Round 15: per 3 s crop).
Same as Round 15: quietest 20 % of a cycle's frames by frame energy E_t = logsumexp over the 128 mel bands of the
raw log-mel, up to 30 frames per cycle; per synthetic item one random bank cycle, frames drawn with replacement in
random order, mixed in the POWER domain:  L_mix = log(exp(L_fake) + g * exp(L_noise)),  g ~ log-uniform [0.2, 0.7],
L = raw Kaldi log-mel = 2 * 4.5689974 * a - 4.2677393 (a = AST input). Mixed once per item (fixed, not per epoch).

    $PY GAN_FINETUNING/bgnoise.py            # builds GAN_FINETUNING/noise_bank/noise_bank.npz (+ .json); else lazy
"""
import os
import csv
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import na_paths as _P
RAW = os.path.join(_P.OUT, "gan_data")                # the gfdata log-mel cache (ICBHI training cycles)
BANK = os.path.join(_P.OUT, "noise_bank", "noise_bank.npz")
AST_MEAN, AST_STD = -4.2677393, 4.5689974
QUIET_FRAC, PER_CYCLE, GAIN = 0.2, 30, (0.2, 0.7)

_NB = None


def to_L(a):
    return np.asarray(a, np.float32) * (2 * AST_STD) + AST_MEAN


def to_a(L):
    return ((L - AST_MEAN) / (2 * AST_STD)).astype(np.float32)


def build(path=BANK):
    rows = [r for r in csv.DictReader(open(os.path.join(RAW, "meta.csv"))) if r["gan_split"] == "train"]
    fb = np.load(os.path.join(RAW, "fbank.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    frames, cnt, rec = [], [], []
    for r in rows:
        o, n = int(r["fb_off"]), int(r["fb_len"])
        L = np.asarray(fb[o:o + n], np.float32)                           # raw log-mel [T,128]
        m = L.max(1, keepdims=True)
        E = np.log(np.exp(L - m).sum(1)) + m[:, 0]                        # logsumexp over mel
        q = np.where(E <= np.quantile(E, QUIET_FRAC))[0]
        q = rng.choice(q, min(PER_CYCLE, len(q)), replace=False)
        frames.append(L[q].astype(np.float16)); cnt.append(len(q)); rec.append(r["rec_id"])
    cnt = np.array(cnt); off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, frames=np.concatenate(frames), off=off, cnt=cnt, rec=np.array(rec))
    info = dict(cycles=int(len(cnt)), frames=int(cnt.sum()), recordings=int(len(set(rec))), quiet_frac=QUIET_FRAC,
                per_cycle=PER_CYCLE, gain=list(GAIN), source="GAN_FINETUNING/data (raw), gan_split=train only; "
                "GAN validation patients and the official test split never used", domain="raw log-mel")
    json.dump(info, open(path.replace(".npz", ".json"), "w"), indent=1)
    print("[bgnoise] bank %s -> %s" % (info, path), flush=True)
    return info


def bank():
    global _NB
    if _NB is None:
        if not os.path.exists(BANK):
            build()
        z = np.load(BANK)
        _NB = (z["frames"], z["off"], z["cnt"], np.array([str(r).split("_")[-1] for r in z["rec"]]))
    return _NB


def mix(x, seed, devices=None):
    """x: AST inputs [N,T,128] (one window per synthetic item) -> same shape/dtype with real background added"""
    fr, off, cnt, bdev = bank()
    rng = np.random.default_rng(seed + 424242)
    out = np.empty_like(x)
    lg_lo, lg_hi = np.log(GAIN[0]), np.log(GAIN[1])
    for i in range(len(x)):
        if devices is not None and devices[i]:          # device-matched noise (2026-11-07): a bank cycle of that device
            pool = np.where(bdev == devices[i])[0]
            j = pool[rng.integers(len(pool))]
        else:
            j = rng.integers(len(cnt))
        Ln = fr[off[j]:off[j] + cnt[j]].astype(np.float32)[rng.integers(0, cnt[j], x.shape[1])]   # [T,128] raw
        g = rng.uniform(lg_lo, lg_hi)
        out[i] = to_a(np.logaddexp(to_L(x[i]), g + Ln)).astype(x.dtype)
    return out


if __name__ == "__main__":
    build()
