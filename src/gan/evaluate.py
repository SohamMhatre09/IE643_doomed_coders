"""
evaluate.py -- evaluation suite for synthetic minority cycles (Step 2).

A "set" is raw AST log-mel x [N,512,128] (gfdata scale) + labels y in {1 crackle, 2 wheeze, 3 both}.
Reference = held-out real val patients (gan_split=val); memorisation is checked against gan_split=train.

Tier A -- does it look/sound like real cycles (per class, vs val):
  desc_W     mean Wasserstein-1 / val-std over 8 clip descriptors (spectral centroid, spread, skewness,
             kurtosis, entropy, flatness, temporal transientness, tonality)            lower = better
  dtw        median over samples of min DTW (MFCC sequences) to 30 same-class val clips; kept only for
             comparison with the cGAN paper -- the controls show it rates blurred cycles as most real
  (the paper's MFCC correlation was dropped: 0.998-0.999 for every control, pure noise included)
Tier B -- is it useful. Two embedding spaces:
  judge = AST fine-tuned on gan-train real (judge.py): sensitive to the GAN's training cycles, so used for
          class fidelity and copy detection only;
  AS    = frozen AudioSet AST (never saw ICBHI): unbiased space for the distribution metrics.
  fid        judge recall of the intended class, macro. Unseen real patients score ~51 (ref line); training
             cycles score ~98, so read it together with copy_emb                         higher = better
  FD         Frechet distance, AS-embedding PCA-32, vs val                               lower = better
  dens / cov density & coverage (Naeem et al. 2020, k=5) in the same space; cov is the robust one
  probe_auc  val-real vs set logistic probe on AS embeddings, 5-fold CV (0.5 = indistinguishable)
  tstr       logistic regression on AS embeddings trained on the set, tested on val real,
             balanced accuracy over 3 classes (chance 33.3)                              higher = better
  copy_emb / copy_pix  share of samples closer to a gan-train cycle than 95% of unseen val cycles are
             (judge embedding / pooled log-mel; copy_pix also fires on blurry sets)       lower = better
  div        median within-set nearest-neighbour distance / the same for val (AS space)  ~1 = good, ~0 = duplicates

Usage:
  $PY GAN_FINETUNING/evaluate.py --controls                 # sanity-check the suite on known sets
  $PY GAN_FINETUNING/evaluate.py --samples run/samples.npz  # npz with x [N,512,128], y [N]
Results are appended to GAN_FINETUNING/eval/results.csv and printed as a table.
"""
import os
import sys
import csv
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct
from scipy.stats import wasserstein_distance
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gfdata as G                                  # noqa: E402
import judge as J                                   # noqa: E402

RESULTS_CSV = os.path.join(J.EVAL_DIR, "results.csv")
MINOR = (1, 2, 3)
N_PER_CLASS = 300
DESC = ["centroid", "spread", "skewness", "kurtosis", "entropy", "flatness", "transient", "tonality"]
COLS = ["fid", "fid_crackle", "fid_wheeze", "fid_both", "desc_W", "dtw", "FD", "dens", "cov",
        "probe_auc", "tstr", "copy_emb", "copy_pix", "div"]


# ============================================================================ Tier A
def _mel_centres():
    lo, hi = 1127 * np.log(1 + 20 / 700), 1127 * np.log(1 + 8000 / 700)      # kaldi fbank defaults
    m = np.linspace(lo, hi, G.N_MELS + 2)[1:-1]
    return 700 * (np.exp(m / 1127) - 1)


FC = _mel_centres()


def frame_mask(x):
    """Real frames = not the constant padding floor (zero spread across mel bins)."""
    m = x.std(axis=1) > 0.05
    return m if m.sum() >= 8 else np.ones(len(x), dtype=bool)


def descriptors(x):
    """x [T,128] raw log-mel -> 8 clip-level descriptors (means over real frames)."""
    x = x[frame_mask(x)]
    P = np.exp(np.clip(x, -30, 30))
    p = P / P.sum(1, keepdims=True)
    c = (p * FC).sum(1)
    sp = np.sqrt((p * (FC - c[:, None]) ** 2).sum(1)) + 1e-9
    sk = (p * (FC - c[:, None]) ** 3).sum(1) / sp ** 3
    ku = (p * (FC - c[:, None]) ** 4).sum(1) / sp ** 4
    en = -(p * np.log(p + 1e-12)).sum(1) / np.log(G.N_MELS)
    fl = np.exp(np.log(P + 1e-12).mean(1)) / (P.mean(1) + 1e-12)
    e = np.log(P.sum(1) + 1e-12)
    d = np.diff(e)
    tr = ((d - d.mean()) ** 4).mean() / (d.var() ** 2 + 1e-12) if len(d) > 3 else 0.0   # crackle-like bursts
    ton = (x.max(1) - np.median(x, 1)).mean()                                          # wheeze-like peaks
    return np.array([c.mean(), sp.mean(), sk.mean(), ku.mean(), en.mean(), fl.mean(), tr, ton])


def mfcc(x, n=20):
    return dct(x[frame_mask(x)], type=2, axis=1, norm="ortho")[:, :n]


@njit(cache=True)
def _dtw(a, b):
    n, m = a.shape[0], b.shape[0]
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.sqrt(((a[i - 1] - b[j - 1]) ** 2).sum())
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


def tier_a(xs, ys, ref, rng):
    out = {}
    W, T = [], []
    for k in MINOR:
        a, r = xs[ys == k], ref["x"][ref["y"] == k]
        da = np.stack([descriptors(v) for v in a])
        dr = ref["desc"][ref["y"] == k]
        W.append(np.mean([wasserstein_distance(da[:, i], dr[:, i]) / (dr[:, i].std() + 1e-9) for i in range(len(DESC))]))
        ra = [mfcc(v)[::2].astype(np.float64) for v in r]
        sel = rng.choice(len(a), min(60, len(a)), replace=False)
        T.append(np.median([min(_dtw(mfcc(a[i])[::2].astype(np.float64), ra[j])
                                for j in rng.choice(len(ra), min(30, len(ra)), replace=False)) for i in sel]))
    out.update(desc_W=np.mean(W), dtw=np.mean(T))
    return out


# ============================================================================ Tier B
def frechet(a, b):
    from scipy.linalg import sqrtm
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = sqrtm(s1 @ s2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def dens_cov(real, fake, k=5):
    """Density & coverage (Naeem et al. 2020): robust to outliers and to collapsed sets."""
    dr = torch.cdist(torch.as_tensor(real), torch.as_tensor(real))
    rad = dr.sort(1).values[:, k]                                       # k-NN radius of each real point
    d = torch.cdist(torch.as_tensor(fake), torch.as_tensor(real))       # [F,R]
    inside = d <= rad[None]
    dens = inside.sum().item() / (k * len(fake))
    cov = (d.min(0).values <= rad).float().mean().item()
    return float(dens), float(cov)


def probe_auc(real, fake, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.RandomState(seed)
    n = min(len(real), len(fake))
    X = np.concatenate([real[rng.choice(len(real), n, replace=False)], fake[rng.choice(len(fake), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return float(cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc").mean())


def pool(x):
    """Pooled log-mel (4x4 average) for pixel-space nearest neighbours."""
    t = torch.as_tensor(np.asarray(x, dtype=np.float32))[:, None]
    return F.avg_pool2d(t, 4).flatten(1).numpy()


def nn_dist(a, b, cosine):
    a, b = torch.as_tensor(a).cuda(), torch.as_tensor(b).cuda()
    if cosine:
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        d = 1 - a @ b.T
    else:
        d = torch.cdist(a, b)
    return d.min(1).values.cpu().numpy()


def tstr(zs, ys, zv, yv):
    """Train on the set, test on val real (frozen AudioSet-AST embeddings); balanced accuracy, 3 classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
    clf.fit(zs, ys)
    pred = clf.predict(zv)
    return float(100 * np.mean([np.mean(pred[yv == k] == k) for k in MINOR]))


def nn_within(z):
    z = F.normalize(torch.as_tensor(z), dim=1)
    d = 1 - z @ z.T
    d.fill_diagonal_(float("inf"))
    return d.min(1).values.median().item()


_AS = None


@torch.no_grad()
def embed_as(x, bs=32):
    """Frozen AudioSet AST (no ICBHI training), (CLS+dist)/2 -> [N,768]."""
    global _AS
    if _AS is None:
        import train_optimal_pipeline as tp
        _AS = tp.build_model(G.WIN).audio_spectrogram_transformer.cuda().eval()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(G.ast_normalize(np.asarray(x[i:i + bs], dtype=np.float32)), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            h = _AS(input_values=xb).last_hidden_state
        out.append(((h[:, 0] + h[:, 1]) / 2).float().cpu().numpy())
    return np.concatenate(out)


def tier_b(xs, ys, ref):
    z, p = J.run(xs)
    za = embed_as(xs)
    out = {}
    rec = {k: 100 * np.mean(p[ys == k].argmax(1) == k) for k in MINOR}
    out.update(fid=np.mean(list(rec.values())), fid_crackle=rec[1], fid_wheeze=rec[2], fid_both=rec[3])
    zp = ref["pca"].transform(za)
    out["FD"] = frechet(ref["av_pca"], zp)
    out["dens"], out["cov"] = dens_cov(ref["av_pca"], zp)
    out["probe_auc"] = probe_auc(ref["av"], za)
    out["tstr"] = tstr(za, ys, ref["av"], ref["y"])
    out["copy_emb"] = float(np.mean(nn_dist(z, ref["zt"], True) < ref["tau_emb"]))
    out["copy_pix"] = float(np.mean(nn_dist(pool(xs), ref["pt"], False) < ref["tau_pix"]))
    out["div"] = float(np.mean([nn_within(za[ys == k]) / ref["div_ref"][k] for k in MINOR]))
    return out


CLASS_COLS = ["fid", "FD", "cov", "probe_auc", "div", "copy_emb", "copy_pix", "maxval"]
CLASS_CSV = os.path.join(J.EVAL_DIR, "results_class.csv")


def evaluate_class(name, xs, k, ref, note=""):
    """Single-class set (e.g. a per-class specialist GAN) against that class's held-out real cycles only:
    judge recall of class k, FD / coverage / probe in the frozen-AST space vs class-k real, diversity vs class-k real,
    copy rates, and the largest raw value (exploded samples). Rows -> eval/results_class.csv."""
    xs = np.asarray(xs, dtype=np.float32)
    z, p = J.run(xs)
    za = embed_as(xs)
    m = ref["y"] == k
    zp = ref["pca"].transform(za)
    out = dict(name=name, k=k, n=len(xs), fid=100 * float(np.mean(p.argmax(1) == k)),
               FD=frechet(ref["av_pca"][m], zp), cov=dens_cov(ref["av_pca"][m], zp)[1],
               probe_auc=probe_auc(ref["av"][m], za), div=nn_within(za) / ref["div_ref"][k],
               copy_emb=float(np.mean(nn_dist(z, ref["zt"], True) < ref["tau_emb"])),
               copy_pix=float(np.mean(nn_dist(pool(xs), ref["pt"], False) < ref["tau_pix"])),
               maxval=float(xs.max()), note=note)
    new = not os.path.exists(CLASS_CSV)
    with open(CLASS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "k", "n"] + CLASS_COLS + ["note"])
        if new:
            w.writeheader()
        w.writerow({c: out[c] for c in ["name", "k", "n"] + CLASS_COLS + ["note"]})
    print("[%s] class %d: %s" % (name, k, {c: round(out[c], 3) for c in CLASS_COLS}), flush=True)
    return out


# ============================================================================ reference
def build_reference():
    from sklearn.decomposition import PCA
    xv, yv, _ = J.real_set("val", MINOR)
    xt, yt, _ = J.real_set("train", MINOR)
    zv, pv = J.run(xv)
    zt, _ = J.run(xt)
    av, at = embed_as(xv), embed_as(xt)
    pca = PCA(32, random_state=0).fit(at)
    pt, pvp = pool(xt), pool(xv)
    ref = dict(x=xv, y=yv, zv=zv, zt=zt, pt=pt, pca=pca, av=av, av_pca=pca.transform(av), xt=xt, yt=yt,
               div_ref={k: nn_within(av[yv == k]) for k in MINOR},
               desc=np.stack([descriptors(v) for v in xv]),
               tau_emb=float(np.percentile(nn_dist(zv, zt, True), 5)),
               tau_pix=float(np.percentile(nn_dist(pvp, pt, False), 5)),
               judge_val_recall={k: 100 * np.mean(pv[yv == k].argmax(1) == k) for k in MINOR})
    return ref


HF_RESULTS_CSV = os.path.join(J.EVAL_DIR, "results_hf.csv")
HF_REF_NPZ = os.path.join(J.EVAL_DIR, "hf_ref_train_emb.npz")
HF_COLS = ["copy_hf_icbhi_style", "copy_hf_hf_style", "hf_desc_W", "hf_FD", "hf_cov", "hf_probe", "hf_tstr",
           "hf_div", "hf_FD_litt3200", "hf_probe_litt3200"]


def build_hf_reference(val_per_class=300, seed=0):
    """HF_Lung reference (README E19): real minority samples of the held-out HF patients (hfdata.holdout_patients),
    frozen AudioSet-AST embeddings of ALL HF training minority samples (cached) for the copy check, and the conditions
    (label, device of each held-out sample) used to generate HF-style samples for the comparison."""
    import hfdata
    from sklearn.decomposition import PCA
    rows, fbm = hfdata.load_hf()
    hold = hfdata.holdout_patients()
    win = lambda r: G.crop_pad(np.asarray(fbm[r["fb_off"]:r["fb_off"] + r["fb_len"]], dtype=np.float32))[0]
    tr = [r for r in rows if r["patient"] not in hold and r["label"] in MINOR]
    rng = np.random.RandomState(seed)
    va = []
    for k in MINOR:
        rk = [r for r in rows if r["patient"] in hold and r["label"] == k]
        va += [rk[i] for i in rng.choice(len(rk), min(val_per_class, len(rk)), replace=False)]
    if os.path.exists(HF_REF_NPZ) and int(np.load(HF_REF_NPZ)["n"]) == len(tr):
        at = np.load(HF_REF_NPZ)["at"]
    else:                                                # ~35k windows, embedded once in chunks and cached
        at = np.concatenate([embed_as(np.stack([win(r) for r in tr[i:i + 2048]])) for i in range(0, len(tr), 2048)])
        np.savez(HF_REF_NPZ, at=at, n=len(tr))
    xv = np.stack([win(r) for r in va])
    yv = np.array([r["label"] for r in va])
    dv = np.array([r["device"] for r in va])
    av = embed_as(xv)
    pca = PCA(32, random_state=0).fit(at[rng.choice(len(at), min(8000, len(at)), replace=False)])
    return dict(x=xv, y=yv, dev=dv, av=av, at=at, pca=pca, av_pca=pca.transform(av),
                desc=np.stack([descriptors(v) for v in xv]),
                div_ref={k: nn_within(av[yv == k]) for k in MINOR},
                tau_hf=float(np.percentile(nn_dist(av, at, True), 5)),
                gen_rows=va, gen_y=yv, gen_dev=dv)


def evaluate_hf(name, x_icbhi_style, x_hf_style, y_hf, dev_hf, ref):
    """Copy check against HF_Lung training samples for both sample sets, and HF-style samples vs the HF hold-out
    (overall and Littmann 3200 only). Appends a row to eval/results_hf.csv."""
    zi, zh = embed_as(np.asarray(x_icbhi_style, np.float32)), embed_as(np.asarray(x_hf_style, np.float32))
    xh = np.asarray(x_hf_style, np.float32)
    out = dict(name=name, n=len(y_hf))
    out["copy_hf_icbhi_style"] = float(np.mean(nn_dist(zi, ref["at"], True) < ref["tau_hf"]))
    out["copy_hf_hf_style"] = float(np.mean(nn_dist(zh, ref["at"], True) < ref["tau_hf"]))
    out["hf_desc_W"] = tier_a(xh, y_hf, ref, np.random.RandomState(0))["desc_W"]
    zp = ref["pca"].transform(zh)
    out["hf_FD"] = frechet(ref["av_pca"], zp)
    out["hf_cov"] = dens_cov(ref["av_pca"], zp)[1]
    out["hf_probe"] = probe_auc(ref["av"], zh)
    out["hf_tstr"] = tstr(zh, y_hf, ref["av"], ref["y"])
    out["hf_div"] = float(np.mean([nn_within(zh[y_hf == k]) / ref["div_ref"][k] for k in MINOR]))
    m, mv = dev_hf == "Litt3200", ref["dev"] == "Litt3200"
    out["hf_FD_litt3200"] = frechet(ref["av_pca"][mv], zp[m]) if m.sum() > 40 and mv.sum() > 40 else float("nan")
    out["hf_probe_litt3200"] = probe_auc(ref["av"][mv], zh[m]) if m.sum() > 40 and mv.sum() > 40 else float("nan")
    new = not os.path.exists(HF_RESULTS_CSV)
    with open(HF_RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "n"] + HF_COLS)
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    print("[%s HF] %s" % (name, {k: round(float(out[k]), 3) for k in HF_COLS}), flush=True)
    return out


def evaluate_set(name, xs, ys, ref, note=""):
    t0 = time.time()
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.int64)
    out = dict(name=name, n=len(ys))
    out.update(tier_a(xs, ys, ref, np.random.RandomState(0)))
    out.update(tier_b(xs, ys, ref))
    out["note"] = note
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "n"] + COLS + ["note"])
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    print("[%s] %.0fs %s" % (name, time.time() - t0, {k: round(float(out[k]), 3) for k in COLS}), flush=True)
    return out


def fmt_table(rows):
    L = ["| set | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for r in rows:
        L.append("| %s | " % r["name"] + " | ".join("%.3f" % float(r[c]) if abs(float(r[c])) < 10 else "%.1f" % float(r[c]) for c in COLS) + " |")
    return "\n".join(L)


# ============================================================================ controls
def controls(ref):
    """Known-answer sets drawn from gan-train real minority cycles (N_PER_CLASS per class)."""
    rng = np.random.RandomState(0)
    idx = np.concatenate([rng.choice(np.where(ref["yt"] == k)[0], N_PER_CLASS, replace=(ref["yt"] == k).sum() < N_PER_CLASS) for k in MINOR])
    x, y = ref["xt"][idx].copy(), ref["yt"][idx].copy()
    sets = {}
    sets["copy"] = (x, y, "gan-train real cycles: realism target, must be flagged as copies")
    nc = x.copy()
    for i in range(len(nc)):
        m = frame_mask(nc[i])
        nc[i, m] = np.roll(nc[i, m], rng.randint(-20, 21), axis=0)
    nc = nc + rng.uniform(-0.2, 0.2, (len(nc), 1, 1)) + rng.normal(0, 0.1, nc.shape)
    sets["near_copy"] = (nc.astype(np.float32), y, "copies + time roll, gain, noise: copy detector should still fire")
    sets["shuffled"] = (x, rng.permutation(y), "real cycles with permuted labels: fid / tstr must fail")
    from scipy.ndimage import gaussian_filter
    sets["blur"] = (np.stack([gaussian_filter(v, (6, 3)) for v in x]).astype(np.float32), y,
                    "real cycles blurred (6 frames x 3 mel): coarse-GAN look, realism + crackle fid should drop")
    nz = np.empty_like(x)
    for k in MINOR:
        mu, sd = x[y == k].mean(0), x[y == k].std(0)
        nz[y == k] = mu + sd * rng.normal(size=((y == k).sum(),) + mu.shape)
    sets["noise"] = (nz.astype(np.float32), y, "per-pixel Gaussian with real class mean/std: everything should fail")
    col = np.empty_like(x)
    for k in MINOR:
        src = x[y == k][rng.choice((y == k).sum(), 5, replace=False)]
        col[y == k] = src[rng.randint(0, 5, (y == k).sum())] + rng.normal(0, 0.05, ((y == k).sum(),) + x.shape[1:])
    sets["collapse"] = (col.astype(np.float32), y, "5 real cycles per class repeated: div must be low, copies flagged")
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    ref = build_reference()
    print("reference: val minority %s | judge val recall %s | tau_emb %.4f tau_pix %.3f"
          % (np.bincount(ref["y"], minlength=4)[1:].tolist(), {G.CLASS_NAMES[k]: round(v, 1) for k, v in ref["judge_val_recall"].items()},
             ref["tau_emb"], ref["tau_pix"]))
    rows = []
    if a.controls:
        for n, (x, y, note) in controls(ref).items():
            rows.append(evaluate_set("ctrl_" + n, x, y, ref, note))
        md = os.path.join(J.EVAL_DIR, "controls.md")
        with open(md, "w") as f:
            f.write("# Evaluation-suite controls (v2)\n\nReference: held-out val patients, minority classes %s. "
                    "Judge recall on that real reference (the fid line for unseen real cycles): %s.\n\n" % (np.bincount(ref["y"], minlength=4)[1:].tolist(),
                    json.dumps({G.CLASS_NAMES[k]: round(v, 1) for k, v in ref["judge_val_recall"].items()})))
            f.write(fmt_table(rows) + "\n\n")
            for r in rows:
                f.write("- **%s**: %s\n" % (r["name"], r["note"]))
        print(open(md).read())
    for path in a.samples:
        d = np.load(path)
        rows.append(evaluate_set(a.name or os.path.basename(os.path.dirname(os.path.abspath(path))), d["x"], d["y"], ref, path))
    if a.samples:
        print(fmt_table(rows))


if __name__ == "__main__":
    main()
