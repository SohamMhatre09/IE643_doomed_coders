"""
measure.py -- azimuthally integrated spectra (Durall et al. 2020) of real ICBHI cycles vs every generator family.
Same input for all: raw log-mel windows [512 frames, 128 mel], the GAN window (real: crop_pad of gan-train cycles).
Writes SPECTRAL/spectra.png (1D azimuthal + time-axis + frequency-axis spectra, ratio to real) and spectra.md
(high-frequency power error per set: mean |fake - real| of the log spectrum over the top 25 % of radii, and overall).

    $PY SPECTRAL/measure.py
"""
import os
import sys
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P
import gfdata as G                                  # noqa: E402
from azimuth import Azimuth, N_BINS                  # noqa: E402

GR = os.path.join(_P.OUT, "gan_runs")
MR = os.path.join(_P.OUT, "diffusion_runs")
HERE = _P.sub("spectral")
N = 900


def load_sets():
    """Real gan-val minority cycles + every generator found in the outputs: GANs at checkpoint SPEC_CKPT
    (gan_runs/<run>/samples_<SPEC_CKPT>.npz) and diffusion fake pools (diffusion_runs/<run>/pool_w*.npz)."""
    import glob
    rows = [r for r in G.load_meta() if r["gan_split"] == "val" and r["label"] in (1, 2, 3)]
    rng = np.random.RandomState(0)
    real = np.stack([G.crop_pad(G.get_fbank(rows[i]))[0] for i in rng.choice(len(rows), N, replace=len(rows) < N)])
    sets = {"real (gan-val minority)": real}
    ck = int(os.environ.get("SPEC_CKPT", "10000"))
    cand = [("GAN %s" % os.path.basename(os.path.dirname(f)), f) for f in sorted(glob.glob(os.path.join(GR, "*", "samples_%d.npz" % ck)))]
    cand += [("Diffusion %s %s" % (os.path.basename(os.path.dirname(f)), os.path.basename(f)[5:-4].split("_")[0]), f)
             for f in sorted(glob.glob(os.path.join(MR, "*", "pool_w*.npz")))]
    for name, f in cand:
        x = np.load(f)["x"].astype(np.float32)
        sets[name] = x[np.random.RandomState(1).choice(len(x), min(N, len(x)), replace=False)]
    return sets


@torch.no_grad()
def main():
    az = Azimuth().cuda() if torch.cuda.is_available() else Azimuth()
    dev = az.M.device
    res, curves = {}, {}
    for name, x in load_sets().items():
        t = torch.from_numpy(x).to(dev).transpose(1, 2)              # [N, 128 mel, 512 frames]
        s = torch.cat([az(t[i:i + 128]) for i in range(0, len(t), 128)]).mean(0).cpu().numpy()
        ts, fs = Azimuth.axis_spectra(t)
        curves[name] = dict(az=s, t=ts.mean(0).cpu().numpy(), f=fs.mean(0).cpu().numpy())
    ref = curves["real (gan-val minority)"]
    hi = slice(int(0.75 * N_BINS), N_BINS)
    for name, c in curves.items():
        d = c["az"] - ref["az"]
        res[name] = dict(az_err_all=float(np.abs(d).mean()), az_err_high=float(np.abs(d[hi]).mean()),
                         az_bias_high=float(d[hi].mean()),
                         time_err_high=float(np.abs((c["t"] - ref["t"])[len(ref["t"]) * 3 // 4:]).mean()),
                         freq_err_high=float(np.abs((c["f"] - ref["f"])[len(ref["f"]) * 3 // 4:]).mean()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, c in curves.items():
        lw = 2.6 if name.startswith("real") else 1.4
        ax[0].plot(np.linspace(0, 0.5, N_BINS), c["az"], lw=lw, label=name)
        ax[1].plot(np.linspace(0, 0.5, len(c["t"])), c["t"] - ref["t"], lw=lw)
        ax[2].plot(np.linspace(0, 0.5, len(c["f"])), c["f"] - ref["f"], lw=lw)
    ax[0].set(title="Azimuthally integrated spectrum (log power)", xlabel="normalised radial frequency")
    ax[1].set(title="Time-axis spectrum, difference to real", xlabel="cycles / frame")
    ax[2].set(title="Mel-axis spectrum, difference to real", xlabel="cycles / mel bin")
    for a in ax[1:]:
        a.axhline(0, color="k", lw=0.8)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "spectra.png"), dpi=140)
    L = ["# Azimuthal integration (Durall et al., CVPR 2020) of real vs generated log-mel windows\n",
         "Mean |log-spectrum difference to real|; 'high' = top 25 %% of radial frequencies (where up-convolution "
         "artefacts live); bias > 0 = too much high-frequency power. %d samples per set.\n" % N,
         "| set | azimuthal, all | azimuthal, high | azimuthal bias, high | time axis, high | mel axis, high |",
         "|---|---|---|---|---|---|"]
    for name, r in res.items():
        L.append("| %s | %.3f | %.3f | %+.3f | %.3f | %.3f |" % (name, r["az_err_all"], r["az_err_high"], r["az_bias_high"],
                                                              r["time_err_high"], r["freq_err_high"]))
    open(os.path.join(HERE, "spectra.md"), "w").write("\n".join(L) + "\n")
    json.dump({k: {kk: v.tolist() for kk, v in c.items()} for k, c in curves.items()}, open(os.path.join(HERE, "spectra.json"), "w"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
