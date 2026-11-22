"""
gate_check.py -- did the dual-branch generator's gate learn something, or collapse? (no training)

For each DualGen run at a checkpoint: 300 fakes per class (normal, crackle, wheeze, both), and
  - mean gate M per class (does the gate open more for abnormal classes, where events are?)
  - share of pixels with M > 0.5, and the gate's spread (collapse = mean near 0 or 1 with tiny spread)
  - correlation, per sample, between M and the local detail energy |D| and the output's fine detail
    (|x - blur(x)|): does the gate open where the detail is?
  - contribution of each branch: power of E (global) vs M * D (gated local)
Writes SPECTRAL/gate_<run>.md and gate_<run>.png (4 examples per class: output, global E, gate M, gated detail M*D).

    $PY SPECTRAL/gate_check.py dg1_dual 10000
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P
import gan as M                                     # noqa: E402
HERE = _P.sub("spectral")

NAMES = ["normal", "crackle", "wheeze", "both"]


@torch.no_grad()
def main(run, it):
    d = os.path.join(M.RUNS, run)
    cfg = json.load(open(os.path.join(d, "config.json")))
    attrs = [tuple(a) for a in cfg["attrs"]]
    G = M.DualGen(attrs, aa=cfg.get("aa_up", False), noise=cfg.get("noise_inj", False), bound=cfg.get("bound")).cuda().eval()
    G.load_state_dict(torch.load(os.path.join(d, "G_ema_%d.pt" % it), map_location="cuda"))
    names = [a for a, _ in attrs]
    rows, ex = [], {}
    gen = torch.Generator(device="cuda").manual_seed(0)
    for k in range(4):
        v = {"crackle": int(k in (1, 3)), "wheeze": int(k in (2, 3)), "source": 0, "device": 0, "region": 0, "one": 0}
        c = torch.tensor([[v[a] for a in names]] * 300, device="cuda")
        Ms, Es, Ds, Xs = [], [], [], []
        for i in range(0, 300, 100):
            x = G(torch.randn(100, M.Z, device="cuda", generator=gen), c[i:i + 100])
            Ms.append(G.last_gate); Es.append(G.last_E); Ds.append(G.last_D); Xs.append(x)
        Mg, E, D, X = (torch.cat(t)[:, 0].float() for t in (Ms, Es, Ds, Xs))
        fine = (X - F.avg_pool2d(X[:, None], 5, 1, 2)[:, 0]).abs()
        corr = lambda a, b: torch.stack([torch.corrcoef(torch.stack([a[j].flatten(), b[j].flatten()]))[0, 1]
                                         for j in range(len(a))]).nanmean().item()
        rows.append(dict(cls=NAMES[k], gate_mean=Mg.mean().item(), gate_sd=Mg.std().item(),
                         open_share=(Mg > 0.5).float().mean().item(),
                         corr_gate_detail=corr(Mg, D.abs()), corr_gate_fine=corr(Mg, fine),
                         power_global=E.pow(2).mean().item(), power_gated_local=(Mg * D).pow(2).mean().item()))
        ex[k] = (X[:4].cpu().numpy(), E[:4].cpu().numpy(), Mg[:4].cpu().numpy(), (Mg * D)[:4].cpu().numpy())
    L = ["# Gate check: %s @%d (dual-branch generator, x = E + M * D)\n" % (run, it),
         "| class | mean gate | gate sd | share open (>0.5) | corr(gate, abs D) | corr(gate, fine detail of output) | power global E | power gated local |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| %s | %.3f | %.3f | %.3f | %+.3f | %+.3f | %.3f | %.3f |" % (r["cls"], r["gate_mean"], r["gate_sd"],
                 r["open_share"], r["corr_gate_detail"], r["corr_gate_fine"], r["power_global"], r["power_gated_local"]))
    mean = np.mean([r["gate_mean"] for r in rows]); sd = np.mean([r["gate_sd"] for r in rows])
    verdict = ("COLLAPSED (gate ~0: local branch unused)" if mean < 0.05 else
               "COLLAPSED (gate ~1: no gating, = single generator)" if mean > 0.95 and sd < 0.05 else
               "gate is active (between 0 and 1, varies over the window)")
    L.append("\n**Verdict:** %s. Abnormal minus normal mean gate: %+.3f." % (verdict, np.mean([r["gate_mean"] for r in rows[1:]]) - rows[0]["gate_mean"]))
    open(os.path.join(HERE, "gate_%s.md" % run), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(8, 4, figsize=(16, 18))
    for k in range(4):
        for j in range(2):
            r = 2 * k + j
            for col, (img, title, cmap) in enumerate(zip([ex[k][i][j] for i in range(4)],
                                                         ["output x", "global E", "gate M", "gated local M*D"],
                                                         ["magma", "magma", "viridis", "coolwarm"])):
                ax[r, col].imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=(0 if col == 2 else None),
                                  vmax=(1 if col == 2 else None))
                ax[r, col].set_title("%s: %s" % (NAMES[k], title), fontsize=8)
                ax[r, col].set_xticks([]); ax[r, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "gate_%s.png" % run), dpi=110)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
