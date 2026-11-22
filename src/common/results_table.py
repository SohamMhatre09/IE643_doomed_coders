"""
results_table.py -- table of finished AST runs: mean +- sd (SE / SP) over seeds and the seed ensemble
(mean softmax, argmax, no threshold) on the official ICBHI test split.

    python results_table.py S1_c1_class_at10k_bgn S1_base ...      (run prefixes; seeds are the _s<seed> folders)
"""
import os
import sys
import csv
import glob

import numpy as np
import na_paths as P

AR = os.path.join(P.OUT, "ast_runs")


def load(d):
    """All seeds stored in run folder d (ast_gan: one seed per folder; sg_exp: several seeds in one folder)."""
    f = os.path.join(d, "per_seed_results.csv")
    out = []
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        z = os.path.join(d, "logits_seed%s.npz" % r["seed"])
        if not os.path.exists(z):
            continue
        z = np.load(z)
        l = z["test_logits"]; e = np.exp(l - l.max(1, keepdims=True))
        out.append((float(r["AS"]), float(r["SE"]), float(r["SP"]), e / e.sum(1, keepdims=True), z["test_y"]))
    return out


def ensemble(rs):
    a = np.mean([r[3] for r in rs], 0).argmax(1); y = rs[0][4]
    se, sp = 100 * (a[y > 0] == y[y > 0]).mean(), 100 * (a[y == 0] == 0).mean()
    return (se + sp) / 2, se, sp


def row(prefix, dirs=None):
    import re
    seed_dirs = [d for d in glob.glob(os.path.join(AR, prefix + "_s*"))
                 if re.fullmatch(re.escape(prefix) + r"_s\d+", os.path.basename(d))]   # exactly <prefix>_s<seed>
    dirs = dirs or sorted(seed_dirs) + ([os.path.join(AR, prefix)] if os.path.isdir(os.path.join(AR, prefix)) else [])
    rs = [r for d in dirs for r in load(d)]
    if not rs:
        return "| %s | - | not finished | |" % prefix
    A = np.array([r[:3] for r in rs]); sd = A[:, 0].std(ddof=1) if len(A) > 1 else 0.0
    e = ensemble(rs)
    return "| %s | %d | %.2f +- %.2f (SE %.1f / SP %.1f) | %.2f (SE %.1f / SP %.1f) |" % (
        prefix, len(rs), A[:, 0].mean(), sd, A[:, 1].mean(), A[:, 2].mean(), *e)


if __name__ == "__main__":
    lines = ["| run | seeds | mean +- sd (SE / SP) | ensemble (SE / SP) |", "|---|---|---|---|"] + [row(p) for p in sys.argv[1:]]
    print("\n".join(lines))
    with open(os.path.join(P.OUT, "results.md"), "a") as f:
        f.write("\n".join(lines) + "\n\n")
