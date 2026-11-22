"""
ensembles.py -- the pooled ensembles quoted in the report, e.g. all diffusion+HF models over every guidance w (62.54).
    python ensembles.py
"""
import os
import glob
import na_paths as P
from results_table import load, ensemble, AR

SETS = [("All diffusion + HF models (every guidance w)", "S1_d3_edm_hf_cfg*_bgn_s*"),
        ("All diffusion ICBHI-only models", "S1_d1_edm_cfg*_bgn_s*")]

if __name__ == "__main__":
    for name, pat in SETS:
        rs = [r for d in sorted(glob.glob(os.path.join(AR, pat))) for r in load(d)]
        if rs:
            print("%-48s models %2d  Score %.2f  SE %.1f  SP %.1f" % ((name, len(rs)) + ensemble(rs)))
        else:
            print("%-48s not run yet" % name)
