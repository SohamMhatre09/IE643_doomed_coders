#!/bin/bash
# Idea 4: mel-spectrogram diffusion (EDM) with classifier-free guidance, ICBHI data only, guidance w=2.
source "$(dirname "$0")/common.sh"
prepare
train_diffusion d1_edm --obj edm --iters "$DIFF_ITERS" --eval-every "$DIFF_EVAL"
say "generating 3,000 fakes at guidance 2"; $PY "$NA_ROOT/src/diffusion/meldiff.py" --pool d1_edm 2
ast_seeds diffusion --diff --run d1_edm --cfg 2 --bg-noise
table S1_d1_edm_cfg2_bgn
