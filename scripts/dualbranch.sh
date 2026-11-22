#!/bin/bash
# Idea 2: our dual-branch generator. A global CNN draws the overall spectrum and breathing envelope, a local CNN the
# fine detail, a learned gate places the detail: x = E + M * D. Trained with the spectral loss; gate check at the end.
source "$(dirname "$0")/common.sh"
prepare
train_gan dg2_dual_spec "$GAN_CKPT" --cond class --iters "$GAN_ITERS" --eval-every "$GAN_EVAL" --dual-gen --spec-reg 1
ast_seeds dualbranch --run dg2_dual_spec --ckpt "$GAN_CKPT" --bg-noise
say "gate check -> $NA_OUT/spectral/gate_dg2_dual_spec.{md,png}"
$PY "$NA_ROOT/src/spectral/gate_check.py" dg2_dual_spec "$GAN_CKPT" | tail -8
table S1_dg2_dual_spec_${KTAG}_bgn
