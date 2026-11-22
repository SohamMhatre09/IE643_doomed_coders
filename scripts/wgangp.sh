#!/bin/bash
# The same S1 AST with and without WGAN-GP samples (report Table 1, top). WGAN-GP: Gulrajani et al. 2017,
# with the "fixes" (DiffAugment + minibatch-stddev + mode-seeking); 1,000 samples per rare class.
source "$(dirname "$0")/common.sh"
prepare
if [ ! -f "$NA_OUT/wgangp_runs/wgangp_fixes/synthetic_specs.npy" ]; then
  say "training WGAN-GP"; $PY "$NA_ROOT/src/gan/wgangp/gan_wgangp.py" --fixes --iters "$WGAN_ITERS" 2>&1 | tee "$LOG/wgangp.log" | tail -3
fi
say "AST without / with WGAN-GP samples (seeds: $SEEDS)"
$PY "$NA_ROOT/src/classifier/sg_exp.py" --exp S1 --epochs "$EPOCHS" --seeds $SEEDS --no-register > "$LOG/sg_S1.log" 2>&1
$PY "$NA_ROOT/src/classifier/sg_exp.py" --exp S1 --gan wgangp_fixes --epochs "$EPOCHS" --seeds $SEEDS --no-register > "$LOG/sg_S1_wgangp.log" 2>&1
table S1 S1_wgangp_fixes
