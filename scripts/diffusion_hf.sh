#!/bin/bash
# Ideas 3+4: diffusion trained on ICBHI + clean HF_Lung (source label), guidance sweep w = 0 1 2 4 6 8,
# then the pooled ensemble of all its models (62.54 in the report). Set GUIDANCE="4" to run only w=4.
source "$(dirname "$0")/common.sh"
prepare; prepare_hf
train_diffusion d3_edm_hf --obj edm --hf --iters "$DIFF_ITERS" --eval-every "$DIFF_EVAL"
for w in ${GUIDANCE:-0 1 2 4 6 8}; do
  say "generating 3,000 fakes at guidance $w"; $PY "$NA_ROOT/src/diffusion/meldiff.py" --pool d3_edm_hf "$w"
  ast_seeds diffusion_hf_w$w --diff --run d3_edm_hf --cfg "$w" --bg-noise
done
table $(for w in ${GUIDANCE:-0 1 2 4 6 8}; do echo S1_d3_edm_hf_cfg${w}_bgn; done)
$PY "$NA_ROOT/src/common/ensembles.py"
