#!/bin/bash
# Controls: (a) the BigGAN-style GAN's fakes with shuffled labels; (b) HF_Lung given to the classifier as REAL
# training data (4,000 cycles, rhonchi/stridor dropped, ICBHI class ratio; HF devices in the device head).
source "$(dirname "$0")/common.sh"
prepare; prepare_hf
train_gan c1_class "$GAN_CKPT" --cond class --iters "$GAN_ITERS" --eval-every "$GAN_EVAL"
ast_seeds shuffle --control shuffle --run c1_class --ckpt "$GAN_CKPT" --bg-noise
$PY "$NA_ROOT/src/data/hfreal.py" 4000 798 > "$LOG/hfreal.log" 2>&1 || true
ast_seeds hfreal --no-synth --hf-real 4000
table S1_c1_class_shuf_bgn S1_base_hfr4000
