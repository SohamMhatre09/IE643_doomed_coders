#!/bin/bash
# Idea 3: external data for the generator. Stage 1: the BigGAN-style GAN trained on ICBHI + clean HF_Lung cycles
# (rhonchi-only "wheeze" removed, source label). Stage 2: fine-tuned on ICBHI with the first critic blocks frozen.
# HF never reaches the classifier. Best method (ensemble 62.10).
source "$(dirname "$0")/common.sh"
prepare; prepare_hf
train_gan h2_cs_hfclean "$GAN_CKPT" --cond cs --hf --hf-clean --iters "$GAN_ITERS" --eval-every "$GAN_EVAL"
train_gan h2_cs_hfclean_st2 "$ST2_CKPT" --cond cs --init-from h2_cs_hfclean --freeze-d 2 --iters "$ST2_ITERS" \
          --lr-g 5e-5 --lr-d 2e-4 --eval-every "$ST2_EVAL"
ast_seeds hfgan --run h2_cs_hfclean_st2 --ckpt "$ST2_CKPT" --reject-max 25 --bg-noise
table S1_h2_cs_hfclean_st2_${KTAG2}_bgn
