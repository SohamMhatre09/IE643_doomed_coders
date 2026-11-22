#!/bin/bash
# Idea 2: BigGAN-style class-conditional GAN (hinge loss, spectral norm, conditional BN), nearest-neighbour
# up-sampling, real stethoscope background noise mixed into every fake. ICBHI data only. Best single model (60.50).
source "$(dirname "$0")/common.sh"
prepare
train_gan c1_class "$GAN_CKPT" --cond class --iters "$GAN_ITERS" --eval-every "$GAN_EVAL"
ast_seeds biggan --run c1_class --ckpt "$GAN_CKPT" --bg-noise
table S1_c1_class_${KTAG}_bgn
