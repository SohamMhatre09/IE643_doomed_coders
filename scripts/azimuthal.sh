#!/bin/bash
# Idea 1: azimuthal integration (Durall et al. 2020). (a) 1D spectrum of real vs generated spectrograms,
# (b) the same GAN trained with Durall's spectral loss, (c) anti-aliased up-sampling for comparison.
source "$(dirname "$0")/common.sh"
prepare
train_gan c1_class "$GAN_CKPT" --cond class --iters "$GAN_ITERS" --eval-every "$GAN_EVAL"
train_gan sp1_nn_spec "$GAN_CKPT" --cond class --iters "$GAN_ITERS" --eval-every "$GAN_EVAL" --spec-reg 1
train_gan sp2_aa "$GAN_CKPT" --cond class --iters "$GAN_ITERS" --eval-every "$GAN_EVAL" --aa-up
ast_seeds sp1 --run sp1_nn_spec --ckpt "$GAN_CKPT" --bg-noise
ast_seeds sp2 --run sp2_aa --ckpt "$GAN_CKPT" --bg-noise
say "azimuthal spectra of every trained generator -> $NA_OUT/spectral/spectra.{png,md}"
SPEC_CKPT=$GAN_CKPT $PY "$NA_ROOT/src/spectral/measure.py" | tail -12
table S1_sp1_nn_spec_${KTAG}_bgn S1_sp2_aa_${KTAG}_bgn
