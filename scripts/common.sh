#!/bin/bash
# common.sh -- shared settings and helpers, sourced by every approach script.
set -e
export NA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "${SMOKE:-0}" = "1" ] && export NA_OUT="${NA_OUT:-$NA_ROOT/outputs_smoke}"   # smoke runs never mix with real outputs
source "$NA_ROOT/configs/paths.sh"
export PYTHONPATH="$NA_ROOT/src/common:$NA_ROOT/src/classifier:$NA_ROOT/src/data:$NA_ROOT/src/gan:$NA_ROOT/src/gan/wgangp:$NA_ROOT/src/diffusion:$NA_ROOT/src/spectral${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$NA_OUT/logs"
LOG="$NA_OUT/logs"

if [ "${SMOKE:-0}" = "1" ]; then            # tiny end-to-end check (minutes): few steps, 1 epoch, 1 seed
  GAN_ITERS=200; GAN_EVAL=100; GAN_CKPT=200
  ST2_ITERS=100; ST2_EVAL=100; ST2_CKPT=100
  DIFF_ITERS=200; DIFF_EVAL=200
  WGAN_ITERS=50; EPOCHS=1; SEEDS="${SEEDS:-13}"
  export NA_SAMPLE_STEPS=2
else                                         # the settings used for the report
  GAN_ITERS=20000; GAN_EVAL=5000; GAN_CKPT=10000
  ST2_ITERS=10000; ST2_EVAL=2500; ST2_CKPT=10000
  DIFF_ITERS=30000; DIFF_EVAL=10000
  WGAN_ITERS=6000; EPOCHS=30; SEEDS="${SEEDS:-13 42 97}"
fi
KTAG="at$((GAN_CKPT / 1000))k"; KTAG2="at$((ST2_CKPT / 1000))k"

say() { echo "[$(date '+%F %T')] $*"; }

prepare() {                                 # ICBHI caches, built once: waveforms, generator log-mel cache, noise bank
  if [ ! -f "$NA_OUT/gan_data/meta.csv" ]; then say "building ICBHI caches"; $PY "$NA_ROOT/src/data/gfdata.py"; fi
  if [ ! -f "$NA_OUT/noise_bank/noise_bank.npz" ]; then say "building real background-noise bank"; $PY "$NA_ROOT/src/classifier/bgnoise.py"; fi
}

prepare_hf() {                              # HF_Lung caches (generator training only)
  if [ ! -f "$NA_OUT/gan_data/hf_meta.csv" ]; then say "building HF_Lung cache"; $PY "$NA_ROOT/src/data/hfdata.py"; fi
  if [ ! -f "$NA_OUT/gan_data/hf_clean.csv" ]; then say "HF_Lung rhonchi clean-up + frame labels"; $PY "$NA_ROOT/src/data/hfprep.py"; fi
}

train_gan() {                               # train_gan NAME CKPT [gan.py args]   (skipped if the checkpoint exists)
  local name=$1 ck=$2; shift 2
  if [ -f "$NA_OUT/gan_runs/$name/G_ema_$ck.pt" ] && [ -f "$NA_OUT/gan_runs/$name/ckpt.pt" ]; then say "GAN $name exists, skipping"; return; fi
  say "training GAN $name"
  $PY "$NA_ROOT/src/gan/gan.py" --name "$name" "$@" 2>&1 | tee "$LOG/gan_$name.log" | grep -E "^\{|DONE|EVAL" || true
  [ -f "$NA_OUT/gan_runs/$name/G_ema_$ck.pt" ] || { echo "GAN $name failed, see $LOG/gan_$name.log"; exit 1; }
}

train_diffusion() {                         # train_diffusion NAME [meldiff.py args]
  local name=$1; shift
  if [ -f "$NA_OUT/diffusion_runs/$name/ema_$DIFF_ITERS.pt" ]; then say "diffusion $name exists, skipping"; return; fi
  say "training diffusion model $name"
  $PY "$NA_ROOT/src/diffusion/meldiff.py" --name "$name" "$@" 2>&1 | tee "$LOG/diff_$name.log" | grep -E "^\{|DONE" || true
  [ -f "$NA_OUT/diffusion_runs/$name/ema_$DIFF_ITERS.pt" ] || { echo "diffusion $name failed, see $LOG/diff_$name.log"; exit 1; }
}

ast_seeds() {                               # ast_seeds TAG [ast_gan.py args]: one AST per seed, MAX_PARALLEL at a time
  local tag=$1; shift; local n=0 pids=()
  for s in $SEEDS; do
    say "AST $tag seed $s"
    $PY "$NA_ROOT/src/classifier/ast_gan.py" "$@" --epochs "$EPOCHS" --seeds "$s" --suffix "_s$s" > "$LOG/ast_${tag}_s$s.log" 2>&1 &
    pids+=($!); n=$((n + 1))
    if [ "$n" -ge "$MAX_PARALLEL" ]; then wait "${pids[0]}" || { echo "AST failed: $LOG/ast_${tag}_*.log"; exit 1; }; pids=("${pids[@]:1}"); n=$((n - 1)); fi
    sleep 20
  done
  for p in "${pids[@]}"; do wait "$p" || { echo "AST failed: $LOG/ast_${tag}_*.log"; exit 1; }; done
}

table() { $PY "$NA_ROOT/src/common/results_table.py" "$@"; }
