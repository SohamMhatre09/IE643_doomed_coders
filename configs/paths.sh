# configs/paths.sh -- where the data is. Defaults assume this folder sits inside the project folder (DATA_ROOT):
#   $DATA_ROOT/ICBHI_final_database/          920 ICBHI 2017 .wav + .txt files
#   $DATA_ROOT/utils/split_details/split.txt  official ICBHI train/test split
#   $DATA_ROOT/external/HF_Lung_V1/           HF_Lung_V1: train/ and test/ with .wav + _label.txt
# Edit DATA_ROOT (or any single path) here, or export it before calling ./run.sh.
export DATA_ROOT="${DATA_ROOT:-$(cd "$NA_ROOT/.." && pwd)}"
export ICBHI_DIR="${ICBHI_DIR:-$DATA_ROOT/ICBHI_final_database}"
export ICBHI_SPLIT="${ICBHI_SPLIT:-$DATA_ROOT/utils/split_details/split.txt}"
export HF_DIR="${HF_DIR:-$DATA_ROOT/external/HF_Lung_V1}"
export NA_OUT="${NA_OUT:-$NA_ROOT/outputs}"                            # everything the pipeline writes
# AudioSet-pretrained AST: a local copy ($DATA_ROOT/ast_model) if present, else downloaded from the HF hub
if [ -z "$AST_MODEL" ]; then
  if [ -f "$DATA_ROOT/ast_model/config.json" ]; then export AST_MODEL="$DATA_ROOT/ast_model" NA_OFFLINE="${NA_OFFLINE:-1}"
  else export AST_MODEL="MIT/ast-finetuned-audioset-10-10-0.4593"; fi
fi
export NA_OFFLINE="${NA_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PY="${PY:-python}"                                              # python with requirements.txt installed
export MAX_PARALLEL="${MAX_PARALLEL:-3}"                               # AST seeds trained in parallel (~7 GB GPU each)
