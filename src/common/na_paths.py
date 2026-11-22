"""
na_paths.py -- every path of the release in one place. Override with environment variables (configs/paths.sh):
  DATA_ROOT   project folder holding the data                            default: parent of this release folder
  NA_OUT      outputs folder (caches, checkpoints, AST runs, tables)      default <release>/outputs
  ICBHI_DIR   folder with the 920 ICBHI .wav/.txt files                  default $DATA_ROOT/ICBHI_final_database
  ICBHI_SPLIT official ICBHI train/test split file                       default $DATA_ROOT/utils/split_details/split.txt
  HF_DIR      HF_Lung_V1 folder with train/ and test/ (wav + _label.txt)  default $DATA_ROOT/external/HF_Lung_V1
  AST_MODEL   AudioSet AST checkpoint (hub id or local folder)          default MIT/ast-finetuned-audioset-10-10-0.4593
  NA_OFFLINE  1 = load AST_MODEL from local files only
"""
import os

REL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(REL))
OUT = os.environ.get("NA_OUT", os.path.join(REL, "outputs"))
ICBHI_DIR = os.environ.get("ICBHI_DIR", os.path.join(DATA_ROOT, "ICBHI_final_database"))
SPLIT_FILE = os.environ.get("ICBHI_SPLIT", os.path.join(DATA_ROOT, "utils", "split_details", "split.txt"))
HF_DIR = os.environ.get("HF_DIR", os.path.join(DATA_ROOT, "external", "HF_Lung_V1"))
AST_MODEL = os.environ.get("AST_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")
OFFLINE = os.environ.get("NA_OFFLINE", "0") == "1"
HF_HOME = os.environ.get("HF_HOME", os.path.join(OUT, "hf_cache"))


def sub(*p):
    d = os.path.join(OUT, *p)
    os.makedirs(d, exist_ok=True)
    return d
