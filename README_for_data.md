# Data: where to download it and where to put it

The code needs two datasets and one official split file. The AudioSet AST checkpoint is optional.
Download each one from the links below and paste it at the path shown.
`<DATA_ROOT>` is the folder that **contains** `novelty_assessment/`. For us that was `/home/sohamd/Deep-Learning-Project/`.

## Final layout

```
<DATA_ROOT>/
├── ICBHI_final_database/                 ← 1. ICBHI 2017
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ...                                  920 .wav + 920 .txt files, directly in this folder
├── utils/
│   └── split_details/
│       └── split.txt                     ← 2. official ICBHI train/test split
├── external/
│   └── HF_Lung_V1/                       ← 3. HF_Lung_V1
│       ├── train/                           *.wav + *_label.txt
│       └── test/                            *.wav + *_label.txt
├── ast_model/                            ← 4. optional: local AudioSet AST checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── novelty_assessment/                   ← this code
```

## 1. ICBHI 2017 Respiratory Sound Database
- **Download:** https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge, from the "Database" section (`ICBHI_final_database.zip`).
  A mirror is on Kaggle: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- **Paste to:** `<DATA_ROOT>/ICBHI_final_database/`. Unzip it so that the 920 `.wav` files and their `.txt` annotation files sit **directly** in this folder, not in a subfolder.
- **Check:** `ls <DATA_ROOT>/ICBHI_final_database/*.wav | wc -l` prints `920`.

## 2. Official ICBHI train / test split
- **Download:** on the same ICBHI challenge page, `ICBHI_challenge_train_test.txt`.
  It is a text file with one line per recording, `<recording name><TAB>train|test`.
- **Paste to:** `<DATA_ROOT>/utils/split_details/split.txt`, renamed to `split.txt`.
- **Check:** the file has 920 entries (`wc -l` prints 919 when the last line has no newline), and every line ends in `train` or `test`.
- **Note:** the code never changes this split, and never uses the test part for training or model choice.

## 3. HF_Lung_V1 (external lung-sound dataset)
- **Download:** https://gitlab.com/techsupportHF/HF_Lung_V1. It ships as split 7-zip archives, `train.7z.001` … and `test.7z.001` ….
- **Paste to:** `<DATA_ROOT>/external/HF_Lung_V1/`, then unpack the archives there:
  ```bash
  cd <DATA_ROOT>/external/HF_Lung_V1
  7z x train.7z.001     # creates train/
  7z x test.7z.001      # creates test/
  ```
- **Check:** `ls train/*.wav | wc -l` + `ls test/*.wav | wc -l` together give about 9,765 files, each with a matching `_label.txt`.
- **Needed only for:** `hfgan`, `diffusion_hf` and `controls`. `baseline`, `wgangp`, `biggan`, `azimuthal`, `dualbranch` and `diffusion` run without it.

## 4. AudioSet-pretrained AST (optional)
- **Default:** the code downloads `MIT/ast-finetuned-audioset-10-10-0.4593` from the Hugging Face hub on first use.
  Model page: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- **For a machine without internet:** download it once and paste it to `<DATA_ROOT>/ast_model/`. The code then uses it automatically, in offline mode.
  ```bash
  huggingface-cli download MIT/ast-finetuned-audioset-10-10-0.4593 --local-dir <DATA_ROOT>/ast_model
  ```

## Different locations?
Edit `configs/paths.sh`, or export the paths before running:
```bash
export ICBHI_DIR=/my/data/ICBHI_final_database
export ICBHI_SPLIT=/my/data/split.txt
export HF_DIR=/my/data/HF_Lung_V1
export AST_MODEL=/my/models/ast_model     # or keep the hub id
./run.sh biggan
```

All outputs (caches, checkpoints, results) are written to `novelty_assessment/outputs/`. Set `NA_OUT` to change this.
The first run of each approach builds the caches from these files automatically.
