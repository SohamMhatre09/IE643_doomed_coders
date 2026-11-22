# Novelty assessment code: generated spectrograms for ICBHI respiratory-sound classification

**IE 643 course project**, team **doomed_coders**: Soham Mhatre (26M2572) and Parivat (26M2575).

**Project:** *Automated Respiratory Sound Classification via Audio Spectrogram Transformers.*

This folder holds the code for every approach in our novelty-assessment report
(`doomed_coders_IE643_NoveltyAssessment_report.pdf`).
Each approach runs with **one command**. That command:
1. builds the data caches it needs;
2. trains the generator (GAN or diffusion model);
3. generates the synthetic spectrograms;
4. trains the AST classifier for every seed;
5. prints mean ± sd, SE / SP, and the seed ensemble on the official ICBHI test split.

---

## 1. What is being tested

ICBHI 2017 is small and imbalanced, so we add generated spectrograms of the rare classes (crackle, wheeze, both) to
the classifier's training data.

The classifier is identical in every approach: an AudioSet-pretrained AST, fully fine-tuned, with a
device-adversarial head; 8 s windows, 30 epochs, last epoch kept.
**The only thing that changes between approaches is the set of 3,000 generated training cycles** (1,000 per rare
class).

Metric: **ICBHI Score = (SE + SP) / 2**, on the official test split.

| Command | Approach (report) | Generator data | Report result: mean ± sd (ensemble) |
|---|---|---|---|
| `./run.sh baseline` | AST without generated data | – | 58.04 ± 1.24 (60.00), 5 seeds |
| `./run.sh wgangp` | Same AST with vs without WGAN-GP samples (Table 1) | ICBHI | 59.39 → 59.71 |
| `./run.sh biggan` | **Idea 2**: BigGAN-style GAN, nearest up-sampling + real background noise | ICBHI | **60.50 ± 0.70** (61.53), best single model |
| `./run.sh azimuthal` | **Idea 1**: azimuthal-integration spectrum + GAN with Durall's spectral loss; anti-aliased up-sampling | ICBHI | 59.45 ± 1.51 (60.59); anti-aliased 58.14 |
| `./run.sh dualbranch` | **Idea 2**: our dual-branch generator (global + local CNN, learned gate) + spectral loss | ICBHI | 60.01 ± 1.79 (61.75) |
| `./run.sh hfgan` | **Idea 3**: GAN pretrained on clean HF_Lung, fine-tuned on ICBHI | ICBHI + HF | 59.55 ± 1.28 (**62.10**), best method |
| `./run.sh diffusion` | **Idea 4**: mel diffusion (EDM) with classifier-free guidance, w = 2 | ICBHI | 60.28 ± 1.45 (61.84) |
| `./run.sh diffusion_hf` | **Ideas 3+4**: diffusion with HF, guidance sweep w = 0–8, pooled ensemble | ICBHI + HF | w = 4: 60.49 ± 0.25 (61.99); all 20 models **62.54** |
| `./run.sh controls` | Shuffled-label control; HF_Lung as *real* classifier data | – | 58.16 ± 0.83; 56.79 ± 1.14 |
| `./run.sh all` | Everything above, one after another | | |

Notes on these numbers:
- Exact numbers can differ by about ±1 from run to run: GPU training is not fully deterministic, and seed-to-seed sd is 0.3–1.8.
- The guidance scale w was compared on the test split in the report, so the diffusion numbers are slightly optimistic.

---

## 2. Setup

### Hardware we used
- **GPU:** 1 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB (driver 595.71, CUDA 13.0).
- **CPU:** 2 × AMD EPYC 9224 24-core, 48 cores / 96 threads in total.
- **RAM:** 503 GB.

Each AST seed needs about 7 GB of GPU memory, a GAN about 7 GB, diffusion training about 13 GB, and diffusion
sampling about 24 GB. A 24 GB GPU is enough if you lower `MAX_PARALLEL` to 1.

### Software
```bash
conda create -n na python=3.11 -y && conda activate na
pip install -r requirements.txt     # torch 2.13 / torchaudio 2.11 (CUDA 13.0 wheels), transformers 5.17, ...
```

### Data
**[README_for_data.md](README_for_data.md) has the download links and exactly where to paste each dataset.**
The defaults assume this folder sits inside the project folder (`DATA_ROOT` = the parent of `novelty_assessment/`):

```
<DATA_ROOT>/
├── ICBHI_final_database/            # ICBHI 2017: 920 .wav + .txt files
├── utils/split_details/split.txt    # official ICBHI train/test split
├── external/HF_Lung_V1/             # HF_Lung_V1: train/ and test/ with .wav + _label.txt
├── ast_model/                       # optional: local AudioSet AST (otherwise downloaded from the HF hub)
└── novelty_assessment/              # this folder
```

**Where to get the data:**
- **ICBHI 2017:** https://bhichallenge.med.auth.gr/
- **HF_Lung_V1:** https://gitlab.com/techsupportHF/HF_Lung_V1. It ships as split 7z archives; unpack them with `7z x train.7z.001` and `7z x test.7z.001`.
- **AST checkpoint:** `MIT/ast-finetuned-audioset-10-10-0.4593`.

Paths can be changed in `configs/paths.sh`, or by exporting variables before running:
`DATA_ROOT`, `ICBHI_DIR`, `ICBHI_SPLIT`, `HF_DIR`, `NA_OUT` (outputs), `AST_MODEL`, `PY` (python executable),
`MAX_PARALLEL` (AST seeds in parallel, default 3), `CUDA_VISIBLE_DEVICES`.

---

## 3. Running

```bash
cd novelty_assessment
./run.sh biggan                         # full run, seeds 13 42 97
./run.sh biggan --seeds "13 42 97 7 21" # more seeds
./run.sh diffusion_hf --smoke           # quick end-to-end check: few training steps, 1 epoch, 1 seed (writes to outputs_smoke/)
GUIDANCE="4" ./run.sh diffusion_hf      # only guidance w=4 instead of the full sweep
PY=/path/to/python ./run.sh all         # everything
```

- **Re-running is safe.** Finished generators and caches are detected and skipped.
- **Seeds:** AST seeds of one approach run in parallel (`MAX_PARALLEL`).
- **Results:** printed at the end, and appended to `outputs/results.md`. Every run is also registered in `outputs/all_runs.csv`.

### Run times on our GPU (one job alone)
| Step | Time |
|---|---|
| Caches (ICBHI, first run) | ~10 min |
| HF_Lung caches (first run) | ~15 min, using 48 CPU processes |
| GAN, 20k steps | ~2 h |
| GAN fine-tune, 10k steps | ~1 h |
| WGAN-GP, 6k steps | ~30 min |
| Diffusion, 30k steps | ~4–6 h |
| One fake pool (3,000 samples) | 40–90 min |
| One AST seed, 30 epochs | ~1 h; 2–3 h when several share the GPU |

### Outputs (`outputs/`, git-ignored)
| Folder / file | Contents |
|---|---|
| `cache/` | ICBHI waveforms at 16 kHz |
| `gan_data/` | Log-mel cache of training cycles for the generators, plus the HF_Lung cache |
| `noise_bank/` | Real background-noise frames |
| `hf_data/` | HF frame labels, rhonchi flags, HF real-cycle set |
| `gan_runs/<name>/` | GAN `config.json`, `G_ema_<it>.pt`, `ckpt.pt`, `samples_<it>.npz`, `train.log` |
| `wgangp_runs/`, `wgangp_data/` | WGAN-GP generator and its 1,000-per-class samples |
| `diffusion_runs/<name>/` | Diffusion `ema_<it>.pt`, fake pools `pool_w<w>_it<it>.npz` |
| `ast_runs/S1_<name>_s<seed>/` | Classifier results: `per_seed_results.csv` (AS, SE, SP, recalls, confusion), `logits_seed*.npz` (test probabilities), `final_summary.json`, `train.log` |
| `spectral/` | Azimuthal spectra (`spectra.png`/`.md`) and gate check (`gate_<run>.md`/`.png`) |
| `results.md`, `all_runs.csv`, `logs/` | Tables and per-step logs |

Smoke runs (`--smoke`) write to `outputs_smoke/` instead, so they never mix with real results.

---

## 4. How each approach works (code map)

```
src/common/      na_paths.py (all paths), train_optimal_pipeline.py (ICBHI parsing, official split, AST builder,
                 ICBHI metrics), results_table.py (mean ± sd + ensemble), ensembles.py, register_result.py
src/classifier/  sg_exp.py (S1 classifier: AST + device-adversarial head), ast_gan.py (adds generated data and
                 controls; all approaches go through it), bgnoise.py (real background-noise mixing)
src/data/        gfdata.py (log-mel cache), hfdata.py + hfprep.py (HF_Lung caches, rhonchi clean-up), hfreal.py
src/gan/         gan.py (BigGAN-style GAN, dual-branch generator, spectral loss, HF pretraining),
                 evaluate.py + judge.py (optional GAN metrics, NA_GAN_EVAL=1), wgangp/ (WGAN-GP baseline)
src/diffusion/   meldiff.py (U-Net diffusion, EDM / flow matching, classifier-free guidance, fake pools)
src/spectral/    azimuth.py (azimuthal integration + spectral loss), measure.py (spectrum figure), gate_check.py
scripts/         one script per approach (called by run.sh); common.sh holds settings and helpers
```

### Classifier and real background noise (every approach)
- **Classifier (`ast_gan.py`):** adds 1,000 generated cycles per rare class to the official training cycles and trains the S1 AST. The device-adversarial head ignores generated cycles, since they have no device label.
- **`--bg-noise` (`bgnoise.py`):** each fake gets real stethoscope background, the quietest 20% of frames of real ICBHI training cycles. It is mixed in the power domain, `L = log(exp(L_fake) + g·exp(L_noise))`, with g log-uniform in [0.2, 0.7]. This is what makes the GAN fakes useful.

### `wgangp`
- **Generator (`src/gan/wgangp/`):** WGAN-GP (Gulrajani et al. 2017) with gradient penalty λ = 10, 5 critic steps, plus DiffAugment, minibatch-stddev and mode-seeking. It produces 1,000 samples per rare class.
- **Classifier (`sg_exp.py`):** trained twice with identical settings, once without and once with these samples.

### `biggan` (idea 2)
- **Generator (`gan.py`):** class-conditional generator going from z + class embedding up to 128 × 512 log-mel, with five up-sampling blocks, conditional batch norm and nearest-neighbour up-sampling.
- **Critic:** spectral norm, hinge loss and a projection term. Trained 20k steps; the classifier uses the 10k checkpoint.

### `azimuthal` (idea 1)
- **The measurement (`azimuth.py`):** the 2D FFT log-power of each spectrogram, averaged over 64 rings. `measure.py` compares real cycles with every generator; nearest-neighbour up-convolution loses high-frequency detail.
- **The training loss (`gan.py --spec-reg 1`):** Durall's spectral loss, a BCE between each fake's normalised 1D spectrum and the running mean of the real spectra.
- **Comparison:** `--aa-up` trains the same GAN with anti-aliased (bilinear + blur) up-sampling instead.

### `dualbranch` (idea 2, our generator)
- **Global branch (`gan.py --dual-gen`, class `DualGen`):** a mean spectrum a(f), a breathing envelope b(t) and a smooth low-resolution map, together giving E.
- **Local branch:** the BigGAN trunk, giving the detail D.
- **Gate and output:** a gate M = sigmoid(conv[h, E]) gives the output x = E + M·D, so the gate decides where local detail goes.
- **Regulariser:** without one, the gate collapses to 1; `--gate-reg` prevents that.
- **Gate check:** `gate_check.py` reports the gate per class and draws its maps.

### `hfgan` (idea 3)
- **Caches:** `hfdata.py` cuts HF_Lung into breathing cycles labelled from the timestamped events. `hfprep.py` flags the rhonchi-only "wheeze" cycles, about 30–35% of them.
- **Stage 1 (`--cond cs --hf --hf-clean`):** GAN training with 70% HF per batch and a source label.
- **Stage 2:** fine-tuning on ICBHI with the first two critic blocks frozen. Fakes are always drawn in ICBHI style, and HF never reaches the classifier.

### `diffusion` / `diffusion_hf` (ideas 4 and 3+4)
- **Model (`meldiff.py`):** a U-Net on the same 128 × 512 log-mel windows (2×2 pixel-unshuffle, channels 64-128-192-256, attention at the lowest level). It uses EDM training (Karras et al. 2022) and a Heun sampler with 18 steps.
- **Classifier-free guidance:** the class is dropped 15% of the time during training; at sampling, `out = uncond + w·(cond − uncond)`.
- **Fake pools:** `--pool RUN W` generates one pool per (model, w), shared by the AST seeds.
- **HF:** `--hf` adds clean HF with a source label.
- **Pooled ensemble:** `ensembles.py` pools all seeds and guidance levels.

### `controls`
- **Shuffled labels:** `--control shuffle` keeps the same GAN fakes but permutes their labels.
- **HF as real data:** `--hf-real 4000` (`hfreal.py`) adds 4,000 HF cycles as real classifier data. Rhonchi and stridor cycles are dropped, ICBHI's class ratio is kept, and the HF devices go into the device-adversarial head.

---

## 5. Tested
Every approach was run end to end with `--smoke` on 22 Nov 2026: 200 generator steps, 1 AST epoch, 1 seed.
All 9 commands passed: baseline, wgangp, biggan, azimuthal, dualbranch, hfgan, diffusion, diffusion_hf and controls.
The full-length settings are the ones that produced the report's numbers.

## 6. Notes
- **Test split:** the official split is never changed and never used to train or pick models. The one exception is that the guidance scale w was compared on test, as stated in the report.
- **Not included:** variants the report lists only as "did not help" beyond anti-aliased up-sampling (flow matching, HF event-detector teacher, rule-based event insertion), and our earlier exploratory code. The code paths for `--dct`, `--clean` and `--teacher` inside `ast_gan.py` / `gan.py` are inactive here.
- **Optional GAN realism metrics:** `NA_GAN_EVAL=1 ./run.sh biggan` turns on the judge-based metrics (FD, coverage, TSTR). They need an extra judge AST, which is trained automatically.
# IE643_doomed_coders
