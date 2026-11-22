#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_optimal_pipeline.py -- AST fine-tuning on ICBHI, OPTIMAL pipeline only.

This is the reduced, single-configuration version of the S1 x S2 x S3 screening
funnel. The funnel already ran (see results.csv / report) and converged on:

    S1 (cleaning)      = C0   -> raw audio, no filtering (bandpass/heartbeat-suppress/
                                  spectral-subtraction all failed to beat raw; see report)
    S2 (windowing)     = W4   -> sliding 50%-overlap multi-crop, MEAN-pooled at inference
    L  (window length) = 3s   -> beat 5s systematically (padding hurt Specificity)
    S3 (augmentation)  = S3-C -> Mixup / SpecMix (alpha=0.4, p=0.8) -- highest AS
    epochs             = 30, seeds = [13, 42, 97]  (final confirmation stage)

No grid, no multi-process coordination, no alternate S1/S2/S3 branches: this script
just builds that one pipeline end to end and reports the 3-seed mean +/- std.

Requirements: torch, torchaudio, transformers, soundfile, scipy, numpy, tqdm.
Usage:
    python train_optimal_pipeline.py [--smoke] [--out-dir DIR]
"""

import os
import sys
import gc
import csv
import json
import math
import time
import random
import logging
import argparse
from glob import glob
from collections import defaultdict, Counter
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# Point HF's cache/token bookkeeping at a directory we can actually write to, and force
# fully-offline/local loading -- we load the AST checkpoint from a local folder, so it
# should never try to touch the hub or the shared /opt/hf-cache (that's what was causing
# the PermissionError on /opt/hf-cache/token).
import na_paths as _P                                  # release: all paths from na_paths / configs/paths.sh
os.environ["HF_HOME"] = _P.HF_HOME
os.environ.pop("HF_HUB_CACHE", None)
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_TOKEN_PATH", None)
if _P.OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import soundfile as sf
from scipy.signal import resample_poly

# =====================================================================================
# CONFIG
# =====================================================================================
ROOT_DIR = _P.ICBHI_DIR.rstrip("/") + "/"
SPLIT_FILE = _P.SPLIT_FILE
OUT_DIR = os.path.join(_P.OUT, "pipeline")
CACHE_DIR = os.path.join(_P.OUT, "cache")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")   # per-seed model checkpoints (needed by gan_pipeline.py)

EXPECTED_N_RECORDINGS = 920
EXPECTED_N_PATIENTS = 126
EXPECTED_N_CYCLES = 6898
EXPECTED_LEAKED_PATIENTS = {156, 218}
CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]

MODEL_NAME = _P.AST_MODEL   # local checkpoint (pretrained AST, config+weights on disk)
SR = 16000
N_BLOCKS = 12
FREEZE_BLOCKS = 8                 # patch-embed + first 8 blocks frozen; last 4 + head train
NUM_CLASSES = 4
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.10
BATCH_SIZE = 8
ACCUM_STEPS = 2                   # effective batch 16
EVAL_BATCH_SIZE = 16
USE_AMP = True
GRAD_CLIP = 1.0
EPOCHS = 30
SEEDS = [13, 42, 97]
NUM_WORKERS = 8
WINDOW_SEC = 3
WINDOW_HOP_FRAC = 0.5
MAX_CROPS = 4                     # W4: evenly subsample if more sliding crops than this
POOL = "mean"                     # W4 == mean pooling over crops at inference
LOG_EVERY_BATCHES = 100

# S3-C: Mixup / SpecMix
MIX_ALPHA = 0.4
MIX_P = 0.8
SPECMIX_PROB = 0.5

log = logging.getLogger("train")


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    h1 = logging.StreamHandler(sys.stdout)
    h1.setFormatter(fmt)
    h2 = logging.FileHandler(os.path.join(out_dir, "train.log"))
    h2.setFormatter(fmt)
    log.addHandler(h1)
    log.addHandler(h2)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================================
# DATA LOADING (raw wavs + annotations + official split; flags known data-quality issues)
# =====================================================================================
def parse_fname(stem):
    p = stem.split("_")
    if len(p) != 5:
        raise ValueError("Unexpected filename (need 5 '_' fields): %s" % stem)
    return dict(patient=int(p[0]), rec_idx=p[1], loc=p[2], mode=p[3], equip=p[4])


def load_annotation(txt_path):
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            t = line.split()
            if len(t) < 4:
                continue
            rows.append((float(t[0]), float(t[1]), int(float(t[2])), int(float(t[3]))))
    return rows


def load_split_file(path):
    m = {}
    with open(path, "r") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 2:
                m[t[0]] = t[1].lower()
    return m


def build_dataset(smoke_stride=1):
    wavs = sorted(glob(os.path.join(ROOT_DIR, "*.wav")))
    log.info("Found %d .wav files under %s", len(wavs), ROOT_DIR)
    split_map = load_split_file(SPLIT_FILE)

    all_recs = []
    for w in wavs:
        stem = os.path.splitext(os.path.basename(w))[0]
        txt = os.path.splitext(w)[0] + ".txt"
        if not os.path.exists(txt):
            log.warning("No annotation .txt for %s -> skipped", stem)
            continue
        meta = parse_fname(stem)
        meta.update(rec_id=stem, wav=w, ann=load_annotation(txt))
        all_recs.append(meta)

    if smoke_stride == 1:
        n_pat = len({r["patient"] for r in all_recs})
        n_cycles = sum(len(r["ann"]) for r in all_recs)
        if len(all_recs) != EXPECTED_N_RECORDINGS:
            log.warning("recordings on disk = %d (expected %d)", len(all_recs), EXPECTED_N_RECORDINGS)
        if n_pat != EXPECTED_N_PATIENTS:
            log.warning("patients = %d (expected %d)", n_pat, EXPECTED_N_PATIENTS)
        if n_cycles != EXPECTED_N_CYCLES:
            log.warning("cycles = %d (expected %d)", n_cycles, EXPECTED_N_CYCLES)

    # recordings on disk but absent from split.txt -> dropped (never trained/tested on)
    recs, dropped = [], []
    for r in all_recs:
        if r["rec_id"] in split_map:
            r["split"] = split_map[r["rec_id"]]
            recs.append(r)
        else:
            dropped.append(r)
    for r in dropped:
        log.warning("DROPPED (absent from split.txt): %s", r["rec_id"])

    # known patient-level leakage (156/218): kept per official split, just flagged
    pat_splits = defaultdict(set)
    for r in recs:
        pat_splits[r["patient"]].add(r["split"])
    leaked = sorted(p for p, s in pat_splits.items() if len(s) > 1)
    if leaked:
        log.warning("LEAKAGE FLAG: patients %s have recordings in both splits (kept per official split)", leaked)
    if set(leaked) != EXPECTED_LEAKED_PATIENTS and smoke_stride == 1:
        log.warning("Leaked-patient set %s differs from expected %s", leaked, sorted(EXPECTED_LEAKED_PATIENTS))

    if smoke_stride > 1:
        recs = recs[::smoke_stride]
        log.warning("SMOKE MODE: using %d recordings", len(recs))

    cycles = []
    for ridx, r in enumerate(recs):
        for (s, e, c, wz) in r["ann"]:
            cycles.append(dict(cid=len(cycles), ridx=ridx, start=s, end=e,
                               label=int(c) + 2 * int(wz), split=r["split"]))
    log.info("Retained: %d recordings, %d cycles | train=%d test=%d",
             len(recs), len(cycles), sum(1 for c in cycles if c["split"] == "train"),
             sum(1 for c in cycles if c["split"] == "test"))
    ctr = Counter((c["split"], c["label"]) for c in cycles)
    for sp in ("train", "test"):
        log.info("  %s class counts: %s", sp, {CLASS_NAMES[k]: ctr.get((sp, k), 0) for k in range(4)})
    return recs, cycles


# =====================================================================================
# WAVEFORM CACHE (S1 = C0: resample only, no filtering)
# =====================================================================================
def load_resample(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def wave_path(rec_id):
    return os.path.join(CACHE_DIR, "wave_c0", rec_id + ".npy")


def ensure_wave_cache(recs):
    d = os.path.join(CACHE_DIR, "wave_c0")
    os.makedirs(d, exist_ok=True)
    for r in tqdm(recs, desc="cache raw waveforms (C0)"):
        dst = wave_path(r["rec_id"])
        if os.path.exists(dst):
            continue
        x = load_resample(r["wav"])
        tmp = dst + ".tmp.npy"
        np.save(tmp, x)
        os.replace(tmp, dst)


# =====================================================================================
# S2 WINDOWING: W4 -- sliding 50%-overlap multi-crop
# =====================================================================================
def _loop_center(x, n):
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    y = np.tile(x, int(math.ceil(n / len(x))))
    s = (len(y) - n) // 2
    return y[s:s + n]


def sliding_crops(x, n):
    """W4: sliding windows, 50% overlap, subsampled to <= MAX_CROPS."""
    if len(x) <= n:
        return _loop_center(x, n)[None]
    hop = max(1, int(n * WINDOW_HOP_FRAC))
    starts = list(range(0, len(x) - n + 1, hop))
    if starts[-1] != len(x) - n:
        starts.append(len(x) - n)
    if len(starts) > MAX_CROPS:
        sel = np.round(np.linspace(0, len(starts) - 1, MAX_CROPS)).astype(int)
        starts = [starts[i] for i in sel]
    return np.stack([x[s:s + n] for s in starts])


def win_prefix():
    return os.path.join(CACHE_DIR, "win_c0_w4_%ds" % WINDOW_SEC)


def ensure_window_cache(recs, cycles):
    """Build (once) the crop bank for C0 / W4 / 3s, reused by every seed/epoch."""
    prefix = win_prefix()
    if all(os.path.exists(prefix + s) for s in (".npy", "_off.npy", "_cnt.npy")):
        return prefix
    ensure_wave_cache(recs)
    n = WINDOW_SEC * SR
    by_rec = defaultdict(list)
    for c in cycles:
        by_rec[c["ridx"]].append(c)
    crops, offsets, counts, total = [], np.zeros(len(cycles), np.int64), np.zeros(len(cycles), np.int64), 0
    for ridx in tqdm(sorted(by_rec), desc="build W4/3s window cache"):
        w = np.load(wave_path(recs[ridx]["rec_id"]), mmap_mode="r")
        for c in by_rec[ridx]:
            s = min(int(round(c["start"] * SR)), max(0, len(w) - 1))
            e = min(int(round(c["end"] * SR)), len(w))
            seg = np.asarray(w[s:e], dtype=np.float32)
            if len(seg) == 0:
                seg = np.zeros(SR // 10, dtype=np.float32)
            cr = sliding_crops(seg, n)
            offsets[c["cid"]], counts[c["cid"]] = total, len(cr)
            total += len(cr)
            crops.append(cr)
    arr = np.concatenate(crops, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for suffix, a in (("_off.npy", offsets), ("_cnt.npy", counts), (".npy", arr)):
        tmp = "%s%s.tmp.npy" % (prefix, suffix)
        np.save(tmp, a)
        os.replace(tmp, prefix + suffix)
    log.info("window cache: %d crops (%.2f/cycle), %.2f GB", len(arr), len(arr) / len(cycles), arr.nbytes / 1e9)
    return prefix


def frames_for(L):
    return 1 + (L * SR - 400) // 160          # Kaldi fbank: 25ms window, 10ms shift


def ckpt_path(seed):
    return os.path.join(CKPT_DIR, "seed%d.pt" % seed)


# =====================================================================================
# S3-C: Mixup / SpecMix (applied at the batch level, train-only)
# =====================================================================================
def apply_mix(x, mask, y):
    """x [B,K,T,F], mask [B,K] -> mixed x, mask, y_a, y_b, lam."""
    B = x.size(0)
    if B < 2 or random.random() > MIX_P:
        return x, mask, y, y, 1.0
    perm = torch.randperm(B, device=x.device)
    lam = float(np.random.beta(MIX_ALPHA, MIX_ALPHA))
    xp = x[perm]
    if random.random() < SPECMIX_PROB:                     # SpecMix: freq stripe + time stripe from partner
        T, Fq = x.shape[-2], x.shape[-1]
        a = 1.0 - math.sqrt(max(1e-6, lam))
        fw, tw = max(1, int(round(a * Fq))), max(1, int(round(a * T)))
        f0, t0 = random.randint(0, Fq - fw), random.randint(0, T - tw)
        xm = x.clone()
        xm[..., f0:f0 + fw] = xp[..., f0:f0 + fw]
        xm[..., t0:t0 + tw, :] = xp[..., t0:t0 + tw, :]
        lam = (1.0 - fw / Fq) * (1.0 - tw / T)
    else:                                                   # plain mixup
        xm = lam * x + (1.0 - lam) * xp
    return xm, (mask | mask[perm]), y, y[perm], lam


# =====================================================================================
# DATASET / LOADER
# =====================================================================================
class Ctx:
    def __init__(self, recs, cycles):
        self.recs, self.cycles = recs, cycles
        self.labels = np.array([c["label"] for c in cycles], dtype=np.int64)
        self.train_ids = [c["cid"] for c in cycles if c["split"] == "train"]
        self.test_ids = [c["cid"] for c in cycles if c["split"] == "test"]
        self._fe = None

    def feature_extractor(self, frames):
        if self._fe is None:
            from transformers import ASTFeatureExtractor
            fe = ASTFeatureExtractor.from_pretrained(MODEL_NAME, local_files_only=_P.OFFLINE)
            fe.max_length = frames
            self._fe = fe
        return self._fe


class CycleDS(Dataset):
    def __init__(self, ids, labels, prefix, fe):
        self.ids, self.labels, self.prefix, self.fe = ids, labels, prefix, fe
        self.offsets = np.load(prefix + "_off.npy")
        self.counts = np.load(prefix + "_cnt.npy")
        self._mm = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.load(self.prefix + ".npy", mmap_mode="r")
        cid = self.ids[i]
        off, cnt = int(self.offsets[cid]), int(self.counts[cid])
        crops = np.array(self._mm[off:off + cnt])
        feats = self.fe(list(crops), sampling_rate=SR, return_tensors="np")["input_values"]   # [K,T,F]
        return torch.from_numpy(np.ascontiguousarray(feats)).float(), int(self.labels[cid])


def collate(batch):
    Kmax = max(b[0].size(0) for b in batch)
    xs, ms, ys = [], [], []
    for x, y in batch:
        K = x.size(0)
        idx = torch.arange(Kmax) % K                       # pad by cycling real crops
        xs.append(x[idx])
        ms.append(torch.arange(Kmax) < K)
        ys.append(y)
    return torch.stack(xs), torch.stack(ms), torch.tensor(ys, dtype=torch.long)


def _worker_init(_):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
    torch.set_num_threads(1)


# =====================================================================================
# MODEL
# =====================================================================================
def get_blocks(ast):
    for path in ("encoder.layer", "layer", "encoder.layers", "layers"):
        obj = ast
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    for _, m in ast.named_modules():
        if isinstance(m, nn.ModuleList) and len(m) == N_BLOCKS:
            return m
    raise AttributeError("Could not locate transformer blocks in %s" % type(ast).__name__)


def build_model(frames):
    from transformers import ASTForAudioClassification
    model = ASTForAudioClassification.from_pretrained(MODEL_NAME, local_files_only=_P.OFFLINE)
    cfg = model.config
    ast = model.audio_spectrogram_transformer

    # adapt pretrained positional embeddings from 1024 frames -> `frames` (center-cut in time)
    f_dim = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t_old = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
    t_new = (frames - cfg.patch_size) // cfg.time_stride + 1
    pe = ast.embeddings.position_embeddings.data
    D = pe.size(-1)
    assert pe.size(1) == 2 + f_dim * t_old
    assert t_new <= t_old
    grid = pe[:, 2:].reshape(1, f_dim, t_old, D)
    t0 = (t_old - t_new) // 2
    grid = grid[:, :, t0:t0 + t_new].reshape(1, f_dim * t_new, D)
    ast.embeddings.position_embeddings = nn.Parameter(torch.cat([pe[:, :2], grid], dim=1).clone())
    cfg.max_length = frames

    # new 4-class head (keeps pretrained head LayerNorm)
    model.classifier.dense = nn.Linear(D, NUM_CLASSES)
    nn.init.normal_(model.classifier.dense.weight, std=0.02)
    nn.init.zeros_(model.classifier.dense.bias)
    cfg.num_labels = NUM_CLASSES

    # freeze embeddings + first FREEZE_BLOCKS transformer blocks
    for p in ast.embeddings.parameters():
        p.requires_grad = False
    blocks = get_blocks(ast)
    assert len(blocks) == N_BLOCKS
    for i in range(FREEZE_BLOCKS):
        for p in blocks[i].parameters():
            p.requires_grad = False
    return model


def model_logits(model, x, mask, train):
    """x [B,K,T,F], mask [B,K] -> pooled-over-crops logits [B,C] (W4 = mean pooling)."""
    B, K = mask.shape
    flat = x[mask]
    h = model.audio_spectrogram_transformer(input_values=flat).last_hidden_state
    cls_t, dist_t = h[:, 0], h[:, 1]
    lf = model.classifier((cls_t + dist_t) / 2.0)
    if K == 1:
        return lf
    logits = lf.new_zeros(B, K, lf.size(-1))
    logits[mask] = lf
    m = mask[..., None].to(logits.dtype)
    return (logits * m).sum(1) / m.sum(1)         # POOL == "mean"


# =====================================================================================
# METRICS
# =====================================================================================
def compute_metrics(y_true, y_pred):
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    n_norm, n_abn = cm[0].sum(), cm[1:].sum()
    sp = cm[0, 0] / max(1, n_norm)
    se = sum(cm[i, i] for i in (1, 2, 3)) / max(1, n_abn)
    as_ = (se + sp) / 2.0
    hs = 2 * se * sp / (se + sp + 1e-12)
    rec = [cm[i, i] / max(1, cm[i].sum()) for i in range(4)]
    m = {"SE": 100 * se, "SP": 100 * sp, "AS": 100 * as_, "HS": 100 * hs}
    for k, n in enumerate(CLASS_NAMES):
        m["recall_" + n] = 100 * rec[k]
    return m, cm


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    ys, ps = [], []
    for x, mask, y in loader:
        x, mask = x.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            lg = model_logits(model, x, mask, train=False)
        ps.append(lg.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# =====================================================================================
# TRAIN ONE SEED
# =====================================================================================
def run_one_seed(ctx, prefix, frames, seed):
    log.info("=" * 100)
    log.info("[START] C0 / W4(mean) / 3s / S3-C(Mixup+SpecMix) | seed=%d | epochs=%d", seed, EPOCHS)
    device = get_device()
    amp = USE_AMP and device.type == "cuda"
    seed_everything(seed)

    fe = ctx.feature_extractor(frames)
    train_ds = CycleDS(ctx.train_ids, ctx.labels, prefix, fe)
    test_ds = CycleDS(ctx.test_ids, ctx.labels, prefix, fe)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=device.type == "cuda",
              worker_init_fn=_worker_init)
    if NUM_WORKERS > 0:
        kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              persistent_workers=NUM_WORKERS > 0, **kw)
    test_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, **kw)

    model = build_model(frames).to(device)
    ast = model.audio_spectrogram_transformer
    bb = [p for n, p in ast.named_parameters() if p.requires_grad]
    head = list(model.classifier.parameters())

    def _groups(params, lr):
        dec = [p for p in params if p.ndim > 1]
        nod = [p for p in params if p.ndim <= 1]
        return [{"params": dec, "lr": lr, "weight_decay": WEIGHT_DECAY},
                {"params": nod, "lr": lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(_groups(bb, LR_BACKBONE) + _groups(head, LR_HEAD))
    log.info("trainable params: %.2fM | train cycles=%d test cycles=%d | frames=%d",
             (sum(p.numel() for p in bb) + sum(p.numel() for p in head)) / 1e6,
             len(ctx.train_ids), len(ctx.test_ids), frames)

    steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
    total = steps_per_epoch * EPOCHS
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    t0 = time.time()
    final_loss = float("nan")
    class_dist_log = []              # persistent record: one row per (epoch, class) -- written to CSV at the end
    run_hard_total = np.zeros(NUM_CLASSES, dtype=np.int64)     # whole-run tallies (all epochs summed)
    run_eff_total = np.zeros(NUM_CLASSES, dtype=np.float64)
    for ep in range(EPOCHS):
        model.train()
        run_loss, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        nb = len(train_loader)
        epoch_hard = np.zeros(NUM_CLASSES, dtype=np.int64)     # actual sample labels seen this epoch
        epoch_eff = np.zeros(NUM_CLASSES, dtype=np.float64)    # lam/(1-lam)-weighted loss exposure this epoch
        for bi, (x, mask, y) in enumerate(tqdm(train_loader, desc="ep%d/%d seed%d" % (ep + 1, EPOCHS, seed))):
            x, mask, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, mask, ya, yb, lam = apply_mix(x, mask, y)     # S3-C

            # --- classwise composition of what's actually being fed to the model this batch ---
            cnt_a = torch.bincount(ya, minlength=NUM_CLASSES).cpu().numpy()
            cnt_b = torch.bincount(yb, minlength=NUM_CLASSES).cpu().numpy()
            epoch_hard += cnt_a                                          # real labels present (perm doesn't change counts)
            epoch_eff += lam * cnt_a + (1.0 - lam) * cnt_b               # what the loss actually weighted them as
            if (bi + 1) % LOG_EVERY_BATCHES == 0:                        # don't spam a line every single batch
                if lam == 1.0:
                    classwise = ", ".join("%s=%d" % (n, c) for n, c in zip(CLASS_NAMES, cnt_a.tolist()))
                    log.info("  [batch %d input] no mix (lam=1.0) | %s", bi + 1, classwise)
                else:
                    classwise_a = ", ".join("%s=%d" % (n, c) for n, c in zip(CLASS_NAMES, cnt_a.tolist()))
                    classwise_b = ", ".join("%s=%d" % (n, c) for n, c in zip(CLASS_NAMES, cnt_b.tolist()))
                    log.info("  [batch %d input] mixed lam=%.3f | A(%s) | B(%s)", bi + 1, lam, classwise_a, classwise_b)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model_logits(model, x, mask, train=True)
            logits = logits.float()
            loss = F.cross_entropy(logits, ya) if lam == 1.0 else \
                lam * F.cross_entropy(logits, ya) + (1.0 - lam) * F.cross_entropy(logits, yb)
            scaler.scale(loss / ACCUM_STEPS).backward()
            if (bi + 1) % ACCUM_STEPS == 0 or (bi + 1) == nb:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            run_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
            if (bi + 1) % LOG_EVERY_BATCHES == 0:
                log.info("  ep %d/%d batch %d/%d | loss %.4f | lr_bb %.2e", ep + 1, EPOCHS, bi + 1, nb,
                         run_loss / n_seen, opt.param_groups[0]["lr"])
        final_loss = run_loss / max(1, n_seen)
        run_hard_total += epoch_hard
        run_eff_total += epoch_eff
        hard_str = ", ".join("%s=%d" % (n, c) for n, c in zip(CLASS_NAMES, epoch_hard.tolist()))
        eff_str = ", ".join("%s=%.1f" % (n, c) for n, c in zip(CLASS_NAMES, epoch_eff.tolist()))
        log.info("[epoch %d/%d] train_loss=%.4f | actual classes seen: %s | effective (mix-weighted): %s",
                 ep + 1, EPOCHS, final_loss, hard_str, eff_str)
        for k, n in enumerate(CLASS_NAMES):
            class_dist_log.append(dict(seed=seed, epoch=ep + 1, cls=n,
                                        hard_count=int(epoch_hard[k]), effective_weight=float(epoch_eff[k])))

    dist_path = os.path.join(OUT_DIR, "class_distribution_seed%d.csv" % seed)
    with open(dist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "epoch", "cls", "hard_count", "effective_weight"])
        w.writeheader()
        for r in class_dist_log:
            w.writerow(r)
    log.info("[record] classwise training-input distribution (per epoch) written to %s", dist_path)
    log.info("[record] whole-run totals | actual: %s | effective: %s",
             ", ".join("%s=%d" % (n, c) for n, c in zip(CLASS_NAMES, run_hard_total.tolist())),
             ", ".join("%s=%.1f" % (n, c) for n, c in zip(CLASS_NAMES, run_eff_total.tolist())))

    yt, yp = evaluate(model, test_loader, device, amp)
    m, cm = compute_metrics(yt, yp)
    log.info("[DONE ] seed=%d | SE=%.2f SP=%.2f AS=%.2f HS=%.2f | recalls N/C/W/B = %.1f/%.1f/%.1f/%.1f | %.0fs",
             seed, m["SE"], m["SP"], m["AS"], m["HS"],
             m["recall_normal"], m["recall_crackle"], m["recall_wheeze"], m["recall_both"], time.time() - t0)
    log.info("       confusion (rows=true N/C/W/B): %s", cm.tolist())

    os.makedirs(CKPT_DIR, exist_ok=True)
    dst = ckpt_path(seed)
    tmp = dst + ".tmp"
    torch.save({"seed": seed, "frames": frames, "state_dict": model.state_dict()}, tmp)
    os.replace(tmp, dst)
    log.info("[record] checkpoint saved to %s", dst)

    row = dict(seed=seed, epochs=EPOCHS, final_train_loss=final_loss, n_train=len(ctx.train_ids),
               n_test=len(ctx.test_ids), confusion=cm.tolist(), train_time_s=time.time() - t0)
    row.update(m)
    for k, n in enumerate(CLASS_NAMES):
        row["train_hard_%s" % n] = int(run_hard_total[k])
        row["train_effective_%s" % n] = float(run_eff_total[k])

    del train_loader, test_loader, model, opt, sched, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return row


# =====================================================================================
# MAIN
# =====================================================================================
def main():
    global OUT_DIR, CACHE_DIR, CKPT_DIR, EPOCHS, SEEDS, NUM_WORKERS, LOG_EVERY_BATCHES
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny sanity run (subset, 1 epoch, 1 seed)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    if args.out_dir:
        OUT_DIR = args.out_dir
    stride = 1
    if args.smoke:
        OUT_DIR = OUT_DIR.rstrip("/") + "_smoke"
        EPOCHS = 1
        SEEDS = [13]
        NUM_WORKERS, LOG_EVERY_BATCHES = 2, 5
        stride = 38
    if args.num_workers is not None:
        NUM_WORKERS = args.num_workers
    CACHE_DIR = os.path.join(_P.OUT, "cache")
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    setup_logging(OUT_DIR)

    log.info("#" * 100)
    log.info("Optimal pipeline: S1=C0 S2=W4(mean) L=3s S3=S3-C(Mixup+SpecMix) | epochs=%d seeds=%s", EPOCHS, SEEDS)
    dev = get_device()
    log.info("torch %s | device=%s%s", torch.__version__, dev.type,
             (" (%s)" % torch.cuda.get_device_name(0)) if dev.type == "cuda" else "")
    log.warning("!!! NO VALIDATION SPLIT: this pipeline was selected via screening on the official TEST split; "
                "the numbers below are optimistic w.r.t. config selection !!!")

    recs, cycles = build_dataset(smoke_stride=stride)
    ctx = Ctx(recs, cycles)
    prefix = ensure_window_cache(recs, cycles)
    frames = frames_for(WINDOW_SEC)

    rows = [run_one_seed(ctx, prefix, frames, s) for s in SEEDS]

    summ = {
        "config": {"s1": "C0", "s2": "W4", "pool": "mean", "L": WINDOW_SEC, "s3": "S3-C",
                   "mix_alpha": MIX_ALPHA, "mix_p": MIX_P, "specmix_prob": SPECMIX_PROB},
        "seeds": SEEDS, "epochs": EPOCHS,
        "note": "Evaluated on the official ICBHI test split with no held-out val set; config selection used the "
                "same test split (optimistic). Patients 156/218 leak across the split (kept, flagged).",
    }
    for k in ["SE", "SP", "AS", "HS", "recall_normal", "recall_crackle", "recall_wheeze", "recall_both"]:
        v = np.array([r[k] for r in rows])
        summ[k] = {"mean": float(v.mean()), "std": float(v.std())}
    summ["train_class_distribution"] = {
        n: {"hard_count": [r["train_hard_%s" % n] for r in rows],
            "effective_weight": [r["train_effective_%s" % n] for r in rows]}
        for n in CLASS_NAMES
    }
    out_path = os.path.join(OUT_DIR, "final_summary.json")
    with open(out_path, "w") as f:
        json.dump(summ, f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "per_seed_results.csv"), "w", newline="") as f:
        fields = ["seed", "epochs", "SE", "SP", "AS", "HS", "recall_normal", "recall_crackle",
                  "recall_wheeze", "recall_both", "final_train_loss", "n_train", "n_test",
                  "confusion", "train_time_s"] + \
                 ["train_hard_%s" % n for n in CLASS_NAMES] + ["train_effective_%s" % n for n in CLASS_NAMES]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    log.info("=" * 100)
    log.info("FINAL RESULT (%d seeds): C0 / W4(mean) / 3s / S3-C", len(SEEDS))
    for k in ["SE", "SP", "AS", "HS", "recall_normal", "recall_crackle", "recall_wheeze", "recall_both"]:
        log.info("   %-15s %.2f +- %.2f", k, summ[k]["mean"], summ[k]["std"])
    log.info("Summary written to %s", out_path)
    log.info("=" * 100)


if __name__ == "__main__":
    main()