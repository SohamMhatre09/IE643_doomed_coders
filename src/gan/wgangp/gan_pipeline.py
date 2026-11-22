#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gan_pipeline.py -- Minority-class (crackle / wheeze / both) spectrogram
augmentation for the ICBHI AST pipeline. Sits on top of
train_optimal_pipeline.py (imported below as `B`) and reuses its wave
cache, sliding-crop windower, and AST model/feature-extractor.

INTEGRATION NOTE: 
This script now dynamically loads acoustic thresholds from 
`feature_analysis/thresholds.json` (if it exists) to replace hardcoded 
constants in Step 4. It safely falls back to hardcoded values if the 
JSON is missing, ensuring robust execution.

STEPS:
  1. Scope   -- only touch crackle / wheeze / both cycles, TRAIN split only.
  2. Duration split:
       short (<= SHORT_MAX_S) -> single window: center-crop if longer than
                                   WINDOW_SEC, zero-pad (not loop-pad) if
                                   shorter. No multi-crop scoring.
       long  (>  SHORT_MAX_S) -> Step 3 multi-crop selection.
  3. Long cycles: slide WINDOW_SEC windows (50% hop, base.sliding_crops),
     score every crop with the 3 seed checkpoints from base.py (logits
     averaged across seeds), pick the crop(s) that maximize the class-relevant logit.
  4. Acoustic sanity filter: reject a picked window if it doesn't show the
     expected crackle/wheeze acoustic signature (using dynamic thresholds 
     from feature_analysis.py, or hardcoded fallbacks); fall back to the
     next-best scored crop instead of just dropping the cycle.
  5. Device balancing: subsample within each class so no single recording
     device dominates that class's GAN training set.
  6. Train ONE class-conditional DCGAN (conditioned on {crackle, wheeze,
     both}) on log-mel spectrograms of the cleaned windows from steps 2-5. 
     Generated spectrograms are written out as augmented training data only.

Requires:
  - train_optimal_pipeline.py importable as `B`
  - `B` checkpoints already trained via `python train_optimal_pipeline.py`

Usage:
    python gan_pipeline.py [--out-dir DIR] [--joint-both] [--gan-epochs N]
"""

import os
import sys
import math
import time
import random
import json
import logging
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_optimal_pipeline as B          # noqa: E402  -- reuse config, wave cache, AST model

log = logging.getLogger("gan_pipeline")

# =====================================================================================
# CONFIG
# =====================================================================================
SHORT_MAX_S = 4.5                 # duration cutoff: short cycles skip multi-crop scoring
MINORITY_LABELS = {1: "crackle", 2: "wheeze", 3: "both"}   # base.CLASS_NAMES indices
CRACKLE, WHEEZE, BOTH = 1, 2, 3

# Step 4 acoustic sanity thresholds (HARDCODED FALLBACKS)
WHEEZE_BAND_HZ = (100, 1000)
WHEEZE_FLATNESS_MAX = 0.35        # narrowband -> low spectral flatness inside the band
WHEEZE_BAND_ENERGY_FRAC_MIN = 0.35  # fraction of total energy that must sit in-band
CRACKLE_RISE_MS_MAX = 50
CRACKLE_TRANSIENT_ZSCORE_MIN = 3.0  # envelope-derivative peak vs its own baseline

# Step 5 device balancing
DEVICE_CAP_FRAC = 0.5             # no single device > this fraction of a class's windows

# Step 6 GAN
GAN_N_MELS = 64
GAN_T = 128                       # fixed time frames per spectrogram (pad/crop)
GAN_LATENT = 100
GAN_BATCH = 32
GAN_LR = 2e-4
GAN_BETA1 = 0.5

OUT_DIR = os.path.join(B.OUT_DIR, "gan_pipeline")

# =====================================================================================
# DYNAMIC THRESHOLD LOADING (From feature_analysis.py)
# =====================================================================================
THRESHOLD_PATH = os.path.join(B.OUT_DIR, "feature_analysis", "thresholds.json")
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        DYNAMIC_THRESHOLDS = json.load(f)
    log.info("[Step 4] Loaded data-driven acoustic thresholds from %s", THRESHOLD_PATH)
else:
    DYNAMIC_THRESHOLDS = {}
    log.warning("[Step 4] thresholds.json not found. Using hardcoded fallbacks.")


def get_thresh(cls_name, feat_name, fallback_val, fallback_rule="<="):
    """Helper to get dynamic threshold from JSON, or fallback to hardcoded constant."""
    if cls_name in DYNAMIC_THRESHOLDS and feat_name in DYNAMIC_THRESHOLDS[cls_name]:
        t = DYNAMIC_THRESHOLDS[cls_name][feat_name]
        return t["threshold"], t["rule"]
    return fallback_val, fallback_rule


def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    h1 = logging.StreamHandler(sys.stdout)
    h1.setFormatter(fmt)
    h2 = logging.FileHandler(os.path.join(out_dir, "gan_pipeline.log"))
    h2.setFormatter(fmt)
    log.addHandler(h1)
    log.addHandler(h2)


# =====================================================================================
# STEP 1+2: SCOPE + DURATION SPLIT
# =====================================================================================
def scope_and_split(cycles):
    """Return (short_cycles, long_cycles): minority-class, train-split cycles only."""
    minority_train = [c for c in cycles if c["label"] in MINORITY_LABELS and c["split"] == "train"]
    short, long_ = [], []
    for c in minority_train:
        dur = c["end"] - c["start"]
        (short if dur <= SHORT_MAX_S else long_).append(c)
    log.info("[step1+2] minority train cycles: %d total | short(<=%.1fs)=%d long(>%.1fs)=%d",
             len(minority_train), SHORT_MAX_S, len(short), SHORT_MAX_S, len(long_))
    for lbl, name in MINORITY_LABELS.items():
        n_s = sum(1 for c in short if c["label"] == lbl)
        n_l = sum(1 for c in long_ if c["label"] == lbl)
        log.info("         %-8s short=%d long=%d", name, n_s, n_l)
    return short, long_


def load_cycle_wave(c, recs):
    """Raw (SR-resampled) waveform segment for one cycle, via base's wave cache."""
    w = np.load(B.wave_path(recs[c["ridx"]]["rec_id"]), mmap_mode="r")
    s = min(int(round(c["start"] * B.SR)), max(0, len(w) - 1))
    e = min(int(round(c["end"] * B.SR)), len(w))
    seg = np.asarray(w[s:e], dtype=np.float32)
    if len(seg) == 0:
        seg = np.zeros(B.SR // 10, dtype=np.float32)
    return seg


def center_crop_or_zero_pad(x, n):
    """Step 2, short cycles: center-crop if longer than n, ZERO-pad (not loop) if shorter."""
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    pad = n - len(x)
    left = pad // 2
    return np.pad(x, (left, pad - left), mode="constant", constant_values=0.0)


# =====================================================================================
# STEP 3: MULTI-SEED LOGIT-MAX WINDOW SELECTION (long cycles only)
# =====================================================================================
def load_seed_models(frames, seeds, device):
    """Load the 3 seed checkpoints saved by train_optimal_pipeline.run_one_seed."""
    models = []
    for s in seeds:
        p = B.ckpt_path(s)
        if not os.path.exists(p):
            raise FileNotFoundError(
                "Missing checkpoint %s -- run `python train_optimal_pipeline.py` first "
                "(it now saves outputs_optimal_pipeline/checkpoints/seed%d.pt)." % (p, s))
        ckpt = torch.load(p, map_location=device)
        m = B.build_model(ckpt.get("frames", frames)).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        models.append(m)
    log.info("[step3] loaded %d seed checkpoints: %s", len(models), seeds)
    return models


@torch.no_grad()
def score_crops(models, crops, fe, device):
    """crops: [K, n_samples] raw audio -> mean logits [K, NUM_CLASSES] across seed models."""
    feats = fe(list(crops), sampling_rate=B.SR, return_tensors="np")["input_values"]   # [K,T,F]
    x = torch.from_numpy(np.ascontiguousarray(feats)).float().to(device)
    acc = None
    for m in models:
        h = m.audio_spectrogram_transformer(input_values=x).last_hidden_state
        lg = m.classifier((h[:, 0] + h[:, 1]) / 2.0)          # [K, NUM_CLASSES]
        acc = lg if acc is None else acc + lg
    return (acc / len(models)).cpu().numpy()


def rank_crops_for_cycle(label, logits):
    """Return crop indices ranked best-first for this cycle's class (Step 3 rule)."""
    if label == CRACKLE:
        score = logits[:, CRACKLE]
    elif label == WHEEZE:
        score = logits[:, WHEEZE]
    else:  # BOTH: joint score = logit_crackle + logit_wheeze
        score = logits[:, CRACKLE] + logits[:, WHEEZE]
    return np.argsort(-score)


def rank_crops_for_cycle_split_both(logits):
    """Alternative for BOTH cycles (--joint-both off): independent argmax per component."""
    return np.argsort(-logits[:, CRACKLE]), np.argsort(-logits[:, WHEEZE])


# =====================================================================================
# STEP 4: ACOUSTIC SANITY FILTER (Now uses dynamic thresholds)
# =====================================================================================
def _spectral_flatness(mag):
    mag = np.clip(mag, 1e-10, None)
    gmean = np.exp(np.mean(np.log(mag)))
    amean = np.mean(mag)
    return float(gmean / amean)


def acoustic_check_wheeze(x, sr=B.SR):
    """Sustained narrowband energy, tonal (low spectral flatness)."""
    nperseg = 1024
    hop = 256
    lo, hi = WHEEZE_BAND_HZ
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    if band.sum() == 0 or len(x) < nperseg:
        return False
        
    win = np.hanning(nperseg)
    n_frames = max(1, (len(x) - nperseg) // hop + 1)
    band_frac_list, band_flat_list = [], []
    
    for i in range(n_frames):
        seg = x[i * hop: i * hop + nperseg]
        if len(seg) < nperseg:
            break
        mag = np.abs(np.fft.rfft(seg * win))
        total = mag.sum() + 1e-10
        
        # Use the 100-300 Hz band, as feature_analysis.py proved it's the strongest separator
        band_100_300 = (freqs >= 100) & (freqs <= 300)
        band_frac_list.append(mag[band_100_300].sum() / total)
        band_flat_list.append(_spectral_flatness(mag[band]))
        
    if not band_frac_list:
        return False
        
    # --- DYNAMIC THRESHOLD OVERRIDE ---
    # Matches the EXACT feature names output by feature_analysis.py
    thr_flat, rule_flat = get_thresh("wheeze", "spectral_flatness", WHEEZE_FLATNESS_MAX, "<=")
    thr_frac, rule_frac = get_thresh("wheeze", "band_100_300_frac", 0.35, ">=")
    
    # Apply rules dynamically
    good_flat = (np.array(band_flat_list) <= thr_flat) if rule_flat == "<=" else (np.array(band_flat_list) >= thr_flat)
    good_frac = (np.array(band_frac_list) >= thr_frac) if rule_frac == ">=" else (np.array(band_frac_list) <= thr_frac)
    
    sustained_frac = (good_flat & good_frac).mean()
    return bool(sustained_frac >= 0.3)

def acoustic_check_crackle(x, sr=B.SR):
    """Broadband transient with fast envelope rise. Uses dynamic transient_count if available."""
    analytic = np.abs(_hilbert_envelope(x))
    if len(analytic) < 3:
        return False
        
    win_smooth = max(1, int(0.005 * sr))        # 5ms smoothing to denoise envelope
    if win_smooth > 1:
        kernel = np.ones(win_smooth) / win_smooth
        env = np.convolve(analytic, kernel, mode="same")
    else:
        env = analytic
        
    rise_n = max(1, int(CRACKLE_RISE_MS_MAX / 1000.0 * sr))
    d = env[rise_n:] - env[:-rise_n]            # amplitude gained over each `rise_n`-sample span
    if len(d) == 0:
        return False
        
    mu, sigma = d.mean(), d.std() + 1e-8
    peak_z = (d.max() - mu) / sigma
    
    # --- DYNAMIC THRESHOLD OVERRIDE ---
    # feature_analysis.py outputs 'transient_count'. We replicate its counting logic (z >= 3.0)
    # and compare the total count against the dynamic threshold.
    thr_count, rule_count = get_thresh("crackle", "transient_count", None, ">=")
    
    if thr_count is not None:
        z = (d - mu) / sigma
        count = int((z >= 3.0).sum())  # Match feature_analysis.py's TRANSIENT_Z = 3.0
        if rule_count == ">=":
            return bool(count >= thr_count)
        else:
            return bool(count <= thr_count)
    else:
        # Fallback to original peak_z logic
        return bool(peak_z >= CRACKLE_TRANSIENT_ZSCORE_MIN)


def _hilbert_envelope(x):
    """Analytic-signal envelope via FFT-based Hilbert transform (no scipy.signal.hilbert dep beyond fft)."""
    n = len(x)
    Xf = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(Xf * h)


def passes_acoustic_check(label, x):
    if label == CRACKLE:
        return acoustic_check_crackle(x)
    if label == WHEEZE:
        return acoustic_check_wheeze(x)
    # BOTH must show *both* signatures somewhere in the window
    return acoustic_check_crackle(x) and acoustic_check_wheeze(x)


# =====================================================================================
# STEPS 2-4 DRIVER: build the cleaned per-cycle window set
# =====================================================================================
def select_windows(recs, cycles, models, fe, device, joint_both=True):
    short, long_ = scope_and_split(cycles)
    n = B.WINDOW_SEC * B.SR
    selected = []   # list of dict(cid, label, ridx, audio[np.float32], equip, source)

    # --- short cycles: single deterministic window, no scoring/filtering ---
    for c in short:
        seg = load_cycle_wave(c, recs)
        win = center_crop_or_zero_pad(seg, n)
        selected.append(dict(cid=c["cid"], label=c["label"], ridx=c["ridx"], audio=win,
                              equip=recs[c["ridx"]]["equip"], source="short"))  # <-- FIXED: 'arecs' -> 'recs'

    # --- long cycles: multi-crop scoring + acoustic sanity filter with fallback ---
    n_kept, n_dropped, n_fallback = 0, 0, 0
    for c in long_:
        seg = load_cycle_wave(c, recs)
        crops = B.sliding_crops(seg, n)                       # [K, n]
        logits = score_crops(models, crops, fe, device)       # [K, NUM_CLASSES]

        if c["label"] == BOTH and not joint_both:
            order_c, order_w = rank_crops_for_cycle_split_both(logits)
            for order, tag in ((order_c, "both-crackle-evidence"), (order_w, "both-wheeze-evidence")):
                picked, _ = _pick_with_fallback(order, crops, c["label"])
                if picked is not None:
                    selected.append(dict(cid=c["cid"], label=c["label"], ridx=c["ridx"], audio=picked,
                                          equip=recs[c["ridx"]]["equip"], source=tag))
                    n_kept += 1
                else:
                    n_dropped += 1
            continue

        order = rank_crops_for_cycle(c["label"], logits)
        picked, used_fallback = _pick_with_fallback(order, crops, c["label"])
        if picked is not None:
            selected.append(dict(cid=c["cid"], label=c["label"], ridx=c["ridx"], audio=picked,
                                  equip=recs[c["ridx"]]["equip"], source="long"))
            n_kept += 1
            n_fallback += int(used_fallback)
        else:
            n_dropped += 1

    log.info("[step3+4] long cycles: kept=%d dropped(all crops failed sanity check)=%d "
             "fell-back-to-non-top-crop=%d", n_kept, n_dropped, n_fallback)
    log.info("[step2-4] total selected windows (short+long): %d", len(selected))
    return selected

def _pick_with_fallback(order, crops, label):
    """Walk the ranked crop list; return (crop, used_fallback) for the first crop that
    passes the Step-4 sanity check, or (None, False) if every candidate fails it
    (cycle dropped)."""
    for rank, idx in enumerate(order):
        cand = crops[idx]
        if passes_acoustic_check(label, cand):
            return cand, rank > 0
    return None, False


# =====================================================================================
# STEP 5: DEVICE BALANCING
# =====================================================================================
def balance_devices(selected, cap_frac=DEVICE_CAP_FRAC, seed=0):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for w in selected:
        by_class[w["label"]].append(w)

    balanced = []
    for label, windows in by_class.items():
        by_device = defaultdict(list)
        for w in windows:
            by_device[w["equip"]].append(w)
        n_total = len(windows)
        cap = max(1, int(math.floor(cap_frac * n_total)))
        kept = []
        for dev, ws in by_device.items():
            if len(ws) > cap:
                rng.shuffle(ws)
                ws = ws[:cap]
            kept.extend(ws)
        balanced.extend(kept)
        dist_before = {d: len(ws) for d, ws in by_device.items()}
        dist_after = {}
        for w in kept:
            dist_after[w["equip"]] = dist_after.get(w["equip"], 0) + 1
        log.info("[step5] class=%-8s before=%s -> after=%s (cap=%d, %.0f%% of %d)",
                 MINORITY_LABELS[label], dist_before, dist_after, cap, 100 * cap_frac, n_total)
    log.info("[step5] total windows after device balancing: %d (was %d)", len(balanced), len(selected))
    return balanced


# =====================================================================================
# STEP 6: CLASS-CONDITIONAL DCGAN (trained jointly on crackle/wheeze/both, label-conditioned)
# =====================================================================================
def log_mel_spectrogram(x, sr=B.SR, n_mels=GAN_N_MELS, t_frames=GAN_T):
    """Log-mel spectrogram, fixed to [n_mels, t_frames] via center-crop/zero-pad in time."""
    import torchaudio
    xt = torch.from_numpy(np.ascontiguousarray(x)).float()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=max(1, len(x) // t_frames),
        n_mels=n_mels, power=2.0)(xt)
    mel = torch.log(mel.clamp_min(1e-10))
    T = mel.size(-1)
    if T >= t_frames:
        s0 = (T - t_frames) // 2
        mel = mel[:, s0:s0 + t_frames]
    else:
        mel = F.pad(mel, (0, t_frames - T))
    # normalize to roughly [-1, 1] for the GAN (tanh output)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    mel = mel.clamp(-4, 4) / 4.0
    return mel.numpy().astype(np.float32)       # [n_mels, t_frames]


class SpecDataset(Dataset):
    """Minority-class log-mel spectrograms, label-mapped to {0:crackle,1:wheeze,2:both}."""
    LABEL_MAP = {CRACKLE: 0, WHEEZE: 1, BOTH: 2}

    def __init__(self, windows):
        self.windows = windows
        self.specs = []
        self.labels = []
        
        log.info("[step6] Precomputing spectrograms for %d windows (this happens only once)...", len(windows))
        for i, w in enumerate(windows):
            spec = log_mel_spectrogram(w["audio"])
            self.specs.append(torch.from_numpy(spec)[None])  # [1, H, W]
            self.labels.append(self.LABEL_MAP[w["label"]])
            
            # Print progress every 500 windows so you know it's working
            if (i + 1) % 500 == 0:
                log.info("  Precomputed %d / %d spectrograms", i + 1, len(windows))
        log.info("[step6] Spectrogram precomputation complete. Starting training.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        # Now it just returns the pre-computed tensor instantly!
        return self.specs[i], self.labels[i]


class Generator(nn.Module):
    def __init__(self, n_classes=3, latent=GAN_LATENT, n_mels=GAN_N_MELS, t_frames=GAN_T):
        super().__init__()
        self.emb = nn.Embedding(n_classes, latent)
        self.h0, self.w0 = n_mels // 16, t_frames // 16
        self.fc = nn.Linear(latent * 2, 256 * self.h0 * self.w0)
        self.net = nn.Sequential(
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z, y):
        c = self.emb(y)
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.view(-1, 256, self.h0, self.w0)
        return self.net(h)


class Discriminator(nn.Module):
    def __init__(self, n_classes=3, n_mels=GAN_N_MELS, t_frames=GAN_T):
        super().__init__()
        self.emb = nn.Embedding(n_classes, n_mels * t_frames)
        self.n_mels, self.t_frames = n_mels, t_frames
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
        )
        h0, w0 = n_mels // 16, t_frames // 16
        self.fc = nn.Linear(256 * h0 * w0, 1)

    def forward(self, x, y):
        c = self.emb(y).view(-1, 1, self.n_mels, self.t_frames)
        h = self.net(torch.cat([x, c], dim=1))
        return self.fc(h.flatten(1))

def train_gan(windows, out_dir, epochs=200, device=None):
    device = device or B.get_device()
    os.makedirs(out_dir, exist_ok=True)   # <-- moved up here, before any saves happen
    ds = SpecDataset(windows)
    dl = DataLoader(ds, batch_size=min(GAN_BATCH, max(2, len(ds))), shuffle=True, drop_last=len(ds) >= GAN_BATCH)

    G = Generator().to(device)
    D = Discriminator().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=GAN_LR, betas=(GAN_BETA1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    log.info("[step6] training conditional DCGAN | windows=%d | epochs=%d | device=%s",
             len(ds), epochs, device.type)
    t0 = time.time()
    
    for ep in range(epochs):
        d_loss_sum = g_loss_sum = 0.0
        nb = 0
        epoch_start = time.time()
        
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            b = x.size(0)
            real_lbl = torch.ones(b, 1, device=device)
            fake_lbl = torch.zeros(b, 1, device=device)

            # --- D step ---
            z = torch.randn(b, GAN_LATENT, device=device)
            fake = G(z, y).detach()
            optD.zero_grad(set_to_none=True)
            d_loss = bce(D(x, y), real_lbl) + bce(D(fake, y), fake_lbl)
            d_loss.backward()
            optD.step()

            # --- G step ---
            z = torch.randn(b, GAN_LATENT, device=device)
            fake = G(z, y)
            optG.zero_grad(set_to_none=True)
            g_loss = bce(D(fake, y), real_lbl)
            g_loss.backward()
            optG.step()

            d_loss_sum += float(d_loss.item())
            g_loss_sum += float(g_loss.item())
            nb += 1
            
        # ---> NOW PRINTS EVERY SINGLE EPOCH <---
        epoch_time = time.time() - epoch_start
        total_time = time.time() - t0
        log.info("  [gan] epoch %03d/%d | D=%.4f G=%.4f | epoch_time=%.1fs | total_time=%.1fm", 
                 ep + 1, epochs, 
                 d_loss_sum / max(1, nb), 
                 g_loss_sum / max(1, nb),
                 epoch_time,
                 total_time / 60.0)

        # Save a checkpoint every 50 epochs just in case
        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            torch.save(G.state_dict(), os.path.join(out_dir, f"generator_ep{ep+1}.pt"))
            torch.save(D.state_dict(), os.path.join(out_dir, f"discriminator_ep{ep+1}.pt"))
    torch.save(G.state_dict(), os.path.join(out_dir, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(out_dir, "discriminator.pt"))
    log.info("[step6] GAN training done in %.1fm, saved to %s", (time.time() - t0)/60.0, out_dir)
    return G, D


@torch.no_grad()
def generate_synthetic(G, n_per_class, out_dir, device=None):
    """Sample synthetic log-mel spectrograms per minority class, save as augmented data."""
    device = device or next(G.parameters()).device
    G.eval()
    specs, labels = [], []
    inv_map = {v: k for k, v in SpecDataset.LABEL_MAP.items()}   # {0:CRACKLE,...} -> real label ints
    for cls_idx in range(3):
        z = torch.randn(n_per_class, GAN_LATENT, device=device)
        y = torch.full((n_per_class,), cls_idx, dtype=torch.long, device=device)
        fake = G(z, y).cpu().numpy()[:, 0]         # [n, H, W]
        specs.append(fake)
        labels.extend([inv_map[cls_idx]] * n_per_class)
    specs = np.concatenate(specs, axis=0)
    labels = np.array(labels, dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "synthetic_specs.npy"), specs)
    np.save(os.path.join(out_dir, "synthetic_labels.npy"), labels)
    log.info("[step6] generated %d synthetic spectrograms (%d/class) -> %s",
             len(specs), n_per_class, out_dir)


# =====================================================================================
# MAIN
# =====================================================================================
def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--joint-both", action="store_true", default=True,
                     help="For 'both' cycles, pick one joint window (crackle+wheeze logit sum). "
                          "Pass --no-joint-both for two independent evidence windows instead.")
    ap.add_argument("--no-joint-both", dest="joint_both", action="store_false")
    ap.add_argument("--gan-epochs", type=int, default=200)
    ap.add_argument("--synth-per-class", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="tiny sanity run")
    args = ap.parse_args()

    if args.out_dir:
        OUT_DIR = args.out_dir
    setup_logging(OUT_DIR)
    log.info("#" * 100)
    log.info("GAN augmentation pipeline (Steps 1-6) | out_dir=%s | joint_both=%s",
             OUT_DIR, args.joint_both)

    device = B.get_device()
    recs, cycles = B.build_dataset(smoke_stride=38 if args.smoke else 1)
    B.ensure_wave_cache(recs)

    frames = B.frames_for(B.WINDOW_SEC)
    models = load_seed_models(frames, B.SEEDS, device)
    ctx = B.Ctx(recs, cycles)
    fe = ctx.feature_extractor(frames)

    selected = select_windows(recs, cycles, models, fe, device, joint_both=args.joint_both)
    balanced = balance_devices(selected)

    epochs = 2 if args.smoke else args.gan_epochs
    G, _ = train_gan(balanced, os.path.join(OUT_DIR, "gan"), epochs=epochs, device=device)
    generate_synthetic(G, n_per_class=(5 if args.smoke else args.synth_per_class),
                        out_dir=os.path.join(OUT_DIR, "synthetic"), device=device)

    log.info("=" * 100)
    log.info("DONE. Synthetic spectrograms + labels written under %s/synthetic/.", OUT_DIR)
    log.info("These are augmentation-only (Step 6 note): wire them into train_optimal_pipeline.py's "
             "training set as extra (spec, label) pairs mixed into each batch alongside real crops -- "
             "do NOT feed them back into window selection (Steps 3-4) to avoid circularity.")
    log.info("=" * 100)


if __name__ == "__main__":
    main()