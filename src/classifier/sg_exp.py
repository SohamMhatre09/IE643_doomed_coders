#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sg_exp.py -- stethoscope-guided cross-domain experiments on ICBHI (4-class, official split).

Based on Kim et al., "Stethoscope-guided Supervised Contrastive Learning for Cross-domain
Adaptation on Respiratory Sound Classification" (ICASSP 2024, arXiv 2312.09603).
Each recording device (Meditron / LittC2SE / Litt3200 / AKGC417L) is treated as a domain.

Experiments (all: AST, 1 fixed-length crop per cycle, official split, same seeds as RESULTS.md):

  S0a  CE baseline, weighted CE          paper's "AST fine-tuning" (59.55 in the paper)
  S0b  CE baseline, plain CE             control: isolates weighted-CE from the input setup
  S1   DAT   (gradient-reversal device head)             paper: hurts CNNs, ~= CE on AST
  S2   SG-SCL (device SupCon + GRL + stop-gradient)      paper: 61.71
  S2n  SG-SCL without stop-gradient                      ablation of the stop-grad on targets
  S2w  SG-SCL + weighted CE                              paper uses plain CE for DA methods; tests both
  S3   S0a + classical waveform aug (speed/gain/noise)   control for "GAN vs classical aug"

  --gan RUN  (any experiment) add GAN-synthetic minority cycles from gan_check/runs/RUN/ to the
             train set only; result goes to outputs/<EXP>_<RUN>/ and is registered as id <EXP>_<RUN>.

Everything this script writes stays inside experiment_stethoscope/ (outputs/<EXP>/, cache/).
When an experiment finishes with the full protocol (not --smoke) it is registered in the global
table results_summary/all_runs.csv via results_summary/register_result.py.

Usage (from anywhere):
    PY=/home/sohamd/miniconda3/envs/oral-cancer/bin/python
    $PY experiment_stethoscope/sg_exp.py --exp S0a S2          # chosen experiments
    $PY experiment_stethoscope/sg_exp.py --exp all             # everything, in priority order
    $PY experiment_stethoscope/sg_exp.py --exp all --smoke     # tiny end-to-end check

Assumptions that the paper does not pin down (flagged so they are not mistaken for paper facts):
  * cycles are cyclically padded / cut to 8 s (as in the paper's AST setup) -- my reading of [7];
  * full fine-tuning (no frozen blocks), AdamW lr 5e-5 backbone / 5e-4 new heads, batch 8, linear
    warmup(10%)+decay, SpecAugment freq<=48 / time<=192 -- common AST-ICBHI defaults, not quoted;
  * DAT / SG-SCL loss = CE + lambda * L_domain with gradient reversal (coefficient 1) on the features,
    lambda ramped 0.0096 -> 1 over 200 steps with the DANN sigmoid schedule.
"""
import os
import sys
import gc
import csv
import json
import math
import time
import random
import argparse
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P

import numpy as np
from tqdm import tqdm
import train_optimal_pipeline as tp          # reuse data parsing, AST builder, metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly

CLASS_NAMES = tp.CLASS_NAMES
DEVICES = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L"]
DEV_ID = {d: i for i, d in enumerate(DEVICES)}
SR = tp.SR
SEG_SEC = 8
SEEDS_DEFAULT = [13, 42, 97]
PAPER_SEEDS = [13, 42, 97, 7, 123]
OUT_ROOT = os.path.join(_P.OUT, "ast_runs")
FAMILY = "AST stethoscope-DA"

# --- hyper-parameters (see assumptions in the module docstring) --------------------------------
BATCH = 8
LR_BACKBONE = 5e-5
LR_HEAD = 5e-4
WD = 0.01
WARMUP_FRAC = 0.10
FREQ_MASK, TIME_MASK = 48, 192
TAU = 0.06
LAMBDA_MIN, LAMBDA_RAMP_STEPS = 0.0096, 200
GRAD_CLIP = 1.0

EXPS = {
    "S0a": dict(label="CE baseline · 8 s, weighted CE (paper AST fine-tune)", method="ce", wce=True),
    "S0b": dict(label="CE baseline · 8 s, plain CE", method="ce", wce=False),
    "S1": dict(label="DAT · device-adversarial, plain CE", method="dat", wce=False),
    "S2": dict(label="SG-SCL · device SupCon + GRL + stop-grad, plain CE", method="sgscl", wce=False, sg=True),
    "S2n": dict(label="SG-SCL without stop-gradient (ablation)", method="sgscl", wce=False, sg=False),
    "S2w": dict(label="SG-SCL · weighted CE", method="sgscl", wce=True, sg=True),
    "S3": dict(label="CE baseline + classical waveform aug (speed/gain/noise)", method="ce", wce=True, wave_aug=True),
}
ORDER = ["S0a", "S0b", "S2", "S1", "S2w", "S2n", "S3"]


# =====================================================================================
# DATA: one 8 s fixed-length crop per breathing cycle, device label from the filename
# =====================================================================================
def wave_aug(x):
    """Classical augmentation (E-S3): speed/pitch perturbation, gain, additive noise; each with p=0.5."""
    if random.random() < 0.5:
        down = int(round(100 * random.uniform(0.9, 1.1)))
        x = resample_poly(x, 100, down).astype(np.float32)
    if random.random() < 0.5:
        x = x * 10 ** (random.uniform(-3.0, 3.0) / 20.0)
    if random.random() < 0.5:
        snr = random.uniform(10.0, 30.0)
        p = float(np.mean(x ** 2)) + 1e-12
        x = x + np.random.randn(len(x)).astype(np.float32) * math.sqrt(p / 10 ** (snr / 10.0))
    return x.astype(np.float32)


class CycleWaveDS(Dataset):
    def __init__(self, ctx, ids, frames, train, use_wave_aug=False):
        self.ctx, self.ids, self.train, self.use_wave_aug = ctx, ids, train, use_wave_aug
        self.fe = ctx.feature_extractor(frames)
        self.n = SEG_SEC * SR
        self._mm = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        c = self.ctx.cycles[self.ids[i]]
        r = self.ctx.recs[c["ridx"]]
        if r["rec_id"] not in self._mm:
            self._mm[r["rec_id"]] = np.load(tp.wave_path(r["rec_id"]), mmap_mode="r")
        w = self._mm[r["rec_id"]]
        s, e = int(c["start"] * SR), int(c["end"] * SR)
        seg = np.array(w[s:max(e, s + 1600)], dtype=np.float32)
        if self.use_wave_aug and self.train:
            seg = wave_aug(seg)
        if len(seg) < self.n:
            seg = np.tile(seg, int(math.ceil(self.n / len(seg))))
        if len(seg) > self.n:
            o = random.randint(0, len(seg) - self.n) if self.train else 0
            seg = seg[o:o + self.n]
        feat = self.fe(seg, sampling_rate=SR, return_tensors="np")["input_values"][0]      # [T,F]
        return torch.from_numpy(np.ascontiguousarray(feat)).float(), c["label"], DEV_ID[r["equip"]]


GAN_RUNS_DIR = os.path.join(_P.OUT, "wgangp_runs")


class GanSynthDS(Dataset):
    """GAN-synthetic spectrograms (gan_check/methods, 8 s, 64 mel x 256 frames, [mel,time], per-sample
    normalised to [-1,1]) mapped onto the AST input exactly like the older GAN pipelines
    (mvst/synthetic_train.py GanSyntheticDS): transpose -> bilinear resize to [frames, 128 mel] ->
    per-sample z-score -> affine map to the real AST features' empirical mean/std.
    Device label = -1 (synthetic has no stethoscope) -> ignored by the DAT domain loss."""
    def __init__(self, run, frames, n_mels, real_mean, real_std, per_class, seed):
        d = os.path.join(GAN_RUNS_DIR, run)
        specs, labels = np.load(os.path.join(d, "synthetic_specs.npy")), np.load(os.path.join(d, "synthetic_labels.npy"))
        rng = np.random.RandomState(seed)
        keep = np.concatenate([rng.permutation(np.where(labels == k)[0])[:per_class] for k in (1, 2, 3)])
        x = torch.from_numpy(specs[np.sort(keep)]).float().unsqueeze(1).transpose(2, 3)    # [N,1,T,F]
        x = F.interpolate(x, size=(frames, n_mels), mode="bilinear", align_corners=False).squeeze(1)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        self.x = x * real_std + real_mean
        self.labels = labels[np.sort(keep)].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x[i], int(self.labels[i]), -1


def add_gan_synth(train_ds, frames, args, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_ds), size=64, replace=False)
    v = torch.cat([train_ds[int(i)][0].flatten() for i in idx])
    synth = GanSynthDS(args.gan, frames, train_ds.fe.num_mel_bins, float(v.mean()), float(v.std()),
                       args.synth_per_class, seed)
    tp.log.info("[gan-synth] %s: +%d synthetic (per class %s) | real feature mean=%.3f std=%.3f",
                args.gan, len(synth), np.bincount(synth.labels, minlength=4).tolist(), float(v.mean()), float(v.std()))
    return torch.utils.data.ConcatDataset([train_ds, synth]), len(synth)


def spec_augment(x):
    """x [B,T,F] -> one random freq-mask and time-mask per sample (batched on GPU)."""
    x = x.clone()
    B, T, Fq = x.shape
    for b in range(B):
        fw = random.randint(0, FREQ_MASK)
        f0 = random.randint(0, Fq - fw)
        x[b, :, f0:f0 + fw] = 0
        tw = random.randint(0, TIME_MASK)
        t0 = random.randint(0, max(0, T - tw))
        x[b, t0:t0 + tw, :] = 0
    return x


# =====================================================================================
# MODEL + LOSSES
# =====================================================================================
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


class Net(nn.Module):
    def __init__(self, frames, method, freeze_blocks):
        super().__init__()
        tp.FREEZE_BLOCKS = freeze_blocks
        self.m = tp.build_model(frames)
        if freeze_blocks == 0:                                  # tp.build_model always freezes the embeddings
            for p in self.m.audio_spectrogram_transformer.embeddings.parameters():
                p.requires_grad = True
        D = self.m.config.hidden_size
        self.dom_head = nn.Linear(D, len(DEVICES)) if method == "dat" else None
        self.proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.BatchNorm1d(D),
                                  nn.Linear(D, D), nn.ReLU(), nn.BatchNorm1d(D)) if method == "sgscl" else None

    def forward(self, x):
        h = self.m.audio_spectrogram_transformer(input_values=x).last_hidden_state
        z = (h[:, 0] + h[:, 1]) / 2.0
        return z, self.m.classifier(z)

    def head_params(self):
        ps = list(self.m.classifier.parameters())
        for mod in (self.dom_head, self.proj):
            if mod is not None:
                ps += list(mod.parameters())
        return ps


def sg_supcon(h1, h2, dev, stop_grad):
    """Device-label supervised contrastive loss (Eq. 2). Anchors keep their gradient; the targets are
    detached (stop-gradient) unless stop_grad=False. Positives = other samples with the same device."""
    f = F.normalize(torch.cat([h1, h2]).float(), dim=1)
    d = torch.cat([dev, dev])
    tgt = f.detach() if stop_grad else f
    sim = f @ tgt.t() / TAU
    eye = torch.eye(len(f), dtype=torch.bool, device=f.device)
    sim = sim.masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (d[:, None] == d[None, :]) & ~eye
    return -(logp.masked_fill(~pos, 0.0).sum(1) / pos.sum(1).clamp(min=1)).mean()


def lambda_at(step):
    p = min(1.0, step / LAMBDA_RAMP_STEPS)
    return max(LAMBDA_MIN, 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


@torch.no_grad()
def evaluate(net, loader, device, amp):
    net.eval()
    ys, ps, ds = [], [], []
    for x, y, d in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, lg = net(x.to(device, non_blocking=True))
        ps.append(lg.argmax(-1).cpu().numpy())
        ys.append(y.numpy())
        ds.append(d.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ds)


@torch.no_grad()
def evaluate_logits(net, loader, device, amp):
    """Like evaluate() but returns the raw logits (for post-hoc calibration / ensembling)."""
    net.eval()
    ys, lgs, ds = [], [], []
    for x, y, d in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, lg = net(x.to(device, non_blocking=True))
        lgs.append(lg.float().cpu().numpy())
        ys.append(y.numpy())
        ds.append(d.numpy())
    return np.concatenate(ys), np.concatenate(lgs), np.concatenate(ds)


def patient_val_split(ctx, frac, seed=0, tries=300):
    """Patient-wise validation split of the TRAIN cycles (no patient on both sides). Among `tries` random
    patient draws, take the one whose class mix is closest to the train mix (the minority classes are small)."""
    pat = {}
    for cid in ctx.train_ids:
        pat.setdefault(ctx.recs[ctx.cycles[cid]["ridx"]]["patient"], []).append(cid)
    pids = sorted(pat)
    lab = ctx.labels
    full = np.bincount(lab[ctx.train_ids], minlength=4) / len(ctx.train_ids)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(tries):
        sel = set(rng.choice(pids, size=max(1, int(round(frac * len(pids)))), replace=False).tolist())
        v = [c for p in sel for c in pat[p]]
        if not 0.8 * frac * len(ctx.train_ids) <= len(v) <= 1.2 * frac * len(ctx.train_ids):
            continue
        err = np.abs(np.bincount(lab[v], minlength=4) / len(v) - full).sum()
        if best is None or err < best[0]:
            best = (err, sorted(sel), v)
    val = set(best[2])
    return [c for c in ctx.train_ids if c not in val], sorted(val), best[1]


# =====================================================================================
# ONE SEED
# =====================================================================================
def run_seed(ctx, cfg, args, seed, frames, out_dir):
    log = tp.log
    device = tp.get_device()
    amp = device.type == "cuda"
    tp.seed_everything(seed)
    log.info("=" * 100)
    log.info("[START] %s | seed=%d epochs=%d", cfg["label"], seed, args.epochs)

    val_frac = getattr(args, "val_frac", 0.0)
    train_ids, val_ids = ctx.train_ids, []
    if val_frac > 0:
        train_ids, val_ids, vpat = patient_val_split(ctx, val_frac)
        log.info("patient-wise val split %.0f%%: train=%d cycles, val=%d cycles (%d patients) | val class counts %s, train %s",
                 100 * val_frac, len(train_ids), len(val_ids), len(vpat), np.bincount(ctx.labels[val_ids], minlength=4).tolist(),
                 np.bincount(ctx.labels[train_ids], minlength=4).tolist())
    train_ds = CycleWaveDS(ctx, train_ids, frames, True, cfg.get("wave_aug", False))
    n_synth = 0
    if args.gan:
        train_ds, n_synth = add_gan_synth(train_ds, frames, args, seed)
    test_ds = CycleWaveDS(ctx, ctx.test_ids, frames, False)
    g = torch.Generator()
    g.manual_seed(seed)
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda", worker_init_fn=tp._worker_init)
    if args.num_workers > 0:
        kw["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, generator=g,
                              persistent_workers=args.num_workers > 0, **kw)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, **kw)
    val_loader = DataLoader(CycleWaveDS(ctx, val_ids, frames, False), batch_size=16, shuffle=False, **kw) if val_ids else None

    net = Net(frames, cfg["method"], args.freeze_blocks).to(device)
    bb = [p for n, p in net.m.audio_spectrogram_transformer.named_parameters() if p.requires_grad]
    heads = net.head_params()

    def groups(params, lr):
        dec = [p for p in params if p.ndim > 1]
        nod = [p for p in params if p.ndim <= 1]
        return [{"params": dec, "lr": lr, "weight_decay": WD}, {"params": nod, "lr": lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups(bb, LR_BACKBONE) + groups(heads, LR_HEAD))
    total = len(train_loader) * args.epochs
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    cw = None
    if cfg["wce"]:                                   # inverse-frequency weights (as in the paper's weighted CE)
        cnt = np.bincount(ctx.labels[train_ids], minlength=4).astype(np.float64)
        w = 1.0 / (cnt / cnt.sum())
        cw = torch.tensor(w / w.sum(), dtype=torch.float32, device=device)
        log.info("weighted CE class weights (N/C/W/B): %s", np.round(cw.cpu().numpy(), 3).tolist())
    log.info("trainable params: backbone %.2fM + heads %.2fM | train=%d test=%d frames=%d",
             sum(p.numel() for p in bb) / 1e6, sum(p.numel() for p in heads) / 1e6,
             len(train_ids) + n_synth, len(ctx.test_ids), frames)

    t0, step, last = time.time(), 0, float("nan")
    for ep in range(args.epochs):
        net.train()
        run, n_seen, run_dom = 0.0, 0, 0.0
        for bi, (x, y, d) in enumerate(tqdm(train_loader, desc="%s ep%d/%d s%d" % (args.cur_exp, ep + 1, args.epochs, seed),
                                           mininterval=30, ncols=100)):
            x, y, d = x.to(device, non_blocking=True), y.to(device, non_blocking=True), d.to(device, non_blocking=True)
            lam = lambda_at(step)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if cfg["method"] == "sgscl":
                    z, lg = net(torch.cat([spec_augment(x), spec_augment(x)]))
                    yy = torch.cat([y, y])
                else:
                    z, lg = net(spec_augment(x))
                    yy = y
            ce = F.cross_entropy(lg.float(), yy, weight=cw)
            if cfg["method"] == "dat":
                dl = (F.cross_entropy(net.dom_head(_GRL.apply(z.float())), d, ignore_index=-1)
                      if (d >= 0).any() else z.new_zeros(()))          # synthetic samples have d = -1
                loss = ce + lam * dl
            elif cfg["method"] == "sgscl":
                h = net.proj(_GRL.apply(z.float()))
                dl = sg_supcon(h[:len(y)], h[len(y):], d, cfg["sg"])
                loss = ce + lam * dl
            else:
                dl, loss = torch.zeros(()), ce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += float(ce.item()) * len(y)
            run_dom += float(dl.item()) * len(y)
            n_seen += len(y)
        last = run / max(1, n_seen)
        log.info("[epoch %d/%d] ce=%.4f domain_loss=%.4f lambda=%.3f", ep + 1, args.epochs, last,
                 run_dom / max(1, n_seen), lam)

    yt, lg_t, dv = evaluate_logits(net, test_loader, device, amp)
    yp = lg_t.argmax(-1)
    sv = dict(test_logits=lg_t, test_y=yt, test_dev=dv)
    if val_loader is not None:
        yv, lg_v, dvv = evaluate_logits(net, val_loader, device, amp)
        sv.update(val_logits=lg_v, val_y=yv, val_dev=dvv)
    np.savez_compressed(os.path.join(out_dir, "logits_seed%d.npz" % seed), **sv)
    m, cm = tp.compute_metrics(yt, yp)
    log.info("[DONE ] seed=%d | SE=%.2f SP=%.2f AS=%.2f HS=%.2f | recalls N/C/W/B = %.1f/%.1f/%.1f/%.1f | %.0fs",
             seed, m["SE"], m["SP"], m["AS"], m["HS"], m["recall_normal"], m["recall_crackle"],
             m["recall_wheeze"], m["recall_both"], time.time() - t0)
    log.info("       confusion (rows=true N/C/W/B): %s", cm.tolist())

    per_dev = []
    for k, name in enumerate(DEVICES):
        sel = dv == k
        if sel.sum() == 0:
            continue
        md, _ = tp.compute_metrics(yt[sel], yp[sel])
        per_dev.append(dict(seed=seed, device=name, n_test=int(sel.sum()), **{k2: v for k2, v in md.items()}))
        log.info("       device %-9s n=%4d | AS=%.1f SE=%.1f SP=%.1f", name, sel.sum(), md["AS"], md["SE"], md["SP"])

    row = dict(seed=seed, epochs=args.epochs, final_train_ce=last, n_train=len(train_ids) + n_synth,
               n_test=len(ctx.test_ids), confusion=cm.tolist(), train_time_s=time.time() - t0)
    row.update(m)
    del train_loader, test_loader, net, opt, sched, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, per_dev


# =====================================================================================
# EXPERIMENT DRIVER
# =====================================================================================
KEYS = ["SE", "SP", "AS", "HS", "recall_normal", "recall_crackle", "recall_wheeze", "recall_both"]


def run_experiment(exp, ctx, args, frames):
    cfg = dict(EXPS[exp])
    if args.gan:
        cfg["label"] += " + GAN %s (%d/class)" % (args.gan, args.synth_per_class)
        exp = "%s_%s" % (exp, args.gan) + ("" if args.synth_per_class == 1000 else "_%dpc" % args.synth_per_class)
    out_dir = os.path.join(OUT_ROOT + ("_smoke" if args.smoke else ""), exp)
    os.makedirs(out_dir, exist_ok=True)
    tp.setup_logging(out_dir)
    args.cur_exp = exp
    log = tp.log
    log.info("#" * 100)
    log.info("EXPERIMENT %s: %s | epochs=%d seeds=%s | %s", exp, cfg["label"], args.epochs, args.seeds,
             datetime.now().isoformat(timespec="seconds"))
    log.warning("No validation split: evaluated on the official test split; patients 156/218 leak across the split.")

    rows, dev_rows = [], []
    for s in args.seeds:
        r, dr = run_seed(ctx, cfg, args, s, frames, out_dir)
        rows.append(r)
        dev_rows += dr

    summ = {"experiment": exp, "label": cfg["label"], "seeds": args.seeds, "epochs": args.epochs,
            "config": dict(cfg, gan=args.gan, synth_per_class=args.synth_per_class if args.gan else 0, seg_sec=SEG_SEC, freeze_blocks=args.freeze_blocks, batch=BATCH,
                           lr_backbone=LR_BACKBONE, lr_head=LR_HEAD, tau=TAU),
            "note": "Official ICBHI test split, no held-out val set; patients 156/218 leak across the split."}
    for k in KEYS:
        v = np.array([r[k] for r in rows])
        summ[k] = {"mean": float(v.mean()), "std": float(v.std())}
    with open(os.path.join(out_dir, "final_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    fields = ["seed", "epochs"] + KEYS + ["final_train_ce", "n_train", "n_test", "confusion", "train_time_s"]
    with open(os.path.join(out_dir, "per_seed_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(os.path.join(out_dir, "per_device_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "device", "n_test"] + KEYS)
        w.writeheader()
        for r in dev_rows:
            w.writerow(r)

    log.info("=" * 100)
    log.info("FINAL RESULT %s (%d seeds): %s", exp, len(args.seeds), cfg["label"])
    for k in KEYS:
        log.info("   %-15s %.2f +- %.2f", k, summ[k]["mean"], summ[k]["std"])

    if not args.smoke and not args.no_register:
        from register_result import register
        rel = os.path.relpath(out_dir, ROOT)
        proto = "" if args.epochs == 30 and args.seeds == SEEDS_DEFAULT else " [%d ep, %d seeds]" % (args.epochs, len(args.seeds))
        cmd = "experiment_stethoscope/sg_exp.py --exp %s" % exp.split("_")[0] + (" --gan %s" % args.gan if args.gan else "") \
            + (" --synth-per-class %d" % args.synth_per_class if args.gan and args.synth_per_class != 1000 else "")
        register("%s" % exp, FAMILY, cfg["label"] + proto, cmd,
                 rel + "/", os.path.join(rel, "final_summary.json"))
        log.info("registered %s in results_summary/all_runs.csv", exp)
    return summ


def write_local_table():
    """experiment_stethoscope/results.csv: every finished experiment in this folder, incl. per-device AS."""
    rows = []
    extra = sorted(d for d in os.listdir(OUT_ROOT) if d not in ORDER and not d.startswith("_"))
    for exp in ORDER + extra:
        p = os.path.join(OUT_ROOT, exp, "final_summary.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        r = {"id": exp, "method": d["label"], "epochs": d["epochs"], "n_seeds": len(d["seeds"])}
        for k in KEYS:
            r[k + "_mean"], r[k + "_std"] = "%.2f" % d[k]["mean"], "%.2f" % d[k]["std"]
        dp = os.path.join(OUT_ROOT, exp, "per_device_results.csv")
        if os.path.exists(dp):
            acc = {}
            for x in csv.DictReader(open(dp)):
                acc.setdefault(x["device"], []).append(float(x["AS"]))
            for dev in DEVICES:
                r["AS_" + dev] = "%.2f" % np.mean(acc[dev]) if dev in acc else ""
        rows.append(r)
    if not rows:
        return
    fields = (["id", "method", "epochs", "n_seeds"] + [k + s for k in KEYS for s in ("_mean", "_std")]
              + ["AS_" + d for d in DEVICES])
    with open(os.path.join(_P.OUT, "sg_exp_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["all"], help="S0a S0b S1 S2 S2n S2w S3 | all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--paper-protocol", action="store_true", help="50 epochs, 5 seeds (the paper's protocol)")
    ap.add_argument("--freeze-blocks", type=int, default=0, help="0 = full fine-tune")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, 1 seed; outputs to outputs_smoke/")
    ap.add_argument("--gan", default=None, help="gan_check/runs/<RUN> to add as synthetic train data, e.g. wgangp_plain")
    ap.add_argument("--synth-per-class", type=int, default=1000, help="synthetic cycles per minority class (old GAN pipelines: 1000)")
    ap.add_argument("--no-register", action="store_true", help="do not touch results_summary/all_runs.csv")
    args = ap.parse_args()
    if args.paper_protocol:
        args.epochs, args.seeds = 50, PAPER_SEEDS
    exps = ORDER if args.exp == ["all"] else args.exp
    for e in exps:
        if e not in EXPS:
            sys.exit("unknown experiment %s (choose from %s)" % (e, ", ".join(ORDER)))
    stride = 1
    if args.smoke:
        args.epochs, args.seeds, args.num_workers, stride = 1, [13], 2, 38

    # shared waveform cache: symlink the existing one (read-only use) so nothing is written outside this folder
    cache = _P.sub("cache")
    tp.CACHE_DIR = cache
    os.makedirs(OUT_ROOT, exist_ok=True)
    tp.setup_logging(os.path.join(OUT_ROOT + ("_smoke" if args.smoke else ""), "_setup"))
    recs, cycles = tp.build_dataset(smoke_stride=stride)
    tp.ensure_wave_cache(recs)
    ctx = tp.Ctx(recs, cycles)
    frames = tp.frames_for(SEG_SEC)

    for e in exps:
        run_experiment(e, ctx, args, frames)
    if not args.smoke:
        write_local_table()


if __name__ == "__main__":
    main()
