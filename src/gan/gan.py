"""
gan.py -- Step 4: conditional GAN on raw AST log-mel windows (128 mel x 512 frames, 10 ms).

Generator: BigGAN-style residual upsampling 4x16 -> 128x512, conditional BN driven by [z, cond] in every
           block, EMA copy (0.999) used for sampling.
Critic:    residual downsampling with spectral norm, hinge loss, projection term per condition attribute.
Regularisation for ~3.5k cycles: DiffAugment on real+fake (time/freq shift, cutout, gain), optional mode-seeking.

Conditions (--cond, cumulative so each run adds one attribute):
  class      crackle flag + wheeze flag (normal = 0,0; both = 1,1)  -> "both" shares what crackle/wheeze learn
  dev        + device (4)
  reg        + chest region (trachea / anterior / post_lat)
Training data: all gan_split=train cycles (normal included), sampled class-balanced and, within a class,
patient-balanced (w ~ 1/sqrt(cycles of that patient)) so a few patients do not define a class.
Eval samples: N_PER_CLASS per minority class; device / region drawn from that class's real joint
distribution in gan-train so the set is comparable to the real reference.

Usage: $PY GAN_FINETUNING/gan.py --name c1_class --cond class [--iters 30000] [--eval-every 10000]
Writes runs/<name>/{train.log, history.json, G_ema.pt, ckpt.pt, samples_<it>.npz}; eval rows -> eval/results.csv
"""
import os
import sys
import gc
import copy
import csv
import json
import shutil
import time
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gfdata as G                                  # noqa: E402

import na_paths as _P
RUNS = os.path.join(_P.OUT, "gan_runs")
Z = 128
N_PER_CLASS = 300
ATTRS = {"class": [("crackle", 2), ("wheeze", 2)], "dev": [("device", 4)], "reg": [("region", 3)],
         # HF_Lung pretraining (README E18 / D6): class flags + 5 devices (ICBHI's 4 + HF-Type-1) + data source
         "pre": [("crackle", 2), ("wheeze", 2), ("device", 5), ("source", 2)],
         # class flags + data source only (g3_t2's class-only conditioning, plus the flag to ask for ICBHI-style output)
         "cs": [("crackle", 2), ("wheeze", 2), ("source", 2)],
         # unconditional (one constant condition): base GAN + single-class specialists share every weight shape
         "one": [("one", 1)]}
COND_ORDER = ["class", "dev", "reg"]
DEVICES5 = G.DEVICES + ["HF-Type-1"]                     # ICBHI's 4 keep their indices; Litt3200 is shared
SOURCES = ["ICBHI", "HF_Lung"]


def attrs_for(cond):
    if cond in ("pre", "cs", "one"):
        return list(ATTRS[cond])
    out = []
    for c in COND_ORDER[:COND_ORDER.index(cond) + 1]:
        out += ATTRS[c]
    return out


def encode(r, attrs):
    v = {"crackle": r["crackle"], "wheeze": r["wheeze"], "device": DEVICES5.index(r["device"]),
         "region": G.REGIONS.index(r["region"]) if "region" in r else 0, "source": r.get("source", 0), "one": 0}
    return [v[a] for a, _ in attrs]


# ============================================================================ data
class Data:
    def __init__(self, attrs, seed, aug=False, hf=False, hf_frac=0.7, mu=None, sd=None, hf_match=False, only_class=0,
                 hf_clean=False):
        """ICBHI gan-train cycles (+ augmented copies) and, with hf=True, the HF_Lung_V1 cache (hfdata.py: cycles +
        event windows, source=1). Sampling: within each source, classes get equal mass and patients within a class
        are down-weighted by 1/sqrt(#cycles); with hf=True the sources are then mixed hf_frac : 1 - hf_frac (D9).
        mu / sd: pass the stage-1 values when fine-tuning so the generator keeps its scale."""
        self.rows = [dict(r, source=0) for r in G.load_meta() if r["gan_split"] == "train"
                     and (not only_class or r["label"] == only_class)]        # single-class specialist
        self.only_class = only_class
        n_real = len(self.rows)
        aug_rows = hf_rows = []
        if aug:                                                  # + augmented minority copies (gfdata.build_aug)
            aug_rows, aug_fb = G.load_aug()
            aug_rows = [dict(r, source=0) for r in aug_rows]
            self.rows += aug_rows
        if hf:
            import hfdata
            hf_rows, hf_fb = hfdata.load_hf()
            hold = hfdata.holdout_patients()                     # HF_Lung reference patients, never trained on
            hf_rows = [dict(r, source=1) for r in hf_rows if r["patient"] not in hold]
            if hf_clean:                                         # 2026-11-08: drop rhonchi-only "wheeze" rows
                ro = {int(q["hid"]) for q in csv.DictReader(open(os.path.join(G.DATA, "hf_clean.csv")))
                      if q["rhonchi_only"] == "1"}
                hf_rows = [r for r in hf_rows if r["hid"] not in ro]
            if hf_match:                                         # pilot: as many HF samples as ICBHI, same class mix
                rs = np.random.RandomState(seed)
                need = Counter(r["label"] for r in self.rows[:n_real])
                pick = []
                for k in range(4):
                    ik = [i for i, r in enumerate(hf_rows) if r["label"] == k]
                    pick += list(rs.choice(ik, min(need[k], len(ik)), replace=False))
                hf_rows = [hf_rows[i] for i in sorted(pick)]
            self.rows += hf_rows
        self.attrs = attrs
        self.rng = random.Random(seed)
        pc = Counter((r["source"], r["label"], r["patient"]) for r in self.rows)
        w = np.array([1.0 / np.sqrt(pc[(r["source"], r["label"], r["patient"])]) for r in self.rows])
        src = np.array([r["source"] for r in self.rows])
        lab = np.array([r["label"] for r in self.rows])
        for sv, mass in ((0, 1.0 - hf_frac if hf else 1.0), (1, hf_frac if hf else 0.0)):
            for k in range(4):                                   # each class gets 1/4 of its source's mass
                m = (src == sv) & (lab == k)
                if m.any():
                    w[m] = w[m] / w[m].sum() / 4 * mass
        self.w = w / w.sum()
        self.cond = np.array([encode(r, attrs) for r in self.rows], dtype=np.int64)
        fb = [G.get_fbank(r) for r in self.rows[:n_real]]
        if aug:
            fb += [np.asarray(aug_fb[r["fb_off"]:r["fb_off"] + r["fb_len"]], dtype=np.float32) for r in aug_rows]
        if hf:
            fb += [np.asarray(hf_fb[r["fb_off"]:r["fb_off"] + r["fb_len"]], dtype=np.float32) for r in hf_rows]
        if mu is None:                                           # streaming mean / std (HF_Lung adds ~17M frames)
            n = sum(f.size for f in fb)
            mu = sum(float(f.sum(dtype=np.float64)) for f in fb) / n
            sd = (sum(float(((f - mu) ** 2).sum(dtype=np.float64)) for f in fb) / n) ** 0.5
        self.mu, self.sd = float(mu), float(sd)
        self.fb = [((f - self.mu) / self.sd).astype(np.float32) for f in fb]

    def start_prefetch(self, n, threads=3, depth=8):
        """Build full-data batches in background threads (crop_pad + stacking on the CPU took ~30% of each step),
        pinned for a non-blocking copy to the GPU. Normal-only batches (T2 phase) stay synchronous."""
        import threading
        import queue
        self.pf_n, self.pf_q = n, queue.Queue(maxsize=depth)

        def produce(seed):
            g, rng = np.random.default_rng(seed), random.Random(seed)
            while True:
                idx = g.choice(len(self.rows), n, p=self.w)
                x = np.stack([G.crop_pad(self.fb[i], rng=rng)[0] for i in idx])
                xt = torch.from_numpy(x).transpose(1, 2)[:, None].contiguous().pin_memory()
                self.pf_q.put((xt, torch.from_numpy(self.cond[idx])))
        for t in range(threads):
            threading.Thread(target=produce, args=(1000 + t,), daemon=True).start()

    def batch(self, n, device, normal_only=False):
        if not normal_only and getattr(self, "pf_n", None) == n:
            x, c = self.pf_q.get()
            return x.to(device, non_blocking=True), c.to(device, non_blocking=True)
        if normal_only:                                          # T2 pretraining phase: normal cycles only
            if not hasattr(self, "w_normal"):
                wn = np.where(np.array([r["label"] == 0 for r in self.rows]), self.w, 0.0)
                self.w_normal = wn / wn.sum()
            idx = np.random.choice(len(self.rows), n, p=self.w_normal)
        else:
            idx = np.random.choice(len(self.rows), n, p=self.w)
        x = np.stack([G.crop_pad(self.fb[i], rng=self.rng)[0] for i in idx])        # [n,512,128]
        x = torch.from_numpy(x).transpose(1, 2)[:, None]                            # [n,1,128(F),512(T)]
        return x.to(device), torch.from_numpy(self.cond[idx]).to(device)

    def eval_conditions(self, rng):
        """N_PER_CLASS conditions per minority class, attributes drawn from that class's real rows."""
        conds, labels = [], []
        if self.only_class:                                      # specialist: all samples are its class
            i = rng.choice(len(self.rows), 3 * N_PER_CLASS)
            return self.cond[i], np.full(len(i), self.only_class)
        for k in (1, 2, 3):                                      # ICBHI-like conditions only (source 0)
            rows = [i for i, r in enumerate(self.rows) if r["label"] == k and r["source"] == 0]
            for i in rng.choice(rows, N_PER_CLASS):
                conds.append(self.cond[i]); labels.append(k)
        return np.array(conds), np.array(labels)


# ============================================================================ models
class CondEmbed(nn.Module):
    """Sum of one embedding per attribute -> [B, dim]."""
    def __init__(self, attrs, dim, sn=False):
        super().__init__()
        f = SN if sn else (lambda m: m)
        self.e = nn.ModuleList([f(nn.Embedding(n, dim)) for _, n in attrs])

    def forward(self, c):
        return sum(e(c[:, i]) for i, e in enumerate(self.e))


class CBN(nn.Module):
    def __init__(self, ch, cdim):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch, affine=False)
        self.g, self.b = nn.Linear(cdim, ch), nn.Linear(cdim, ch)
        nn.init.zeros_(self.g.weight); nn.init.ones_(self.g.bias)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x, c):
        return self.bn(x) * self.g(c)[:, :, None, None] + self.b(c)[:, :, None, None]


def upsample(x, aa=False):
    """x2 upsampling. aa=True: bilinear + [1,2,1] blur (anti-aliased), against the structured high-frequency
    artefacts of nearest-neighbour upsampling found in diagnosis B (fakes had up to 1.63x real HF power)."""
    if not aa:
        return F.interpolate(x, scale_factor=2)
    x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    k = torch.tensor([1.0, 2.0, 1.0], device=x.device, dtype=x.dtype)
    k = (k[:, None] * k[None, :]) / 16.0
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k.expand(x.size(1), 1, 3, 3), groups=x.size(1))


class NoiseInj(nn.Module):
    """Per-pixel noise with a learned per-channel scale (StyleGAN): the natural local grain fakes lacked (diag. B)."""
    def __init__(self, ch):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        return x + self.w * torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)


class GBlock(nn.Module):
    def __init__(self, i, o, cdim, aa=False, noise=False):
        super().__init__()
        self.aa = aa
        self.n1, self.n2 = CBN(i, cdim), CBN(o, cdim)
        self.c1, self.c2 = nn.Conv2d(i, o, 3, 1, 1), nn.Conv2d(o, o, 3, 1, 1)
        self.sk = nn.Conv2d(i, o, 1)
        self.z1, self.z2 = (NoiseInj(o), NoiseInj(o)) if noise else (nn.Identity(), nn.Identity())

    def forward(self, x, c):
        h = upsample(F.relu(self.n1(x, c)), self.aa)
        h = self.z2(self.c2(F.relu(self.n2(self.z1(self.c1(h)), c))))
        return h + self.sk(upsample(x, self.aa))


class Generator(nn.Module):
    CH = [256, 256, 128, 128, 64, 32]                    # 4x16 -> 8x32 -> ... -> 128x512

    def __init__(self, attrs, edim=128, aa=False, noise=False, bound=None):
        super().__init__()
        self.bound = bound                                 # soft output limit B * tanh(x / B) (standardised units)
        self.emb = CondEmbed(attrs, edim)
        cdim = Z + edim
        self.fc = nn.Linear(cdim, self.CH[0] * 4 * 16)
        self.blocks = nn.ModuleList([GBlock(self.CH[i], self.CH[i + 1], cdim, aa, noise) for i in range(5)])
        self.out = nn.Sequential(nn.BatchNorm2d(self.CH[-1]), nn.ReLU(), nn.Conv2d(self.CH[-1], 1, 3, 1, 1))

    def forward(self, z, c, lam=1.0):
        """lam in [0,1] scales the condition (transitional training, Shahbazi et al. 2022): 0 = unconditional."""
        cz = torch.cat([z, self.emb(c) * lam], 1)
        h = self.fc(cz).view(-1, self.CH[0], 4, 16)
        for b in self.blocks:
            h = b(h, cz)
        x = self.out(h)                                    # [B,1,128,512], standardised log-mel
        return self.bound * torch.tanh(x / self.bound) if self.bound else x


class DualGen(nn.Module):
    """--dual-gen (user idea, 2026-11-20): two CNN branches split the work and are combined by learned multiplicative
    weights.
      GLOBAL branch (what is the same over the whole window): a mean spectrum a(f) over the 128 mel bins, a temporal
        energy envelope b(t) (breathing loudness, 64 knots -> 512 frames) and a smooth low-resolution map g(f, t)
        (32 x 128, bilinear up): E = a(f) + b(t) + g(f, t).
      LOCAL branch (fine detail: texture, crackle clicks, wheeze lines): the c1_class generator trunk at full
        resolution -> detail map D and its last feature maps h.
      FUSION: a gate M = sigmoid(conv([h, E])) in [0,1] per pixel decides where and how much local detail is added:
        x = E + M * D.
    Same condition (z + class embedding) for both branches; same critic and losses as the plain GAN."""
    def __init__(self, attrs, edim=128, aa=False, noise=False, bound=None):
        super().__init__()
        self.bound = bound
        self.local = Generator(attrs, edim, aa, noise, bound=None)
        cdim = Z + edim
        mlp = lambda o: nn.Sequential(nn.Linear(cdim, 256), nn.ReLU(), nn.Linear(256, o))
        self.g_spec, self.g_time = mlp(128), mlp(64)
        self.g_fc = nn.Linear(cdim, 64 * 8 * 32)
        self.g_blocks = nn.ModuleList([GBlock(64, 64, cdim, aa, False), GBlock(64, 32, cdim, aa, False)])   # 8x32 -> 32x128
        self.g_out = nn.Conv2d(32, 1, 3, 1, 1)
        ch = Generator.CH[-1]
        self.gate = nn.Sequential(nn.Conv2d(ch + 1, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def forward(self, z, c, lam=1.0):
        L = self.local
        cz = torch.cat([z, L.emb(c) * lam], 1)
        B = len(z)
        a = self.g_spec(cz)[:, None, :, None]                                        # [B,1,128,1]
        b = F.interpolate(self.g_time(cz)[:, None], size=512, mode="linear", align_corners=False)[:, :, None, :]
        g = self.g_fc(cz).view(B, 64, 8, 32)
        for blk in self.g_blocks:
            g = blk(g, cz)
        g = F.interpolate(self.g_out(g), size=(128, 512), mode="bilinear", align_corners=False)
        E = a + b + g                                                                # global component [B,1,128,512]
        h = L.fc(cz).view(-1, L.CH[0], 4, 16)
        for blk in L.blocks:
            h = blk(h, cz)
        D = L.out(h)                                                                 # local detail
        M = torch.sigmoid(self.gate(torch.cat([h, E], 1)))
        self.last_gate, self.gate_mean = M.detach(), M.mean()
        self.last_E, self.last_D = E.detach(), D.detach()
        x = E + M * D
        return self.bound * torch.tanh(x / self.bound) if self.bound else x


class DBlock(nn.Module):
    def __init__(self, i, o, pre=True):
        super().__init__()
        self.pre = pre
        self.c1, self.c2 = SN(nn.Conv2d(i, o, 3, 1, 1)), SN(nn.Conv2d(o, o, 3, 1, 1))
        self.sk = SN(nn.Conv2d(i, o, 1))

    def forward(self, x):
        h = F.relu(x) if self.pre else x
        h = F.avg_pool2d(self.c2(F.relu(self.c1(h))), 2)
        return h + F.avg_pool2d(self.sk(x), 2)


class MinibatchStd(nn.Module):
    """Appends the std across a group of 4 samples as one feature map: lets the critic see 'all fakes alike'."""
    def forward(self, x, group=4):
        b, c, h, w = x.shape
        g = group if b % group == 0 else 1
        s = x.view(g, -1, c, h, w).float().std(0, unbiased=False).mean(dim=(1, 2, 3)).to(x.dtype)
        return torch.cat([x, s.repeat(g).view(b, 1, 1, 1).expand(b, 1, h, w)], 1)


class Critic(nn.Module):
    CH = [1, 16, 64, 128, 256, 256]                     # light first block: it runs at full 128x512

    def __init__(self, attrs, mbstd=False, aux=False):
        super().__init__()
        D = self.CH[-1]
        self.blocks = nn.Sequential(*[DBlock(self.CH[i], self.CH[i + 1], pre=i > 0) for i in range(5)])
        self.mbstd = nn.Sequential(nn.ReLU(), MinibatchStd(), SN(nn.Conv2d(D + 1, D, 3, 1, 1))) if mbstd else None
        self.lin = SN(nn.Linear(D, 1))
        self.proj = CondEmbed(attrs, D, sn=True)
        # auxiliary classifier (ACGAN-style), one head per condition attribute
        self.aux = nn.ModuleList([SN(nn.Linear(D, n)) for _, n in attrs]) if aux else None

    def forward(self, x, c, with_aux=False, lam=1.0):
        h = self.blocks(x)
        if self.mbstd is not None:
            h = self.mbstd(h)
        h = F.relu(h).sum(dim=(2, 3))
        score = self.lin(h).squeeze(1) + lam * (self.proj(c) * h).sum(1)
        if not with_aux:
            return score
        return score, [head(h) for head in self.aux]


class ProjectedD(nn.Module):
    """Projected-GAN discriminator (Sauer et al. 2021): frozen AudioSet AST (never trained on ICBHI) gives
    token maps at depths 4 / 8 / 12; fixed random 1x1 projections (not trained) feed small trainable SN heads,
    each with its own projection-conditioning term; the logits are averaged over depths."""
    LAYERS = (4, 8, 12)

    def __init__(self, attrs, mu, sd, pdim=64):
        super().__init__()
        import train_optimal_pipeline as tp
        self.ast = tp.build_model(G.WIN).audio_spectrogram_transformer
        self.mu, self.sd = mu, sd
        D = self.ast.config.hidden_size
        self.rproj = nn.ModuleList([nn.Conv2d(D, pdim, 1) for _ in self.LAYERS])
        for p in list(self.ast.parameters()) + list(self.rproj.parameters()):
            p.requires_grad = False
        self.heads = nn.ModuleList([nn.Sequential(SN(nn.Conv2d(pdim, pdim, 3, 1, 1)), nn.LeakyReLU(0.2),
                                                  SN(nn.Conv2d(pdim, pdim, 3, 2, 1)), nn.LeakyReLU(0.2))
                                    for _ in self.LAYERS])
        self.lin = nn.ModuleList([SN(nn.Linear(pdim, 1)) for _ in self.LAYERS])
        self.emb = nn.ModuleList([CondEmbed(attrs, pdim, sn=True) for _ in self.LAYERS])
        self.f_dim = (G.N_MELS - 16) // 10 + 1                                     # 12 frequency patches

    def train(self, mode=True):
        super().train(mode)
        self.ast.eval()                                                            # frozen: no dropout
        return self

    def forward(self, x, c, with_aux=False, lam=1.0):
        a = G.ast_normalize(x[:, 0].transpose(1, 2) * self.sd + self.mu)          # -> AST input [B,T,F]
        with torch.set_grad_enabled(torch.is_grad_enabled() and x.requires_grad):  # G step only
            hs = self.ast(input_values=a, output_hidden_states=True).hidden_states
        out = 0
        for k, layer in enumerate(self.LAYERS):
            t = hs[layer][:, 2:]                                                   # drop CLS + dist tokens
            g = t.reshape(t.size(0), self.f_dim, -1, t.size(2)).permute(0, 3, 1, 2)
            h = self.heads[k](self.rproj[k](g)).mean(dim=(2, 3))                  # mean: stable logit scale
            out = out + self.lin[k](h).squeeze(1) + lam * (self.emb[k](c) * h).sum(1)
        return out / len(self.LAYERS)


class DualD(nn.Module):
    """Projected AST critic + an ordinary conv critic side by side (in the spirit of Kumari et al. 2022). Diagnosis C:
    the projected critic alone was fooled (96% of fakes scored real) while the conv critic is not. Returns both
    scores as [B,2]; the hinge / LeCam terms are then averaged over the two critics."""
    def __init__(self, attrs, mu, sd, pdim=64, mbstd=False):
        super().__init__()
        self.proj = ProjectedD(attrs, mu, sd, pdim)
        self.conv = Critic(attrs, mbstd=mbstd)

    def forward(self, x, c, with_aux=False, lam=1.0):
        return torch.stack([self.proj(x, c, lam=lam), self.conv(x, c, lam=lam)], 1)


def aux_loss(logits, c):
    return sum(F.cross_entropy(lg.float(), c[:, i]) for i, lg in enumerate(logits))


# ============================================================================ DiffAugment (spectrogram-safe)
def diff_augment(x):
    B, _, Fq, T = x.shape
    floor = x.amin(dim=(1, 2, 3), keepdim=True)
    x = x + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4                 # gain
    tx = torch.randint(-T // 8, T // 8 + 1, (B,), device=x.device)                # time shift
    fy = torch.randint(-4, 5, (B,), device=x.device)                               # freq shift (mel bins)
    ti = (torch.arange(T, device=x.device)[None] - tx[:, None])
    fi = (torch.arange(Fq, device=x.device)[None] - fy[:, None])
    tv, fv = (ti >= 0) & (ti < T), (fi >= 0) & (fi < Fq)
    x = torch.gather(x, 3, ti.clamp(0, T - 1)[:, None, None].expand(B, 1, Fq, T))
    x = torch.gather(x, 2, fi.clamp(0, Fq - 1)[:, None, :, None].expand(B, 1, Fq, T))
    keep = (tv[:, None, None] & fv[:, None, :, None])
    x = torch.where(keep, x, floor)
    t0, f0 = torch.randint(0, T, (B,), device=x.device), torch.randint(0, Fq, (B,), device=x.device)
    tm = (torch.arange(T, device=x.device)[None] - t0[:, None]).abs() < 32          # cutout 64 frames
    fm = (torch.arange(Fq, device=x.device)[None] - f0[:, None]).abs() < 8          # x 16 mel
    cut = tm[:, None, None] & fm[:, None, :, None]
    return torch.where(cut, floor, x)


# ============================================================================ training
@torch.no_grad()
def generate(Gm, conds, data, device, bs=64, seed=0, lam=1.0, trunc=1.0):
    """lam = the condition strength the generator was trained with at this point (transitional schedule): generating
    with lam=1 during the unconditional phase would feed it condition embeddings it never learned to use."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, len(conds), bs):
        c = torch.as_tensor(conds[i:i + bs], device=device)
        x = Gm(trunc * torch.randn(len(c), Z, device=device, generator=g), c, lam).float()   # trunc < 1: truncation
        out.append((x[:, 0].transpose(1, 2) * data.sd + data.mu).cpu().numpy())    # raw log-mel [n,512,128]
    return np.concatenate(out).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cond", default="class", choices=COND_ORDER + ["pre", "cs", "one"])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-dis", type=int, default=1)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=4e-4)
    ap.add_argument("--ms", type=float, default=0.0, help="mode-seeking weight (0 = off)")
    ap.add_argument("--mbstd", action="store_true", help="minibatch-stddev layer in the critic")
    ap.add_argument("--ms-pool", type=int, default=1,
                    help="measure mode-seeking on k x k average-pooled outputs (e.g. 4): with --noise-inj the per-pixel "
                         "injected noise alone makes two samples differ, which satisfies the plain term without diversity")
    ap.add_argument("--ac", type=float, default=0.0,
                    help="auxiliary-classifier weight (0 = off); D learns it on real only, G on fakes")
    ap.add_argument("--aug-data", action="store_true",
                    help="add the augmented minority copies (gap removal / time-stretch / speed) to the training data")
    ap.add_argument("--proj-d", action="store_true", help="projected discriminator on frozen AudioSet-AST features")
    ap.add_argument("--dual", action="store_true", help="projected critic + ordinary conv critic (implies --proj-d)")
    ap.add_argument("--pdim", type=int, default=64, help="channels of the projected critic heads (capacity)")
    ap.add_argument("--aa-up", action="store_true", help="anti-aliased (bilinear + blur) upsampling in G")
    ap.add_argument("--noise-inj", action="store_true", help="per-pixel noise injection in G")
    ap.add_argument("--trans-start", type=int, default=0, help="T1: unconditional until this iteration ...")
    ap.add_argument("--trans-end", type=int, default=0, help="... then the condition is phased in linearly until here")
    ap.add_argument("--pretrain-normal", type=int, default=0,
                    help="T2: the first N iterations use normal cycles only (and no condition); needs --trans-start >= N")
    ap.add_argument("--lecam", type=float, default=0.0, help="LeCam regularisation weight (Tseng et al. 2021), 0 = off")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast (shared GPU: bf16 is ~1.5x faster)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf", action="store_true", help="stage 1: add the HF_Lung_V1 cache (hfdata.py) to the data")
    ap.add_argument("--hf-frac", type=float, default=0.7, help="share of HF_Lung samples per batch with --hf (D9)")
    ap.add_argument("--hf-match", action="store_true",
                    help="pilot: use only as many HF samples as ICBHI gan-train has, with the same class mix")
    ap.add_argument("--only-class", type=int, default=0, choices=[0, 1, 2, 3],
                    help="single-class specialist: train (and evaluate) on this class only (1 crackle, 2 wheeze, 3 both)")
    ap.add_argument("--bound", action="store_true",
                    help="soft-bound the generator output to the real data range (fixes exploded samples, E24)")
    ap.add_argument("--init-it", type=int, default=0,
                    help="stage 2 from runs/<init-from>/ckpt_<N>.pt (saved with --keep-ckpts) instead of the last ckpt.pt")
    ap.add_argument("--keep-ckpts", action="store_true", help="also keep ckpt_<it>.pt at every eval (for --init-it)")
    ap.add_argument("--init-from", default=None,
                    help="stage 2: start from runs/<NAME>/ckpt.pt (G, G_ema, D) and its mu / sd; optimisers fresh")
    ap.add_argument("--hf-clean", action="store_true", help="with --hf: drop rhonchi-only 'wheeze' rows (hf_clean.csv)")
    ap.add_argument("--teacher", default=None,
                    help="HF_Lung event detector (HF_GAN/eventdet.py ckpt): class-presence loss on the fakes (2026-11-08)")
    ap.add_argument("--dual-gen", action="store_true", help="global + local generator branches, gated fusion (DualGen)")
    ap.add_argument("--gate-reg", type=float, default=0.0,
                    help="DualGen anti-collapse: weight of relu(0.15 - mean gate) + relu(mean gate - 0.85) (inactive in between)")
    ap.add_argument("--spec-reg", type=float, default=0.0,
                    help="Durall et al. 2020 spectral loss weight: BCE of the fakes' azimuthally integrated 1D spectrum vs real")
    ap.add_argument("--teacher-w", type=float, default=1.0, help="weight of the --teacher loss in the G step")
    ap.add_argument("--freeze-d", type=int, default=0, help="stage 2 FreezeD: freeze the first N critic blocks")
    ap.add_argument("--resume", action="store_true",
                    help="continue runs/<name>/ckpt.pt (G, G_ema, D, iteration; optimisers + LeCam anchors if saved)")
    a = ap.parse_args()
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.no_amp)
    logf = open(os.path.join(out, "train.log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    attrs = attrs_for(a.cond)
    init_cfg = json.load(open(os.path.join(RUNS, a.init_from, "config.json"))) if a.init_from else {}
    data = Data(attrs, a.seed, aug=a.aug_data, hf=a.hf, hf_frac=a.hf_frac, mu=init_cfg.get("mu"), sd=init_cfg.get("sd"),
                hf_match=a.hf_match, only_class=a.only_class, hf_clean=a.hf_clean)
    bound = init_cfg.get("bound") or (1.1 * max(float(np.abs(f).max()) for f in data.fb) if a.bound else None)
    data.start_prefetch(a.batch)
    assert a.pretrain_normal <= a.trans_start or a.pretrain_normal == 0, "T2 needs --trans-start >= --pretrain-normal"
    if a.dual:
        assert a.ac == 0, "--dual does not support --ac"
        D = DualD(attrs, data.mu, data.sd, a.pdim, a.mbstd).to(dev)
    elif a.proj_d:
        assert a.ac == 0 and not a.mbstd, "--proj-d supports neither --ac nor --mbstd"
        D = ProjectedD(attrs, data.mu, data.sd, a.pdim).to(dev)
    else:
        D = Critic(attrs, mbstd=a.mbstd, aux=a.ac > 0).to(dev)
    Gm = (DualGen if a.dual_gen else Generator)(attrs, aa=a.aa_up, noise=a.noise_inj, bound=bound).to(dev)
    G_ema = copy.deepcopy(Gm).eval()
    oG = torch.optim.Adam(Gm.parameters(), a.lr_g, betas=(0.0, 0.999))
    if a.init_from:                                       # stage 2: continue from the stage-1 GAN
        ck = torch.load(os.path.join(RUNS, a.init_from, "ckpt_%d.pt" % a.init_it if a.init_it else "ckpt.pt"),
                        map_location="cpu")
        Gm.load_state_dict(ck["G"]); G_ema.load_state_dict(ck["G_ema"])
        missing, unexpected = D.load_state_dict(ck["D"], strict=False)
        assert not unexpected and all("ast." in k for k in missing), (missing[:5], unexpected[:5])
    if a.freeze_d:                                        # FreezeD (Mo et al. 2020): keep the low-level critic features
        conv = D.conv if isinstance(D, DualD) else D
        for b in conv.blocks[:a.freeze_d]:
            for p in b.parameters():
                p.requires_grad = False
    oD = torch.optim.Adam([p for p in D.parameters() if p.requires_grad], a.lr_d, betas=(0.0, 0.999))
    lc = {"r": 0.0, "f": 0.0}                                     # LeCam anchors (EMA of D on real / fake)
    log("%s | cond=%s attrs=%s | %d train cycles | G %.1fM D %.1fM params | args %s"
        % (a.name, a.cond, attrs, len(data.rows), sum(p.numel() for p in Gm.parameters()) / 1e6,
           sum(p.numel() for p in D.parameters() if p.requires_grad) / 1e6, vars(a)))
    json.dump(dict(vars(a), attrs=attrs, mu=data.mu, sd=data.sd, bound=bound), open(os.path.join(out, "config.json"), "w"),
              indent=1)

    teacher, attr_names = None, [n for n, _ in attrs]
    az, spec_ref = None, {"r": None}
    if a.spec_reg > 0:                                    # 2026-11-20: Durall's spectral regularisation
        import azimuth as AZ
        az = AZ.Azimuth().to(dev)
        log("spectral loss (azimuthal integration, Durall 2020), weight %g" % a.spec_reg)
    if a.teacher:
        assert a.ms == 0, "--teacher reuses the ms log column"
        import eventdet
        teacher = eventdet.load(a.teacher, dev)
        log("teacher %s, weight %g" % (a.teacher, a.teacher_w))
    import evaluate as E
    ref, ref_hf, hist, t0 = None, None, [], time.time()
    start_it = 1
    if a.resume:                                          # e.g. a run paused at a 5k checkpoint to free GPU memory
        ck = torch.load(os.path.join(out, "ckpt.pt"), map_location="cpu")
        Gm.load_state_dict(ck["G"]); G_ema.load_state_dict(ck["G_ema"])
        missing, unexpected = D.load_state_dict(ck["D"], strict=False)
        assert not unexpected and all("ast." in k for k in missing), (missing[:5], unexpected[:5])
        if "oG" in ck:
            oG.load_state_dict(ck["oG"]); oD.load_state_dict(ck["oD"]); lc.update(ck["lc"])
        start_it = ck["it"] + 1
        hp = os.path.join(out, "history.json")
        hist = json.load(open(hp)) if os.path.exists(hp) else []
        log("RESUMED at iteration %d (optimiser state %s)" % (start_it, "restored" if "oG" in ck else "re-initialised"))
    econd, elab = data.eval_conditions(np.random.RandomState(123))
    def cond_strength(it):
        """Transitional conditioning: 0 before trans_start, linear to 1 at trans_end (always 1 if not set)."""
        if a.trans_end <= 0 or it >= a.trans_end:
            return 1.0
        return max(0.0, (it - a.trans_start) / max(1, a.trans_end - a.trans_start))

    def train_step(it):
        lam = cond_strength(it)
        normal_only = it <= a.pretrain_normal
        for _ in range(a.n_dis):
            xr, cr = data.batch(a.batch, dev, normal_only)
            with torch.no_grad(), amp:
                xf = Gm(torch.randn(a.batch, Z, device=dev), cr, lam)
            with amp:
                if a.ac > 0:
                    sr, lr_ = D(diff_augment(xr), cr, with_aux=True, lam=lam)
                    d_aux = lam * aux_loss(lr_, cr)
                else:
                    sr, d_aux = D(diff_augment(xr), cr, lam=lam), torch.zeros((), device=dev)
                sr, sf = sr.float(), D(diff_augment(xf.float()), cr, lam=lam).float()
                d_loss = F.relu(1 - sr).mean() + F.relu(1 + sf).mean() + a.ac * d_aux
                if a.lecam > 0:
                    d_loss = d_loss + a.lecam * (F.relu(sr - lc["f"]).pow(2).mean() + F.relu(lc["r"] - sf).pow(2).mean())
            oD.zero_grad(set_to_none=True); d_loss.backward(); oD.step()
            if az is not None:                           # running mean of the real normalised 1D spectrum
                with torch.no_grad():
                    r_ = AZ.minmax(az(xr.float())).mean(0, keepdim=True)
                    spec_ref["r"] = r_ if spec_ref["r"] is None else 0.99 * spec_ref["r"] + 0.01 * r_
            lc["r"] = 0.99 * lc["r"] + 0.01 * sr.mean().item()
            lc["f"] = 0.99 * lc["f"] + 0.01 * sf.mean().item()
        _, c = data.batch(a.batch, dev, normal_only)
        z1 = torch.randn(a.batch, Z, device=dev)
        with amp:
            f1 = Gm(z1, c, lam).float()
            if a.ac > 0:
                sf, lf = D(diff_augment(f1), c, with_aux=True, lam=lam)
                g_loss = -sf.float().mean() + a.ac * lam * aux_loss(lf, c)
            else:
                g_loss = -D(diff_augment(f1), c, lam=lam).float().mean()
            ms = torch.zeros((), device=dev)
            if a.gate_reg > 0:                           # DualGen: keep the gate from collapsing to all-0 / all-1
                gm = Gm.gate_mean.float()
                g_loss = g_loss + a.gate_reg * (F.relu(0.15 - gm) + F.relu(gm - 0.85))
            if az is not None:
                with torch.autocast("cuda", enabled=False):
                    s_loss = AZ.spectral_loss(az, f1.float(), spec_ref["r"])
                g_loss = g_loss + a.spec_reg * s_loss
                ms = s_loss.detach()                     # logged in the 'ms' column (no mode seeking in these runs)
            if teacher is not None:                      # HF event detector: crackle fakes must contain crackles, ...
                with torch.autocast("cuda", enabled=False):
                    t_loss = teacher.class_loss(f1[:, 0].transpose(1, 2) * data.sd + data.mu,
                                                c[:, attr_names.index("crackle")], c[:, attr_names.index("wheeze")])
                g_loss = g_loss + a.teacher_w * t_loss
                ms = t_loss.detach()                     # logged in the 'ms' column of history (no mode seeking here)
            if a.ms > 0:
                z2 = torch.randn(a.batch, Z, device=dev)
                f2 = Gm(z2, c, lam).float()
                p1, p2 = (F.avg_pool2d(f1, a.ms_pool), F.avg_pool2d(f2, a.ms_pool)) if a.ms_pool > 1 else (f1, f2)
                ms = 1.0 / ((p1 - p2).abs().mean((1, 2, 3)) / (z1 - z2).abs().mean(1)).mean().clamp_min(1e-5)
                g_loss = g_loss + a.ms * ms
        oG.zero_grad(set_to_none=True); g_loss.backward(); oG.step()
        return d_loss, g_loss, ms

    for it in range(start_it, a.iters + 1):
        for attempt in range(1, 61):                     # shared GPU: wait out other users' memory peaks
            try:
                d_loss, g_loss, ms = train_step(it)
                break
            except torch.OutOfMemoryError:
                oD.zero_grad(set_to_none=True); oG.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log("OOM at it %d (try %d): waiting 60 s" % (it, attempt))
                time.sleep(60)
        else:
            raise RuntimeError("still out of GPU memory after 60 tries at it %d" % it)
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), Gm.parameters()):
                pe.lerp_(p, 1 - 0.999)
            for be, b in zip(G_ema.buffers(), Gm.buffers()):
                be.copy_(b)
        if it % 500 == 0:
            rec = dict(it=it, d=round(d_loss.item(), 4), g=round(g_loss.item(), 4), ms=round(ms.item(), 4),
                       min=round((time.time() - t0) / 60, 1))
            if hasattr(Gm, "gate_mean"):
                rec["gate"] = round(float(Gm.gate_mean), 3)
            hist.append(rec)
            log(json.dumps(rec))
        if it % a.eval_every == 0 or it == a.iters:
            torch.save(G_ema.state_dict(), os.path.join(out, "G_ema_%d.pt" % it))
            torch.save({"G": Gm.state_dict(), "G_ema": G_ema.state_dict(), "it": it,
                        "D": {k: v for k, v in D.state_dict().items() if "ast." not in k},
                        "oG": oG.state_dict(), "oD": oD.state_dict(), "lc": dict(lc)},
                       os.path.join(out, "ckpt.pt"))
            if a.keep_ckpts:
                shutil.copyfile(os.path.join(out, "ckpt.pt"), os.path.join(out, "ckpt_%d.pt" % it))
            x = generate(G_ema, econd, data, dev, lam=cond_strength(it))
            np.savez(os.path.join(out, "samples_%d.npz" % it), x=x, y=elab, cond=econd)
            try:                                         # shared GPU: an OOM here must not kill training
                if os.environ.get("NA_GAN_EVAL", "0") != "1":   # release: judge/FD suite is opt-in
                    raise StopIteration
                if ref is None:
                    ref = E.build_reference()
                if a.only_class:                          # specialist: metrics against its own class only
                    r = E.evaluate_class("%s@%d" % (a.name, it), x, a.only_class, ref)
                    hist.append(dict(it=it, eval={k: round(float(r[k]), 4) for k in E.CLASS_COLS}))
                else:
                    r = E.evaluate_set("%s@%d" % (a.name, it), x, elab, ref, "gan.py cond=%s" % a.cond)
                    hist.append(dict(it=it, eval={k: round(float(r[k]), 4) for k in E.COLS}))
                log("EVAL " + json.dumps(hist[-1]))
                if a.hf:                                  # HF_Lung copy check + HF-style samples vs HF hold-out
                    if ref_hf is None:
                        ref_hf = E.build_hf_reference()
                    hcond = np.array([encode(dict(rr, source=1), attrs) for rr in ref_hf["gen_rows"]], dtype=np.int64)
                    xh = generate(G_ema, hcond, data, dev, lam=cond_strength(it))
                    np.savez(os.path.join(out, "samples_hf_%d.npz" % it), x=xh, y=ref_hf["gen_y"], cond=hcond)
                    rh = E.evaluate_hf("%s@%d" % (a.name, it), x, xh, ref_hf["gen_y"], ref_hf["gen_dev"], ref_hf)
                    hist[-1]["eval_hf"] = {k: round(float(rh[k]), 4) for k in E.HF_COLS}
                    log("EVAL_HF " + json.dumps(hist[-1]["eval_hf"]))
            except StopIteration:
                pass
            except torch.OutOfMemoryError:
                log("EVAL SKIPPED (OOM) at %d -- samples saved, run evaluate.py --samples later" % it)
            finally:                                     # unload judge + AudioSet AST so training has its memory back
                E.J._J, E._AS = None, None
                gc.collect()
                torch.cuda.empty_cache()
            json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
    log("DONE %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
