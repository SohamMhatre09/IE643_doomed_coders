"""
gan_v2.py -- improved class-conditional GAN for crackle / wheeze / both log-mel spectrograms.

Same data and same spectrogram format (64 mel x 128 frames, [-1, 1]) as gan_pipeline.py,
so the output is a drop-in replacement for outputs_optimal_pipeline/gan_pipeline/synthetic.

What changed vs the old DCGAN (and why):
  1. Spectral norm on every D layer + hinge loss   (SNGAN / Miyato 2018) -> stable D, no BCE saturation
  2. Projection discriminator for the label        (Miyato & Koyama 2018) -> real class conditioning
     instead of concatenating a 64x128 label plane.
  3. DiffAugment on real AND fake before D         (Zhao et al. 2020) -> stops D memorising a small
     (1550-sample) training set, the main cause of collapse in low-data GANs. Spectrogram-safe ops only:
     time shift, small freq shift, time/freq masks, gain offset.
  4. Minibatch-stddev feature in D                 (Karras 2018) -> D can see "all fakes look alike".
  5. Mode-seeking regulariser on G                 (Mao et al. 2019, MSGAN) -> different z => different x.
  6. Upsample+conv instead of ConvTranspose        -> no checkerboard artefacts.
  7. TTUR (D lr 4x G lr), Adam betas (0, 0.999), EMA copy of G used for sampling.

Usage:
  PYTHONPATH=. python gan_check/gan_v2.py [--iters 6000] [--synth-per-class 1000]
"""
import os, sys, copy, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
N_CLASSES, LATENT, H, W = 3, 128, 64, 128
NAMES = ["crackle", "wheeze", "both"]
import na_paths as _P
REAL_CACHE = os.path.join(_P.sub("wgangp_data"), "real_specs.npz")
log = logging.getLogger("gan_v2")


# ----------------------------------------------------------------------------- data
def load_real_specs():
    """Real GAN-training spectrograms (steps 1-5 of gan_pipeline.py), cached to npz."""
    if os.path.exists(REAL_CACHE):
        d = np.load(REAL_CACHE)
        return d["x"], d["y"]
    import gan_pipeline as GP
    B = GP.B
    dev = B.get_device()
    recs, cycles = B.build_dataset(smoke_stride=1)
    B.ensure_wave_cache(recs)
    frames = B.frames_for(B.WINDOW_SEC)
    models = GP.load_seed_models(frames, B.SEEDS, dev)
    fe = B.Ctx(recs, cycles).feature_extractor(frames)
    wins = GP.balance_devices(GP.select_windows(recs, cycles, models, fe, dev))
    x = np.stack([GP.log_mel_spectrogram(w["audio"]) for w in wins]).astype(np.float32)
    y = np.array([GP.SpecDataset.LABEL_MAP[w["label"]] for w in wins], dtype=np.int64)
    np.savez(REAL_CACHE, x=x, y=y)
    return x, y


HOP = 500                       # 31 ms at 16 kHz -> 8 s = 256 frames, 16 s = 512 frames


def repeat_pad(x, n, xfade=0):
    """Tile a cycle until it fills n samples (ICBHI standard), centre-crop if it is longer.
    xfade > 0: linear crossfade of `xfade` samples at every join, so the tiling leaves no
    click / vertical seam line in the spectrogram."""
    if len(x) >= n:
        s = (len(x) - n) // 2
        return x[s:s + n]
    L = min(xfade, len(x) // 4)
    if L <= 0:
        return np.tile(x, int(np.ceil(n / len(x))))[:n]
    ramp = np.linspace(0, 1, L, dtype=np.float32)
    out = x.astype(np.float32).copy()
    while len(out) < n:
        out = np.concatenate([out[:-L], out[-L:] * (1 - ramp) + x[:L] * ramp, x[L:]])
    return out[:n]


def load_real_specs_window(sec, xfade_ms=0):
    """Whole minority train cycles, repeat-padded to `sec` seconds, device-balanced (step 5).
    No multi-crop scoring needed: >99% of cycles already fit in an 8 s window."""
    cache = os.path.join(_P.sub("wgangp_data"), f"real_specs_{sec}s{'_xf%d' % xfade_ms if xfade_ms else ''}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    import torchaudio
    import gan_pipeline as GP
    B = GP.B
    recs, cycles = B.build_dataset(smoke_stride=1)
    B.ensure_wave_cache(recs)
    n, t_frames = sec * B.SR, sec * B.SR // HOP
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=B.SR, n_fft=1024, hop_length=HOP, n_mels=H, power=2.0)
    wins = []
    for c in cycles:
        if c["label"] in GP.MINORITY_LABELS and c["split"] == "train":
            wins.append(dict(label=c["label"], equip=recs[c["ridx"]]["equip"],
                             audio=repeat_pad(GP.load_cycle_wave(c, recs), n, xfade_ms * B.SR // 1000)))
    wins = GP.balance_devices(wins)
    xs = []
    for w in wins:
        m = torch.log(mel(torch.from_numpy(w["audio"].copy())).clamp_min(1e-10))[:, :t_frames]
        m = ((m - m.mean()) / (m.std() + 1e-6)).clamp(-4, 4) / 4.0
        xs.append(m.numpy())
    x = np.stack(xs).astype(np.float32)
    y = np.array([GP.SpecDataset.LABEL_MAP[w["label"]] for w in wins], dtype=np.int64)
    np.savez(cache, x=x, y=y)
    return x, y


# ----------------------------------------------------------------------------- DiffAugment
def diff_augment(x):
    """Differentiable, spectrogram-safe augmentations applied identically in spirit to real & fake."""
    b = x.size(0)
    # gain offset (log domain => loudness change)
    x = x + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.2
    # time shift up to 1/8 of the width, frequency shift up to 2 mel bins (zero-filled via grid_sample)
    tx = (torch.rand(b, device=x.device) - 0.5) * 2 * (1 / 8) * 2      # in [-1,1] grid units
    ty = (torch.randint(-2, 3, (b,), device=x.device).float()) * 2 / H
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = 1; theta[:, 1, 1] = 1; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x + 1, grid, padding_mode="zeros", align_corners=False) - 1   # pad with "silence" (-1)
    # SpecAugment-style cutout: one time mask (<=16 frames) and one freq mask (<=8 bins)
    t0 = torch.randint(0, W, (b, 1), device=x.device); tw = torch.randint(0, 17, (b, 1), device=x.device)
    f0 = torch.randint(0, H, (b, 1), device=x.device); fw = torch.randint(0, 9, (b, 1), device=x.device)
    tt = torch.arange(W, device=x.device)[None]; ff = torch.arange(H, device=x.device)[None]
    tm = ((tt >= t0) & (tt < t0 + tw)).float()[:, None, None, :]
    fm = ((ff >= f0) & (ff < f0 + fw)).float()[:, None, :, None]
    keep = 1 - torch.clamp(tm + fm, 0, 1)
    return x * keep + x.mean(dim=(2, 3), keepdim=True) * (1 - keep)     # masked area -> sample mean


# ----------------------------------------------------------------------------- models
class CondBN(nn.Module):
    """Class-conditional BatchNorm (BigGAN-style): label + z modulate each G layer."""
    def __init__(self, ch, cdim):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch, affine=False)
        self.g = nn.Linear(cdim, ch); self.b = nn.Linear(cdim, ch)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.b.weight)
        nn.init.ones_(self.g.bias); nn.init.zeros_(self.b.bias)

    def forward(self, x, c):
        return self.bn(x) * self.g(c)[:, :, None, None] + self.b(c)[:, :, None, None]


class GBlock(nn.Module):
    def __init__(self, cin, cout, cdim):
        super().__init__()
        self.bn1, self.bn2 = CondBN(cin, cdim), CondBN(cout, cdim)
        self.c1 = nn.Conv2d(cin, cout, 3, 1, 1); self.c2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.sc = nn.Conv2d(cin, cout, 1)

    def forward(self, x, c):
        h = F.interpolate(F.relu(self.bn1(x, c)), scale_factor=2, mode="nearest")
        h = self.c2(F.relu(self.bn2(self.c1(h), c)))
        return h + self.sc(F.interpolate(x, scale_factor=2, mode="nearest"))


class Generator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.emb = nn.Embedding(N_CLASSES, 64)
        cdim = 64 + LATENT
        self.fc = nn.Linear(LATENT, ch * 8 * (H // 16) * (W // 16))
        self.blocks = nn.ModuleList([GBlock(ch * 8, ch * 4, cdim), GBlock(ch * 4, ch * 2, cdim),
                                     GBlock(ch * 2, ch, cdim), GBlock(ch, ch, cdim)])
        self.out = nn.Sequential(nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch, 1, 3, 1, 1), nn.Tanh())
        self.ch = ch

    def forward(self, z, y):
        c = torch.cat([z, self.emb(y)], 1)
        h = self.fc(z).view(-1, self.ch * 8, H // 16, W // 16)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


class MinibatchStd(nn.Module):
    def forward(self, x, group=4):
        b, c, h, w = x.shape
        g = min(group, b)
        if b % g:
            g = 1
        s = x.view(g, -1, c, h, w).std(0, unbiased=False).mean(dim=(1, 2, 3))   # [b/g]
        s = s.repeat(g).view(b, 1, 1, 1).expand(b, 1, h, w)
        return torch.cat([x, s], 1)


class Discriminator(nn.Module):
    """SN-ResNet-ish D with minibatch-stddev and a projection head for the label."""
    def __init__(self, ch=64):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(SN(nn.Conv2d(i, o, 3, 1, 1)), nn.LeakyReLU(0.2),
                                 SN(nn.Conv2d(o, o, 4, 2, 1)), nn.LeakyReLU(0.2))
        self.body = nn.Sequential(blk(1, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4), blk(ch * 4, ch * 8))
        self.mbstd = MinibatchStd()
        self.final = nn.Sequential(SN(nn.Conv2d(ch * 8 + 1, ch * 8, 3, 1, 1)), nn.LeakyReLU(0.2))
        self.lin = SN(nn.Linear(ch * 8, 1))
        self.emb = SN(nn.Embedding(N_CLASSES, ch * 8))

    def forward(self, x, y):
        h = self.final(self.mbstd(self.body(x))).sum(dim=(2, 3))      # global sum pooling
        return self.lin(h).squeeze(1) + (self.emb(y) * h).sum(1)    # projection term


# ----------------------------------------------------------------------------- train
def train(x, y, out_dir, iters=6000, bs=64, ms_weight=1.0, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    X = torch.from_numpy(x)[:, None].to(device); Y = torch.from_numpy(y).to(device)
    # class-balanced sampling so the small "both" class is not starved
    w = (1.0 / np.bincount(y))[y]; w = torch.from_numpy(w / w.sum()).float().to(device)
    G, D = Generator().to(device), Discriminator().to(device)
    G_ema = copy.deepcopy(G).eval()
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.999))
    oD = torch.optim.Adam(D.parameters(), 4e-4, betas=(0.0, 0.999))
    t0 = time.time()
    for it in range(1, iters + 1):
        idx = torch.multinomial(w, bs, replacement=True)
        xr, yr = X[idx], Y[idx]
        # ---- D (hinge)
        z = torch.randn(bs, LATENT, device=device)
        with torch.no_grad():
            xf = G(z, yr)
        d_loss = F.relu(1 - D(diff_augment(xr), yr)).mean() + F.relu(1 + D(diff_augment(xf), yr)).mean()
        oD.zero_grad(set_to_none=True); d_loss.backward(); oD.step()
        # ---- G (hinge + mode seeking)
        z1, z2 = torch.randn(bs, LATENT, device=device), torch.randn(bs, LATENT, device=device)
        f1, f2 = G(z1, yr), G(z2, yr)
        g_adv = -D(diff_augment(torch.cat([f1, f2])), torch.cat([yr, yr])).mean()
        ms = (f1 - f2).abs().mean(dim=(1, 2, 3)) / (z1 - z2).abs().mean(1)
        g_ms = 1.0 / (ms.mean() + 1e-5)
        g_loss = g_adv + ms_weight * g_ms
        oG.zero_grad(set_to_none=True); g_loss.backward(); oG.step()
        # ---- EMA
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1 - 0.999)
            for be, b in zip(G_ema.buffers(), G.buffers()):
                be.copy_(b)
        if it % 250 == 0:
            log.info("it %5d/%d | D=%.3f G_adv=%.3f MS=%.3f | %.1fm", it, iters, d_loss.item(),
                     g_adv.item(), g_ms.item(), (time.time() - t0) / 60)
        if it % 2000 == 0 or it == iters:
            torch.save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "D": D.state_dict()},
                       os.path.join(out_dir, f"ckpt_it{it}.pt"))
    torch.save(G_ema.state_dict(), os.path.join(out_dir, "generator_ema.pt"))
    return G_ema


@torch.no_grad()
def sample(G, n_per_class, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for k in range(N_CLASSES):
        xs = []
        for i in range(0, n_per_class, 250):
            m = min(250, n_per_class - i)
            z = torch.randn(m, LATENT, device=device, generator=g)
            xs.append(G(z, torch.full((m,), k, device=device)).cpu().numpy()[:, 0])
        out[k] = np.concatenate(xs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--synth-per-class", type=int, default=1000)
    ap.add_argument("--window-sec", type=int, default=0,
                    help="0 = original 3 s zero-padded windows (64x128); 8 or 16 = whole cycles, repeat-padded")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    global W
    if a.window_sec:
        W = a.window_sec * 16000 // HOP
    a.out_dir = a.out_dir or os.path.join(HERE, f"gan_v2_out_{a.window_sec}s" if a.window_sec else "gan_v2_out")
    os.makedirs(a.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(a.out_dir, "train.log"))])
    torch.manual_seed(0); np.random.seed(0)
    x, y = load_real_specs_window(a.window_sec) if a.window_sec else load_real_specs()
    log.info("real specs: %s | per class %s", x.shape, np.bincount(y))
    G = train(x, y, a.out_dir, iters=a.iters, bs=a.batch)
    fake = sample(G, a.synth_per_class)
    specs = np.concatenate([fake[k] for k in range(N_CLASSES)])
    # labels in base.CLASS_NAMES indices, same as gan_pipeline.py (crackle=1, wheeze=2, both=3)
    labels = np.concatenate([np.full(a.synth_per_class, k + 1) for k in range(N_CLASSES)]).astype(np.int64)
    np.save(os.path.join(a.out_dir, "synthetic_specs.npy"), specs.astype(np.float32))
    np.save(os.path.join(a.out_dir, "synthetic_labels.npy"), labels)
    log.info("saved %d synthetic spectrograms to %s", len(specs), a.out_dir)


if __name__ == "__main__":
    main()
