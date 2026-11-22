"""
gan_common.py -- shared pieces so every GAN method differs ONLY in its loss / regularisation.

  * data      : 8 s whole minority cycles, repeat-padded, log-mel 64x256 in [-1,1] (gan_v2 cache)
  * Generator : same for every method (residual upsampling, class-conditional BN, z in every layer)
  * Critic    : same conv body; options = spectral norm, minibatch-stddev, projection head, AC head
  * fixes     : the mode-collapse "fixes" switched on together by --fixes:
                DiffAugment (real+fake), minibatch-stddev in D, mode-seeking loss on G
  * training  : class-balanced sampling, EMA generator, checkpoints, 1000 samples/class at the end

Each method file (gan_wgan.py, gan_wgangp.py, ...) only provides  d_loss(...)  and  g_loss(...)
plus its optimiser settings, and calls  run(method).
"""
import os, sys, copy, json, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

HERE = os.path.dirname(os.path.abspath(__file__))
GAN_CHECK = os.path.dirname(HERE)
sys.path.insert(0, GAN_CHECK)
sys.path.insert(0, os.path.dirname(GAN_CHECK))
import gan_v2 as V2                                     # noqa: E402  data loader, DiffAugment, G blocks

# Window setting, chosen with env var GAN_WINDOW:
#   "8"    (default) 8 s, plain repeat padding          -> runs/
#   "4"    4 s, plain repeat padding                     -> runs_4s/
#   "8xf"  8 s, repeat padding with 30 ms crossfade      -> runs_8sxf/
WINDOW_TAG = os.environ.get("GAN_WINDOW", "8")
WINDOW_SEC = int(WINDOW_TAG.replace("xf", ""))
XFADE_MS = 30 if WINDOW_TAG.endswith("xf") else 0
N_CLASSES, LATENT, H = 3, V2.LATENT, V2.H
W = WINDOW_SEC * 16000 // V2.HOP                        # 8 s -> 256 frames, 4 s -> 128
V2.W = W                                                # diff_augment / Generator read V2.W
NAMES = V2.NAMES
import na_paths as _P
RUNS_DIR = os.path.join(_P.OUT, "wgangp_runs" if WINDOW_TAG == "8" else f"wgangp_runs_{WINDOW_SEC}s")
log = logging.getLogger("gan")


def load_data():
    return V2.load_real_specs_window(WINDOW_SEC, XFADE_MS)


# ----------------------------------------------------------------------------- models
Generator = V2.Generator


class Critic(nn.Module):
    """Shared discriminator/critic. forward -> (adv_score[b], class_logits[b,3] or None)."""
    def __init__(self, sn=True, mbstd=False, projection=True, aux=False, ch=64):
        super().__init__()
        S = spectral_norm if sn else (lambda m: m)
        def blk(i, o):
            return nn.Sequential(S(nn.Conv2d(i, o, 3, 1, 1)), nn.LeakyReLU(0.2),
                                 S(nn.Conv2d(o, o, 4, 2, 1)), nn.LeakyReLU(0.2))
        self.body = nn.Sequential(blk(1, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4), blk(ch * 4, ch * 8))
        self.mbstd = V2.MinibatchStd() if mbstd else None
        self.final = nn.Sequential(S(nn.Conv2d(ch * 8 + int(mbstd), ch * 8, 3, 1, 1)), nn.LeakyReLU(0.2))
        self.lin = S(nn.Linear(ch * 8, 1))
        self.emb = S(nn.Embedding(N_CLASSES, ch * 8)) if projection else None
        self.cls = S(nn.Linear(ch * 8, N_CLASSES)) if aux else None

    def forward(self, x, y):
        h = self.body(x)
        if self.mbstd is not None:
            h = self.mbstd(h)
        h = self.final(h).sum(dim=(2, 3))
        out = self.lin(h).squeeze(1)
        if self.emb is not None:
            out = out + (self.emb(y) * h).sum(1)
        return out, (self.cls(h) if self.cls is not None else None)


# ----------------------------------------------------------------------------- method spec
class Method:
    """Override in each method file."""
    name = "base"
    sn, projection, aux = True, True, False
    n_critic = 1
    lr_g, lr_d, betas = 1e-4, 4e-4, (0.0, 0.999)
    optim = "adam"

    def __init__(self, fixes):
        self.fixes = fixes

    def augment(self, x):                       # D input augmentation (DiffAugment when fixes on)
        return V2.diff_augment(x) if self.fixes else x

    def d_loss(self, D, xr, xf, y):             # -> scalar, dict of logs
        raise NotImplementedError

    def g_loss(self, D, xf, y):                 # -> scalar (adversarial part only)
        raise NotImplementedError

    def after_d_step(self, D):                  # e.g. weight clipping
        pass


# ----------------------------------------------------------------------------- training loop
def train(method, x, y, out_dir, iters, bs, device="cuda"):
    X = torch.from_numpy(x)[:, None].to(device); Y = torch.from_numpy(y).to(device)
    wts = (1.0 / np.bincount(y))[y]
    wts = torch.from_numpy(wts / wts.sum()).float().to(device)          # class-balanced sampling
    G = Generator().to(device)
    D = Critic(sn=method.sn, mbstd=method.fixes, projection=method.projection, aux=method.aux).to(device)
    G_ema = copy.deepcopy(G).eval()
    if method.optim == "rmsprop":
        oG = torch.optim.RMSprop(G.parameters(), method.lr_g)
        oD = torch.optim.RMSprop(D.parameters(), method.lr_d)
    else:
        oG = torch.optim.Adam(G.parameters(), method.lr_g, betas=method.betas)
        oD = torch.optim.Adam(D.parameters(), method.lr_d, betas=method.betas)
    hist, t0 = [], time.time()
    for it in range(1, iters + 1):
        # ---------------- critic / discriminator
        for _ in range(method.n_critic):
            idx = torch.multinomial(wts, bs, replacement=True)
            xr, yr = X[idx], Y[idx]
            with torch.no_grad():
                xf = G(torch.randn(bs, LATENT, device=device), yr)
            d_loss, dlog = method.d_loss(D, xr, xf, yr)
            oD.zero_grad(set_to_none=True); d_loss.backward(); oD.step()
            method.after_d_step(D)
        # ---------------- generator
        idx = torch.multinomial(wts, bs, replacement=True)
        yr = Y[idx]
        z1 = torch.randn(bs, LATENT, device=device)
        if method.fixes:                                                 # mode-seeking (MSGAN)
            z2 = torch.randn(bs, LATENT, device=device)
            f1, f2 = G(z1, yr), G(z2, yr)
            g_adv = method.g_loss(D, torch.cat([f1, f2]), torch.cat([yr, yr]))
            ms = (f1 - f2).abs().mean(dim=(1, 2, 3)) / (z1 - z2).abs().mean(1)
            g_ms = 1.0 / (ms.mean() + 1e-5)
            g_loss = g_adv + g_ms
        else:
            g_adv = method.g_loss(D, G(z1, yr), yr)
            g_ms = torch.zeros(())
            g_loss = g_adv
        oG.zero_grad(set_to_none=True); g_loss.backward(); oG.step()
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1 - 0.999)
            for be, b in zip(G_ema.buffers(), G.buffers()):
                be.copy_(b)
        if it % 250 == 0:
            rec = dict(it=it, d=float(d_loss), g=float(g_adv), ms=float(g_ms), **dlog,
                       min=(time.time() - t0) / 60)
            hist.append(rec)
            log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if it % 2000 == 0 or it == iters:
            torch.save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "D": D.state_dict(), "it": it},
                       os.path.join(out_dir, "ckpt.pt"))
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    torch.save(G_ema.state_dict(), os.path.join(out_dir, "generator_ema.pt"))
    return G_ema


def run(method_cls):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixes", action="store_true", help="DiffAugment + minibatch-stddev + mode-seeking")
    ap.add_argument("--iters", type=int, default=6000, help="generator iterations")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--synth-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    method = method_cls(a.fixes)
    tag = method.name + ("_fixes" if a.fixes else "_plain")
    out_dir = os.path.join(RUNS_DIR, tag); os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", force=True, handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "train.log"))])
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    x, y = load_data()
    log.info("%s | data %s per-class %s | iters=%d bs=%d n_critic=%d", tag, x.shape, np.bincount(y),
             a.iters, a.batch, method.n_critic)
    G = train(method, x, y, out_dir, a.iters, a.batch)
    fake = V2.sample(G, a.synth_per_class)
    np.save(os.path.join(out_dir, "synthetic_specs.npy"),
            np.concatenate([fake[k] for k in range(N_CLASSES)]).astype(np.float32))
    np.save(os.path.join(out_dir, "synthetic_labels.npy"),       # base.CLASS_NAMES ids: 1 crackle, 2 wheeze, 3 both
            np.concatenate([np.full(a.synth_per_class, k + 1) for k in range(N_CLASSES)]).astype(np.int64))
    log.info("DONE %s", tag)
