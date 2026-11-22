"""
meldiff.py -- class-conditional DIFFUSION on the GAN's log-mel windows, with classifier-free guidance (2026-11-08).

Why: a different generator family than the GANs, and the guidance scale w is a continuous "class clarity" dial:
  w = 0  unconditional (no class information in the sample, label assigned afterwards)
  w = 1  plain conditional model
  w > 1  over-guided, caricature-clear class evidence (lower diversity)
so one trained model gives a dose-response curve of AST score vs class clarity (the HF-teacher arms suggested
clearer fakes hurt specificity).

Data: exactly the GAN's (gan.Data): ICBHI gan-train cycles, crop_pad to 512 frames x 128 mel, normalised by the
dataset mu / sd. --hf adds the HF_Lung cache (rhonchi-only rows dropped, source flag = 1; samples for the AST are
always drawn with source 0). HF_Lung only ever trains this generator, never the AST.

Objectives (--obj):
  edm  Karras et al. 2022 (EDM) preconditioning + loss weighting, Heun sampler (18 steps, Karras sigma schedule)
  fm   flow matching / rectified flow (Lipman 2023, Liu 2023): x_t = (1-t) x + t eps, predict v = eps - x, Euler 32 steps
Network: 2x2 pixel-unshuffle -> U-Net on 64 x 256 (4 levels, 64-128-192-256 channels, 2 ResBlocks per level,
self-attention at the 8 x 32 level and the middle) -> pixel-shuffle, noise-level + class (+ source) embedding with scale/shift modulation. Class dropout 15% (null class = 4).

    $PY MELDIFF/meldiff.py --name d1_edm --obj edm
    $PY MELDIFF/meldiff.py --name d3_edm_hf --obj edm --hf
Writes MELDIFF/runs/<name>/{config.json, ema_<it>.pt, samples_<it>.npz (w=1), train.log}; evaluation rows (same
suite as the GANs) go to GAN_FINETUNING/eval/results.csv as "<name>@<it>".
"""
import os
import sys
import json
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P
RUNS = os.path.join(_P.OUT, "diffusion_runs")
NULL = 4                                             # class token for "no class" (classifier-free guidance)


# ------------------------------------------------------------------------------------------------ network
class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, drop=0.1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.e(e)[:, :, None, None].chunk(2, 1)
        h = self.c2(self.drop(F.silu(self.n2(h) * (1 + s) + b)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n, self.qkv, self.o, self.h = nn.GroupNorm(32, ch), nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1), heads
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).transpose(3, 4).unbind(1)
        y = F.scaled_dot_product_attention(q, k, v)                    # [B, h, HW, C/h]
        return x + self.o(y.transpose(2, 3).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, ch=(64, 128, 192, 256), nres=2, attn_levels=(3,), emb=256, n_src=0):
        super().__init__()
        self.emb_dim = emb
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cls = nn.Embedding(5, emb)
        self.src = nn.Embedding(n_src, emb) if n_src else None
        self.inp = nn.Conv2d(4, ch[0], 3, padding=1)        # after a 2x2 pixel-unshuffle: 64 x 256
        self.down, chans, c = nn.ModuleList(), [ch[0]], ch[0]
        for i, co in enumerate(ch):
            for _ in range(nres):
                self.down.append(nn.ModuleList([ResBlock(c, co, emb), Attn(co) if i in attn_levels else nn.Identity()]))
                c = co; chans.append(c)
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb), Attn(c), ResBlock(c, c, emb)])
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(nres + 1):
                self.up.append(nn.ModuleList([ResBlock(c + chans.pop(), co, emb), Attn(co) if i in attn_levels else nn.Identity()]))
                c = co
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 4, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def temb(self, t):
        half = self.emb_dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = 1000 * t[:, None].float() * f[None]
        return self.t_mlp(torch.cat([a.sin(), a.cos()], 1))

    def forward(self, x, t, cls, src=None):
        e = self.temb(t) + self.cls(cls)
        if self.src is not None:
            e = e + self.src(src if src is not None else torch.zeros_like(cls))
        h = self.inp(F.pixel_unshuffle(x, 2))
        hs = [h]
        for m in self.down:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](h, e))
            else:
                h = m(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for m in self.up:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](torch.cat([h, hs.pop()], 1), e))
            else:
                h = m(h)
        return F.pixel_shuffle(self.out(h), 2)


# ------------------------------------------------------------------------------------------------ objectives
SD = 1.0                                             # data sd after the dataset normalisation


def edm_denoise(net, x, sigma, cls, src=None):
    """EDM preconditioned denoiser D(x; sigma)."""
    s = sigma.view(-1, 1, 1, 1)
    c_skip, c_out = SD ** 2 / (s ** 2 + SD ** 2), s * SD / (s ** 2 + SD ** 2).sqrt()
    c_in = 1 / (s ** 2 + SD ** 2).sqrt()
    return c_skip * x + c_out * net(c_in * x, sigma.log() / 4, cls, src).float()


def loss_fn(net, obj, x, cls, src):
    B = len(x)
    if obj == "edm":
        sigma = (torch.randn(B, device=x.device) * 1.2 - 1.2).exp()
        w = (sigma ** 2 + SD ** 2) / (sigma * SD) ** 2
        n = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
        D = edm_denoise(net, x + n, sigma, cls, src)
        return (w.view(-1, 1, 1, 1) * (D - x) ** 2).mean()
    t = torch.sigmoid(torch.randn(B, device=x.device))                 # logit-normal t (SD3)
    eps = torch.randn_like(x)
    xt = (1 - t.view(-1, 1, 1, 1)) * x + t.view(-1, 1, 1, 1) * eps
    v = net(xt, t, cls, src).float()
    return ((v - (eps - x)) ** 2).mean()


@torch.no_grad()
def sample(net, obj, cls, w=1.0, steps=None, seed=0, src=None, bs=128):
    """cls: LongTensor [N] (0..3) -> normalised samples [N,1,128,512]. CFG: out = uncond + w (cond - uncond)."""
    dev = next(net.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    out = []
    for i in range(0, len(cls), bs):
        c = cls[i:i + bs].to(dev)
        s_ = None if src is None else src[i:i + bs].to(dev)
        n = len(c)
        cc, ss = torch.cat([c, torch.full_like(c, NULL)]), None if s_ is None else torch.cat([s_, s_])

        def guided(fn, x, lvl):
            if w == 1.0:
                return fn(x, lvl, c, s_)
            if w == 0.0:
                return fn(x, lvl, torch.full_like(c, NULL), s_)
            d = fn(torch.cat([x, x]), torch.cat([lvl, lvl]), cc, ss)
            dc, du = d[:n], d[n:]
            return du + w * (dc - du)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if obj == "edm":
                K = steps or 18                                              # 35 network evaluations
                smin, smax, rho = 0.002, 80.0, 7.0
                r = torch.arange(K, device=dev) / (K - 1)
                sig = (smax ** (1 / rho) + r * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
                sig = torch.cat([sig, torch.zeros(1, device=dev)])
                x = torch.randn(n, 1, 128, 512, device=dev, generator=g) * sig[0]
                fn = lambda x_, s1, c_, s2: edm_denoise(net, x_, s1, c_, s2)
                for j in range(K):
                    s0, s1 = sig[j], sig[j + 1]
                    d0 = (x - guided(fn, x, s0.expand(n))) / s0
                    xn = x + (s1 - s0) * d0
                    if s1 > 0:                                           # Heun correction
                        d1 = (xn - guided(fn, xn, s1.expand(n))) / s1
                        xn = x + (s1 - s0) * 0.5 * (d0 + d1)
                    x = xn
            else:
                K = steps or 32
                x = torch.randn(n, 1, 128, 512, device=dev, generator=g)
                fn = lambda x_, t_, c_, s2: net(x_, t_, c_, s2).float()
                ts = torch.linspace(1, 0, K + 1, device=dev)
                for j in range(K):
                    v = guided(fn, x, ts[j].expand(n))
                    x = x + (ts[j + 1] - ts[j]) * v
        out.append(x.float())
    return torch.cat(out)


# ------------------------------------------------------------------------------------------------ run I/O
def load_run(run_dir, it=None, device="cuda"):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    it = it or max(int(f[4:-3]) for f in os.listdir(run_dir) if f.startswith("ema_") and f.endswith(".pt"))
    net = UNet(n_src=10 if cfg.get("dev") else 2 if cfg["hf"] else 0).to(device)
    net.load_state_dict(torch.load(os.path.join(run_dir, "ema_%d.pt" % it), map_location=device))
    return net.eval(), cfg, it


def generate(run_dir, labels, w=1.0, seed=0, it=None, steps=None, src_idx=None):
    """-> raw log-mel [N,512,128] float32 for the class labels (0..3), source 0 (ICBHI style)."""
    net, cfg, it = load_run(run_dir, it)
    cls = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    src = torch.zeros_like(cls) if cfg["hf"] else None
    if cfg.get("dev"):                                   # --dev: source * 5 + device (source 0 = ICBHI style)
        src = torch.as_tensor(np.asarray(src_idx), dtype=torch.long)
    x = sample(net, cfg["obj"], cls, w=w, steps=steps, seed=seed, src=src)
    x = x[:, 0].transpose(1, 2) * cfg["sd"] + cfg["mu"]
    del net
    torch.cuda.empty_cache()
    return x.cpu().numpy().astype(np.float32), it


# ------------------------------------------------------------------------------------------------ training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--obj", choices=["edm", "fm"], default="edm")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--p-uncond", type=float, default=0.15)
    ap.add_argument("--hf", action="store_true", help="+ HF_Lung cache (rhonchi-only dropped), source-conditioned")
    ap.add_argument("--hf-frac", type=float, default=0.5)
    ap.add_argument("--dev", action="store_true",
                    help="2026-11-20: also condition on the stethoscope (ICBHI's 4 + HF-Type-1; HF steth = Litt3200), "
                         "so fakes can be drawn in the style of a given test device (Litt3200: 41 ICBHI train cycles)")
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import gan as M
    out = os.path.join(RUNS, a.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logf = open(os.path.join(out, "train.log"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    attrs = M.attrs_for("pre" if a.dev else "cs")           # crackle, wheeze, (device,) source
    data = M.Data(attrs, a.seed, hf=a.hf, hf_frac=a.hf_frac, hf_clean=True)
    data.start_prefetch(a.batch)
    nsrc = 10 if a.dev else 2 if a.hf else 0
    net = UNet(n_src=nsrc).cuda()
    ema = UNet(n_src=nsrc).cuda().eval()
    ema.load_state_dict(net.state_dict())
    for p in ema.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(net.parameters(), a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    json.dump(dict(vars(a), mu=data.mu, sd=data.sd), open(os.path.join(out, "config.json"), "w"), indent=1)
    log("%s | %d train samples | %.1fM params | %s" % (a.name, len(data.rows), sum(p.numel() for p in net.parameters()) / 1e6, vars(a)))
    import evaluate as E
    ref, t0 = None, time.time()
    econd, elab = data.eval_conditions(np.random.RandomState(123))
    for it in range(1, a.iters + 1):
        x, c = data.batch(a.batch, "cuda")
        cls = 2 * 0 + c[:, 0] + 2 * c[:, 1]                     # (crackle, wheeze) -> 0 normal 1 crackle 2 wheeze 3 both
        cls = torch.where(torch.rand(len(cls), device="cuda") < a.p_uncond, torch.full_like(cls, NULL), cls)
        src = (c[:, 3] * 5 + c[:, 2]) if a.dev else c[:, 2] if a.hf else None
        lr = a.lr * min(1.0, it / 1000)
        for gp in opt.param_groups:
            gp["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(net, a.obj, x.float(), cls, src)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - a.ema)
        if it % 500 == 0:
            log(json.dumps(dict(it=it, loss=round(loss.item(), 4), min=round((time.time() - t0) / 60, 1))))
        if it % a.eval_every == 0 or it == a.iters:
            torch.save(ema.state_dict(), os.path.join(out, "ema_%d.pt" % it))
            cls_e = torch.as_tensor(elab, dtype=torch.long)
            src_e = torch.as_tensor(econd[:, 3] * 5 + econd[:, 2]) if a.dev else torch.zeros_like(cls_e) if a.hf else None
            xs = sample(ema, a.obj, cls_e, w=1.0, seed=0, src=src_e)
            xr = (xs[:, 0].transpose(1, 2) * data.sd + data.mu).cpu().numpy().astype(np.float16)
            np.savez(os.path.join(out, "samples_%d.npz" % it), x=xr, y=elab, cond=econd)
            try:
                if os.environ.get("NA_GAN_EVAL", "0") != "1":   # release: judge/FD suite is opt-in
                    raise StopIteration
                if ref is None:
                    ref = E.build_reference()
                r = E.evaluate_set("%s@%d" % (a.name, it), xr, elab, ref, "meldiff obj=%s hf=%s w=1" % (a.obj, a.hf))
                log("EVAL " + json.dumps(dict(it=it, eval={k: round(float(r[k]), 4) for k in E.COLS})))
            except StopIteration:
                pass
            except torch.OutOfMemoryError:
                log("EVAL SKIPPED (OOM) at %d" % it)
            finally:
                E.J._J, E._AS = None, None
                torch.cuda.empty_cache()
    log("DONE %.1f min" % ((time.time() - t0) / 60))


POOL_SEED = 0


def pool_path(run_dir, w, it):
    return os.path.join(run_dir, "pool_w%g_it%d.npz" % (w, it))


def pool(run, w, per_class=1000, it=None):
    """The AST fake set of one (model, guidance scale): per_class per minority class, generated ONCE (seed 0) and
    shared by the AST seeds (sampling is ~100x a GAN's cost); the AST seeds still differ in the real background noise
    mixed in and in training. Cached in runs/<run>/pool_w<w>_it<it>.npz (raw log-mel float16)."""
    run_dir = os.path.join(RUNS, run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    it = it or cfg["iters"]
    f = pool_path(run_dir, w, it)
    if os.path.exists(f):
        z = np.load(f)
        return z["x"].astype(np.float32), z["y"], it, (list(z["devs"]) if "devs" in z else None)
    y = np.concatenate([[k] * per_class for k in (1, 2, 3)])
    t0 = time.time()
    devs, src_idx = None, None
    if cfg.get("dev"):                                   # each fake gets a test device, uniform (ast_gan.TEST_DEVICES)
        import gan as M
        devs = list(np.random.RandomState(POOL_SEED + 7).choice(["AKGC417L", "Litt3200", "Meditron"], len(y)))
        src_idx = np.array([M.DEVICES5.index(d) for d in devs])          # source 0 (ICBHI style) * 5 + device
    x, it = generate(run_dir, y, w=w, seed=POOL_SEED, it=it, src_idx=src_idx,
                     steps=int(os.environ.get("NA_SAMPLE_STEPS", "0")) or None)   # smoke runs: few steps
    np.savez(f, x=x.astype(np.float16), y=y, **({"devs": np.array(devs)} if devs else {}))
    print("[meldiff] pool %s w=%g @%d: %d samples in %.1f min" % (run, w, it, len(y), (time.time() - t0) / 60), flush=True)
    return x, y, it, devs


if __name__ == "__main__":
    if sys.argv[1:2] == ["--pool"]:                      # $PY MELDIFF/meldiff.py --pool RUN W [W ...]
        for w_ in sys.argv[3:]:
            pool(sys.argv[2], float(w_))
    else:
        main()
