"""
ast_gan.py -- Step 5: AST (sg_exp S1 protocol: DAT, plain CE, 8 s, 30 epochs) + synthetic cycles from a
GAN_FINETUNING GAN, evaluated on the official ICBHI test split.

Identical to `experiment_stethoscope/sg_exp.py --exp S1 --gan <old run>` except for the synthetic data:
  * generator  runs/<RUN>/G_ema_<it>.pt, checkpoint = best evaluated one by tstr (ties: lower FD)
  * 1000 per minority class (same as the old GAN rows); class via crackle / wheeze flags
  * device     uniform over --synth-devices (default: the TEST devices AKGC417L / Litt3200 / Meditron) for
               device-conditioned GANs, passed to DAT as a real device label; class-only GANs -> -1 (ignored by DAT)
  * region     drawn from the class's real train distribution (region-conditioned GANs only)
  * length     the generated cycle (frames up to the silence floor) is tiled to 8 s like real cycles in
               sg_exp (cyclic padding), then AST-normalised -> same input format as real cycles
Outputs go to GAN_FINETUNING/ast_runs/S1_<NAME>/ (final_summary.json, per_seed_results.csv, train.log).
Registered in results_summary/all_runs.csv as GF_S1_<NAME> (sg_exp's own registration is off: it would record a
wrong command line). The method name says how many seeds were run.

Non-GAN arms (--control), same protocol and size, for comparison with the GAN rows:
  noise    class-mean Gaussian "spectrograms" (per-pixel mean/std of real minority windows): tests whether any
           extra minority-labelled input gives the gain
  shuffle  samples of --run with minority labels permuted: tests whether the GAN's class information matters
  --no-synth  no synthetic data at all (S1 itself, for the training variants below)

Training variants taken from the old screening funnel (outputs_combined_s1s2s3, README E12), both untested
in the 8 s + DAT pipeline until now; each adds a suffix to NAME:
  --mixup    (_mix) S3-C Mixup / SpecMix exactly as mainv2.py (alpha 0.4, p 0.8, SpecMix prob 0.5), applied after
             SpecAugment; class AND device (DAT) targets are mixed with the same lambda
  --zeropad  (_zp)  cycles are zero-padded to 8 s (silence) instead of tiled; real + synthetic, train + test
  --synth-devices AKGC417L,Meditron  (_dev-AM) e.g. no synthetic Litt3200 cycles (the GAN saw only 30 real ones)
  --synth-per-class N  (_pcN when != 1000)

Usage: $PY GAN_FINETUNING/ast_gan.py --run c2_dev [--seeds 13]
       $PY GAN_FINETUNING/ast_gan.py --no-synth --mixup
       $PY GAN_FINETUNING/ast_gan.py --control noise
       $PY GAN_FINETUNING/ast_gan.py --control shuffle --run c1_class
"""
import os
import sys
import math
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import na_paths as _P
import gfdata as G                                  # noqa: E402
import gan as M                                     # noqa: E402

TEST_DEVICES = ["AKGC417L", "Litt3200", "Meditron"]
MIX = dict(alpha=0.4, p=0.8, specmix_prob=0.5)      # mainv2.py S3-C defaults (screening-funnel winner)
PMCL = dict(w=1.0, tau=0.06, alpha=1.0)             # --patchmix: loss weight, temperature, Beta(alpha, alpha) mix ratio
ZEROPAD = False                                     # set by --zeropad


FORCE_CKPT = 0                                      # set by --ckpt
REJECT_MAX = 0                                      # set by --reject-max (0 = off, the behaviour of all runs before 2026-10-22)
BG_NOISE = False                                    # set by --bg-noise (bgnoise.py: Round 15 real background, 2026-10-24)
FILTER = 0                                          # --filter K: generate K x, keep the judge's most class-confident (2026-11-07)
TRUNC = 1.0                                         # --trunc psi: truncation of the GAN's z
SOFTW = False                                       # --soft-weight: each fake's loss weighted by the judge's confidence
DEV_NOISE = False                                   # --dev-noise: fakes get a device (DAT label) + noise of that device
DCT = False                                         # set by --dct: --run is a DCT_GAN/runs/<RUN> (1D DCT GAN, 2026-10-28)


def pick_checkpoint(run_dir):
    """Best evaluated checkpoint by TSTR (ties: lower FD), among FULLY conditional checkpoints only (it >= trans_end):
    samples for AST are generated with full conditioning, so a checkpoint from the transitional / unconditional phase
    must not be picked (its in-training eval also used the wrong condition strength before 2026-10-21 21:35)."""
    hist = json.load(open(os.path.join(run_dir, "history.json")))
    if FORCE_CKPT:
        return FORCE_CKPT, next((h["eval"] for h in hist if h.get("it") == FORCE_CKPT and "eval" in h), None)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    min_it = cfg.get("trans_end", 0) or 0
    ev = [h for h in hist if "eval" in h and h["it"] >= min_it
          and os.path.exists(os.path.join(run_dir, "G_ema_%d.pt" % h["it"]))]
    if ev:
        best = max(ev, key=lambda h: (round(h["eval"]["tstr"], 1), -h["eval"]["FD"]))
        return best["it"], best["eval"]
    its = sorted(int(f[6:-3]) for f in os.listdir(run_dir) if f.startswith("G_ema_") and int(f[6:-3]) >= min_it)
    return its[-1], None


def valid_span(x):
    """Frames of the generated cycle: from the first to the last frame that is not silence floor."""
    v = np.where(x.std(axis=1) > 0.5)[0]
    return x[v[0]:v[-1] + 1] if len(v) >= 50 else x


_SILENCE = None


def silence_frame():
    """Raw log-mel of a zero waveform (what zero-padding produces in the real-cycle path)."""
    global _SILENCE
    if _SILENCE is None:
        _SILENCE = G.fbank(np.zeros(G.SR, dtype=np.float32))[5]
    return _SILENCE


def to_ast(x, frames):
    """Raw log-mel windows [N,512,128] -> AST input [N,frames,128]: the cycle frames tiled to 8 s like sg_exp's
    padding, or followed by silence with --zeropad."""
    out = []
    for v in x:
        sp = valid_span(v)[:frames]
        if ZEROPAD:
            sp = np.concatenate([sp, np.tile(silence_frame(), (frames - len(sp), 1))])
        else:
            sp = np.tile(sp, (int(np.ceil(frames / len(sp))), 1))[:frames]
        out.append(G.ast_normalize(sp).astype(np.float16))
    return np.stack(out)


@torch.no_grad()
def synthesize(run, per_class, seed, frames, devices=TEST_DEVICES):
    run_dir = os.path.join(M.RUNS, run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    attrs = [tuple(a) for a in cfg["attrs"]]
    it, ev = pick_checkpoint(run_dir)
    Gm = (M.DualGen if cfg.get("dual_gen") else M.Generator)(attrs, aa=cfg.get("aa_up", False), noise=cfg.get("noise_inj", False),
                     bound=cfg.get("bound")).cuda().eval()
    Gm.load_state_dict(torch.load(os.path.join(run_dir, "G_ema_%d.pt" % it), map_location="cuda"))
    rng = np.random.RandomState(seed)
    rows = [r for r in G.load_meta() if r["gan_split"] == "train"]
    names = [a for a, _ in attrs]
    conds, labels, devs = [], [], []
    keep_n = {k: (per_class[k] if isinstance(per_class, dict) else per_class) for k in (1, 2, 3)}
    for k in (1, 2, 3):
        cls_rows = [r for r in rows if r["label"] == k]
        for _ in range(keep_n[k] * max(1, FILTER)):
            r = cls_rows[rng.randint(len(cls_rows))]
            dev = devices[rng.randint(len(devices))] if "device" in names else None
            v = {"crackle": int(k in (1, 3)), "wheeze": int(k in (2, 3)),
                 "device": M.DEVICES5.index(dev) if dev else 0, "region": G.REGIONS.index(r["region"]),
                 "source": 0, "one": 0}                 # always generate ICBHI-style samples
            conds.append([v[a] for a in names]); labels.append(k); devs.append(dev)
    data = type("D", (), {"mu": cfg["mu"], "sd": cfg["sd"]})()
    conds, labels = np.array(conds), np.array(labels)
    # generate at the condition strength this checkpoint was TRAINED with (transitional runs: lam < 1 before
    # trans_end); full strength at a transitional checkpoint makes most samples explode (2026-10-22 check)
    ts, te = cfg.get("trans_start", 0) or 0, cfg.get("trans_end", 0) or 0
    lam = 1.0 if te <= 0 or it >= te else max(0.0, (it - ts) / max(1, te - ts))
    print("[gan-synth] %s @%d generated at condition strength %.2f" % (run, it, lam), flush=True)
    x = M.generate(Gm, conds, data, "cuda", seed=seed, lam=lam, trunc=TRUNC).astype(np.float32)   # raw [N,512,128]
    if REJECT_MAX:
        # 2026-10-22: a few % of samples "explode" (values in the hundreds / thousands; real log-mel peaks at ~7),
        # the generator output is unbounded. They are redrawn with new noise until every sample is in range.
        bad, rnd, dropped = x.reshape(len(x), -1).max(1) > REJECT_MAX, 0, 0
        while bad.any() and rnd < 20:
            rnd += 1; dropped += int(bad.sum())
            x[bad] = M.generate(Gm, conds[bad], data, "cuda", seed=seed + 1000 * rnd, lam=lam, trunc=TRUNC).astype(np.float32)
            bad = x.reshape(len(x), -1).max(1) > REJECT_MAX
        assert not bad.any(), "still %d exploded samples after 20 redraws" % bad.sum()
        print("[gan-synth] %s @%d: %d exploded samples (max log-mel > %g) redrawn in %d rounds"
              % (run, it, dropped, REJECT_MAX, rnd), flush=True)
    labels, devs, conf = np.asarray(labels), list(devs), None
    if FILTER or SOFTW:                             # judge = AST fine-tuned on gan-train real cycles (not the final AST)
        import judge as J
        p = J.run(x)[1]
        J._J = None
        torch.cuda.empty_cache()
        own = p[np.arange(len(x)), labels]
        p2 = p.copy(); p2[np.arange(len(x)), labels] = -1
        margin = own - p2.max(1)
        if FILTER:                                  # keep the most class-confident keep_n[k] of each class
            sel = np.concatenate([np.where(labels == k)[0][np.argsort(-margin[labels == k])[:keep_n[k]]] for k in (1, 2, 3)
                                  if keep_n[k]])
            print("[gan-synth] %s @%d: judge filter kept %d of %d (margin mean %.3f -> %.3f, own-class p %.3f -> %.3f)"
                  % (run, it, len(sel), len(x), margin.mean(), margin[sel].mean(), own.mean(), own[sel].mean()), flush=True)
            x, labels, devs, own = x[sel], labels[sel], [devs[i] for i in sel], own[sel]
        conf = own
    if DEV_NOISE:                                   # a test device per fake (uniform), used for noise and as DAT label
        r_ = np.random.RandomState(seed + 7)
        devs = [d if d else TEST_DEVICES[r_.randint(len(TEST_DEVICES))] for d in devs]
    xa = to_ast(x, frames)
    if BG_NOISE:                                    # after tiling to the AST window, so the noise does not repeat
        import bgnoise
        xa = bgnoise.mix(xa, seed, devs if DEV_NOISE else None)
        print("[gan-synth] %s @%d: real background noise mixed into %d windows (bgnoise.py%s)"
              % (run, it, len(xa), ", device-matched" if DEV_NOISE else ""), flush=True)
    if SOFTW:                                       # labels carry the weight: y + 4 * round(10 * confidence), 1..10
        labels = labels + 4 * np.clip(np.round(10 * conf), 1, 10).astype(labels.dtype)
    return xa, labels, devs, it, ev


@torch.no_grad()
def synthesize_dct(run, per_class, seed, frames):
    """Same as synthesize() for a 1D DCT GAN (DCT_GAN/dctgan.py): coefficients -> inverse MDCT -> + real noise from
    its noise bank (unless the run was trained --no-noise) -> Kaldi fbank. Class-only conditioning, so no device
    label (-1, ignored by DAT); the output is bounded, so there are no exploded samples to redraw."""
    sys.path.insert(0, os.path.join(ROOT, "DCT_GAN"))
    import dctgan as DG
    run_dir = os.path.join(DG.RUNS, run)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    if cfg.get("version") == "C":                   # option C: real normal cycle + rule-based events (optionc.py)
        import optionc as OC
        x, labels, devs = OC.ast_set(run_dir, per_class, seed, frames)
        print("[gan-synth] option C %s: %d samples, canvas devices passed to DAT, params %s" % (run, len(x), cfg["params"]),
              flush=True)
        if BG_NOISE:
            import bgnoise
            x = bgnoise.mix(x, seed)
        hist = json.load(open(os.path.join(run_dir, "history.json")))
        return x, labels, devs, 0, hist[-1].get("eval")
    it, ev = pick_checkpoint(run_dir)
    if cfg.get("version") == 2:                     # v2: DDSP harmonic + noise + transient synthesiser (dctgan2.py)
        import dctgan2 as DG2
        Gm, syn, nb, _ = DG2.load_run(run_dir, it)
        labels = np.concatenate([[k] * (per_class[k] if isinstance(per_class, dict) else per_class) for k in (1, 2, 3)])
        conds = np.array([[int(k in (1, 3)), int(k in (2, 3))] for k in labels], dtype=np.int64)
        x = DG2.generate(Gm, syn, nb, conds, "cuda", seed=seed).astype(np.float32)
        print("[gan-synth] DCT GAN v2 %s @%d: %d samples, real noise in the loop: %s" % (run, it, len(x), nb is not None),
              flush=True)
        xa = to_ast(x, frames)
        if BG_NOISE:
            import bgnoise
            xa = bgnoise.mix(xa, seed)
        return xa, labels.astype(np.int64), [None] * len(labels), it, ev
    Gm = DG.Generator1D([tuple(v) for v in cfg["attrs"]], cfg["coef_sd"], cfg["bound"]).cuda().eval()
    Gm.load_state_dict(torch.load(os.path.join(run_dir, "G_ema_%d.pt" % it), map_location="cuda"))
    syn = DG.Synth().cuda()
    nb = None if cfg["no_noise"] else DG.NoiseBank("cuda", tuple(float(v) for v in cfg["noise_gain"].split(",")))
    labels = np.concatenate([[k] * (per_class[k] if isinstance(per_class, dict) else per_class) for k in (1, 2, 3)])
    conds = np.array([[int(k in (1, 3)), int(k in (2, 3))] for k in labels], dtype=np.int64)
    x = DG.generate(Gm, syn, nb, conds, "cuda", seed=seed).astype(np.float32)        # raw log-mel [N,512,128]
    print("[gan-synth] DCT GAN %s @%d: %d samples, real noise in the loop: %s" % (run, it, len(x), nb is not None),
          flush=True)
    xa = to_ast(x, frames)
    if BG_NOISE:
        import bgnoise
        xa = bgnoise.mix(xa, seed)
    return xa, labels.astype(np.int64), [None] * len(labels), it, ev


def synthesize_diff(run, per_class, seed, frames, w):
    """Same as synthesize() for a mel diffusion model (MELDIFF/meldiff.py) at classifier-free guidance scale w. The fake
    pool is generated once per (model, w) and shared by the AST seeds (meldiff.pool); bgnoise is seed-dependent."""
    import meldiff as MD
    assert per_class == 1000, "the shared pool has 1000 per class"
    x, labels, it, devs = MD.pool(run, w)
    print("[gan-synth] diffusion %s @%d w=%g: %d samples (shared pool)%s" % (run, it, w, len(x),
          ", device-conditioned: DAT labels + device-matched noise" if devs else ""), flush=True)
    xa = to_ast(x, frames)
    if BG_NOISE:
        import bgnoise
        xa = bgnoise.mix(xa, seed, devs)
    return xa, labels.astype(np.int64), devs or [None] * len(labels), it, None


def minority_rows(split=None):
    return [r for r in G.load_meta() if r["label"] in (1, 2, 3) and (split is None or r["gan_split"] == split)]


def control_set(mode, per_class, seed, frames, run=None, devices=TEST_DEVICES):
    """-> (AST inputs, labels, device names or None, info dict) for the non-GAN arms."""
    rng = np.random.RandomState(seed)
    if mode == "shuffle":
        x, y, devs, it, ev = synthesize(run, per_class, seed, frames, devices)
        return x, rng.permutation(y), devs, dict(control="shuffle", run=run, checkpoint=it, val_eval=ev)
    if mode == "allnormal":                         # 2026-11-08: the same fakes, every one labelled normal
        x, y, devs, it, ev = synthesize(run, per_class, seed, frames, devices)
        return x, np.zeros_like(y), devs, dict(control="allnormal", run=run, checkpoint=it, val_eval=ev)
    if mode == "bgonly":                            # 2026-11-08: real background noise only (no generator content),
        import bgnoise                              # labelled like the GAN arm (per_class per minority class)
        ys = np.concatenate([[k] * (per_class[k] if isinstance(per_class, dict) else per_class) for k in (1, 2, 3)])
        floor = np.full((len(ys), frames, 128), -15.0, np.float32)      # raw log-mel far below every real cycle
        x = bgnoise.mix(G.ast_normalize(floor).astype(np.float32), seed)
        return x.astype(np.float16), ys, [None] * len(ys), dict(control="bgonly")
    if mode == "noise":
        rows = minority_rows("train")
        xs, ys = [], []
        for k in (1, 2, 3):
            real = np.stack([G.crop_pad(G.get_fbank(r))[0] for r in rows if r["label"] == k])
            mu, sd = real.mean(0), real.std(0)
            xs.append(mu + sd * rng.normal(size=(per_class,) + mu.shape)); ys += [k] * per_class
        return to_ast(np.concatenate(xs).astype(np.float32), frames), np.array(ys), [None] * len(ys), dict(control="noise")
    if mode == "realaug":
        # Kim et al. 2023's Mixed-N without a generator: their DiffWave re-synthesises REAL mel-spectrograms, i.e.
        # slightly varied copies of real cycles. Here: copies of real official-train cycles with mild waveform
        # augmentation (gfdata.augment_wave: gap removal / time-stretch 0.85-1.15 / speed 0.92-1.08), then exactly
        # the sg_exp real-cycle path (tile to 8 s, AST fbank). Device = the source cycle's real stethoscope.
        rows = [r for r in G.load_meta() if r["label"] in (1, 2, 3)]
        n = frames * 160 + 240
        feats, ys, devs = [], [], []
        for k in (1, 2, 3):
            cr = [r for r in rows if r["label"] == k]
            for _ in range(per_class[k] if isinstance(per_class, dict) else per_class):
                r = cr[rng.randint(len(cr))]
                w = G.augment_wave(G.get_wave(r), rng)
                w = np.tile(w, int(np.ceil(n / len(w))))[:n]
                feats.append(G.ast_normalize(G.fbank(w)[:frames]).astype(np.float16)); ys.append(k); devs.append(r["device"])
        return np.stack(feats), np.array(ys), devs, dict(control="realaug", per_class=per_class)
    raise ValueError(mode)


class SynthDS(torch.utils.data.Dataset):
    def __init__(self, x, y, dev_ids):
        self.x, self.y, self.d = torch.from_numpy(x), y, dev_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i].float(), int(self.y[i]), int(self.d[i])


# ============================================================================ training variants
def install_mixup(S):
    """Mixup / SpecMix (mainv2.py S3-C) inside sg_exp's unchanged training loop: sg_exp calls spec_augment(x) once
    per training batch and then F.cross_entropy for the class and the DAT device loss; both are wrapped so the
    batch is mixed after SpecAugment and both losses use lam * CE(y) + (1 - lam) * CE(y[perm])."""
    orig_aug, state = S.spec_augment, {"perm": None, "lam": 1.0}

    def aug_and_mix(x):
        x = orig_aug(x)
        state["perm"], state["lam"] = None, 1.0
        B = x.size(0)
        if B < 2 or random.random() > MIX["p"]:
            return x
        perm = torch.randperm(B, device=x.device)
        lam = float(np.random.beta(MIX["alpha"], MIX["alpha"]))
        xp = x[perm]
        if random.random() < MIX["specmix_prob"]:          # SpecMix: one freq + one time stripe from the partner
            T, Fq = x.shape[-2], x.shape[-1]
            a = 1.0 - math.sqrt(max(1e-6, lam))
            fw, tw = max(1, int(round(a * Fq))), max(1, int(round(a * T)))
            f0, t0 = random.randint(0, Fq - fw), random.randint(0, T - tw)
            xm = x.clone()
            xm[..., f0:f0 + fw] = xp[..., f0:f0 + fw]
            xm[..., t0:t0 + tw, :] = xp[..., t0:t0 + tw, :]
            lam = (1.0 - fw / Fq) * (1.0 - tw / T)
        else:
            xm = lam * x + (1.0 - lam) * xp
        state["perm"], state["lam"] = perm, lam
        return xm

    class MixedF:
        """torch.nn.functional for sg_exp, with cross_entropy mixing the targets of the current training batch."""
        def __getattr__(self, k):
            return getattr(F, k)

        @staticmethod
        def cross_entropy(inp, target, *args, **kw):
            p, lam = state["perm"], state["lam"]
            if p is None or target.size(0) != p.size(0):
                return F.cross_entropy(inp, target, *args, **kw)
            return lam * F.cross_entropy(inp, target, *args, **kw) + (1 - lam) * F.cross_entropy(inp, target[p], *args, **kw)

    S.spec_augment, S.F = aug_and_mix, MixedF()


def install_patchmix(S):
    """--patchmix (user, 2026-10-28): Patch-Mix contrastive learning (Bae et al., Interspeech 2023) added to S1. For every
    training batch a second, patch-mixed view is built at the AST patch-embedding level: a random share (1 - lam) of the
    patch tokens of each sample (same positions) is replaced by those of a partner sample (random permutation), lam ~
    Beta(1, 1); CLS / distillation tokens are kept. A projection head maps both views' (CLS + dist) / 2 embeddings to a
    normalised space and the mixed view is pulled towards its two sources in proportion to the patches it took from each:
        L_pm = -mean[ lam' log softmax(s_i,i / tau) + (1 - lam') log softmax(s_i,perm(i) / tau) ],  tau = 0.06,
    lam' = share of patches kept. The total loss is CE(original view) + PMCL['w'] * L_pm + the unchanged DAT term.
    Test-time forward is identical to S1 (the mixed view only exists in training)."""
    import torch.nn as nn
    state = {"loss": None}
    Base = S.Net

    class PMNet(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            D = self.m.config.hidden_size
            self.pm_proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))

        def head_params(self):
            return super().head_params() + list(self.pm_proj.parameters())

        def encode(self, emb):
            ast = self.m.audio_spectrogram_transformer          # = ASTModel.forward after the embeddings (no mask)
            h = emb
            for layer in ast.layers:
                h = layer(h, None)
                h = h[0] if isinstance(h, tuple) else h
            h = ast.layernorm(h)
            return (h[:, 0] + h[:, 1]) / 2.0

        def forward(self, x):
            emb = self.m.audio_spectrogram_transformer.embeddings(x)
            z = self.encode(emb)
            if self.training and x.size(0) > 1:
                B, Np = x.size(0), emb.size(1) - 2
                perm = torch.randperm(B, device=x.device)
                lam = float(np.random.beta(PMCL["alpha"], PMCL["alpha"]))
                k = int(round((1.0 - lam) * Np))
                idx = torch.rand(B, Np, device=x.device).argsort(1)[:, :k]
                take = torch.zeros(B, Np, dtype=torch.bool, device=x.device).scatter_(1, idx, True)
                em = torch.cat([emb[:, :2], torch.where(take[..., None], emb[perm, 2:], emb[:, 2:])], 1)
                zm = self.encode(em)
                lk = 1.0 - k / Np
                p = F.normalize(self.pm_proj(zm.float()), dim=1)
                q = F.normalize(self.pm_proj(z.float()), dim=1)
                lp = F.log_softmax(p @ q.T / PMCL["tau"], 1)
                ar = torch.arange(B, device=x.device)
                state["loss"] = -(lk * lp[ar, ar] + (1.0 - lk) * lp[ar, perm]).mean()
            return z, self.m.classifier(z)

    prevF = S.F

    class PMF:
        """sg_exp's F with the Patch-Mix loss added to the FIRST cross_entropy after a training forward (the class CE;
        the DAT device CE comes after it and is left unchanged)."""
        def __getattr__(self, k):
            return getattr(prevF, k)

        @staticmethod
        def cross_entropy(*args, **kw):
            ce = prevF.cross_entropy(*args, **kw)
            if state["loss"] is not None:
                ce, state["loss"] = ce + PMCL["w"] * state["loss"], None
            return ce

    S.Net, S.F = PMNet, PMF()
    S.PM_STATE = state


def install_softweight(S):
    """--soft-weight (2026-11-07): synthetic labels arrive as y + 4 * q (q = 1..10 = judge confidence x 10); the class
    cross-entropy decodes them and weights each fake by q / 10 (real cycles keep weight 1). The DAT device CE (targets
    0..3) is left unchanged. Test labels are never encoded."""
    prevF = S.F

    class SWF:
        def __getattr__(self, k):
            return getattr(prevF, k)

        @staticmethod
        def cross_entropy(inp, target, *args, **kw):
            if target.dim() != 1 or int(target.max()) < 4:
                return prevF.cross_entropy(inp, target, *args, **kw)
            w = torch.where(target >= 4, (target // 4).float() / 10.0, torch.ones_like(target, dtype=torch.float))
            ce = F.cross_entropy(inp, target % 4, *args, **dict(kw, reduction="none"))
            return (w * ce).sum() / w.sum()

    S.F = SWF()


def install_seg(S, seg):
    """--seg (user, 2026-10-22 23:00): AST input length in seconds instead of 8 s. 5.135 s = exactly the GAN's 512-frame
    window (Kaldi fbank 25 ms / 10 ms: 511 * 160 + 400 samples). Cycles are still tiled to the window (train and test)."""
    S.SEG_SEC = seg

    class SegDS(S.CycleWaveDS):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.n = int(round(S.SEG_SEC * S.SR))

    S.CycleWaveDS = SegDS


def install_roll(S):
    """--roll (user, 2026-10-22 22:40, "space augmentation"): every TRAINING window (real and synthetic) is circularly
    shifted along time by a random amount, drawn anew each time, so the cycle no longer always starts at frame 0 of the
    8 s tiled window. sg_exp calls spec_augment(x) once per training batch only, so test windows are untouched."""
    orig = S.spec_augment

    def roll_then_aug(x):
        x = torch.stack([torch.roll(v, random.randint(0, v.shape[0] - 1), dims=0) for v in x])
        return orig(x)

    S.spec_augment = roll_then_aug


def install_zeropad(S):
    """Real cycles zero-padded to 8 s (train and test) instead of tiled; long cycles cropped as in sg_exp."""
    tp = S.tp

    class ZeroPadDS(S.CycleWaveDS):
        def __getitem__(self, i):
            c = self.ctx.cycles[self.ids[i]]
            r = self.ctx.recs[c["ridx"]]
            if r["rec_id"] not in self._mm:
                self._mm[r["rec_id"]] = np.load(tp.wave_path(r["rec_id"]), mmap_mode="r")
            w = self._mm[r["rec_id"]]
            s, e = int(c["start"] * S.SR), int(c["end"] * S.SR)
            seg = np.array(w[s:max(e, s + 1600)], dtype=np.float32)
            if len(seg) > self.n:
                o = random.randint(0, len(seg) - self.n) if self.train else 0
                seg = seg[o:o + self.n]
            seg = np.pad(seg, (0, self.n - len(seg)))
            feat = self.fe(seg, sampling_rate=S.SR, return_tensors="np")["input_values"][0]
            return torch.from_numpy(np.ascontiguousarray(feat)).float(), c["label"], S.DEV_ID[r["equip"]]

    S.CycleWaveDS = ZeroPadDS


def install_clean(S, sub="icbhi"):
    """--clean: every real cycle (train AND test) comes from the user's cleaned audio (cleaned_datasets/icbhi:
    band-pass 100-1800 Hz -> spectral denoise -> per-cycle z-score), then exactly the sg_exp path (tiled to 8 s,
    ASTFeatureExtractor). Matched by recording + start time; a missing cycle is an error."""
    import csv
    import soundfile as sf
    root = os.path.join(ROOT, "cleaned_datasets", sub)
    files = {(r["recording"], round(float(r["start_s"]), 3)): os.path.join(root, "wav", r["file"])
             for r in csv.DictReader(open(os.path.join(root, "metadata.csv")))}

    class CleanDS(S.CycleWaveDS):
        def __getitem__(self, i):
            c = self.ctx.cycles[self.ids[i]]
            r = self.ctx.recs[c["ridx"]]
            seg, sr = sf.read(files[(r["rec_id"], round(float(c["start"]), 3))], dtype="float32")
            if len(seg) < 1600:
                seg = np.pad(seg, (0, 1600 - len(seg)))
            if len(seg) < self.n:
                seg = np.tile(seg, int(math.ceil(self.n / len(seg))))
            if len(seg) > self.n:
                o = random.randint(0, len(seg) - self.n) if self.train else 0
                seg = seg[o:o + self.n]
            feat = self.fe(seg, sampling_rate=S.SR, return_tensors="np")["input_values"][0]
            return torch.from_numpy(np.ascontiguousarray(feat)).float(), c["label"], S.DEV_ID[r["equip"]]

    S.CycleWaveDS = CleanDS


def install_lowpass(S):
    """--lowpass2k (2026-11-20): every ICBHI cycle (train AND test) goes through 16 kHz -> 4 kHz -> 16 kHz
    (resample_poly), exactly HF_Lung's own path, so ICBHI and HF have the same bandwidth (nothing above 2 kHz).
    Otherwise identical to sg_exp.CycleWaveDS."""
    from scipy.signal import resample_poly

    class LowpassDS(S.CycleWaveDS):
        def __getitem__(self, i):
            c = self.ctx.cycles[self.ids[i]]
            r = self.ctx.recs[c["ridx"]]
            if r["rec_id"] not in self._mm:
                self._mm[r["rec_id"]] = np.load(S.tp.wave_path(r["rec_id"]), mmap_mode="r")
            w = self._mm[r["rec_id"]]
            s, e = int(c["start"] * S.SR), int(c["end"] * S.SR)
            seg = np.array(w[s:max(e, s + 1600)], dtype=np.float32)
            if self.use_wave_aug and self.train:
                seg = S.wave_aug(seg)
            seg = resample_poly(resample_poly(seg, 1, 4), 4, 1).astype(np.float32)
            if len(seg) < self.n:
                seg = np.tile(seg, int(math.ceil(self.n / len(seg))))
            if len(seg) > self.n:
                o = random.randint(0, len(seg) - self.n) if self.train else 0
                seg = seg[o:o + self.n]
            feat = self.fe(seg, sampling_rate=S.SR, return_tensors="np")["input_values"][0]
            return torch.from_numpy(np.ascontiguousarray(feat)).float(), c["label"], S.DEV_ID[r["equip"]]

    S.CycleWaveDS = LowpassDS


def install_sgscl_mask(S):
    """Plain SG-SCL for --da sgscl: samples without a device (synthetic, label -1) are left out of the device
    contrastive loss instead of silently forming a fifth 'device' group (that is --da sgscl5)."""
    orig = S.sg_supcon

    def masked(h1, h2, dev, stop_grad):
        keep = dev >= 0
        if int(keep.sum()) < 2:
            return h1.sum() * 0.0
        return orig(h1[keep], h2[keep], dev[keep], stop_grad)

    S.sg_supcon = masked


def main():
    global ZEROPAD, FORCE_CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--control", choices=["noise", "shuffle", "realaug", "allnormal", "bgonly"], default=None)
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--mixup", action="store_true")
    ap.add_argument("--patchmix", action="store_true", help="Patch-Mix contrastive learning (Bae et al. 2023), _pmcl")
    ap.add_argument("--zeropad", action="store_true")
    ap.add_argument("--seg", type=float, default=8.0, help="AST input seconds (5.135 = the GAN's 512 frames); _segX")
    ap.add_argument("--roll", action="store_true", help="random circular time shift of every training window (_roll)")
    ap.add_argument("--clean-dir", default="icbhi",
                    help="--clean: subfolder of cleaned_datasets (icbhi = band-pass + denoise + z-score; icbhi_dn = denoise only)")
    ap.add_argument("--clean", action="store_true",
                    help="real train + test cycles from cleaned_datasets/icbhi (user's cleaning); suffix _clean")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="hold out this fraction of TRAIN patients as a validation set; the final val+test logits are saved (_valNN)")
    ap.add_argument("--synth-devices", default=",".join(TEST_DEVICES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--ckpt", type=int, default=0, help="use G_ema_<N>.pt instead of the best-TSTR checkpoint")
    ap.add_argument("--ckpts", default="", help="with --run a,b,c (one specialist GAN per class): their checkpoints")
    ap.add_argument("--reject-max", type=float, default=0,
                    help="redraw generated samples whose raw log-mel exceeds this (exploded samples; e.g. 25); 0 = off")
    ap.add_argument("--bg-noise", action="store_true",
                    help="mix real background noise (quietest train frames, bgnoise.py) into every synthetic window (_bgn)")
    ap.add_argument("--synth-per-class", type=int, default=1000)
    ap.add_argument("--filter", type=int, default=0, help="generate K x the fakes, keep the judge's most class-confident (_fK)")
    ap.add_argument("--trunc", type=float, default=1.0, help="truncation psi of the GAN's z (_tpsi)")
    ap.add_argument("--soft-weight", action="store_true", help="weight each fake's CE by the judge's confidence (_sw)")
    ap.add_argument("--dev-noise", action="store_true", help="fakes get a test device: DAT label + device-matched noise (_dn)")
    ap.add_argument("--dct", action="store_true", help="--run is a 1D DCT GAN in DCT_GAN/runs (DCT_GAN/dctgan.py)")
    ap.add_argument("--diff", action="store_true", help="--run is a mel diffusion model in MELDIFF/runs (meldiff.py)")
    ap.add_argument("--cfg", type=float, default=1.0, help="with --diff: classifier-free guidance scale (_cfgW)")
    ap.add_argument("--hf-real", type=int, default=0,
                    help="+ N HF_Lung cycles as REAL training data (HF_GAN/hfreal.py: teammate's drop rule, ICBHI class "
                         "ratio), HF devices in the DAT head (steth = Litt3200, trunc = new HF-Type-1) (_hfrN)")
    ap.add_argument("--lowpass2k", action="store_true",
                    help="ICBHI train AND test cycles band-limited like HF (16k -> 4k -> 16k) (_lp2k)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--suffix", default="", help="appended to NAME, e.g. _s42 to run extra seeds in parallel")
    ap.add_argument("--freeze-blocks", type=int, default=0,
                    help="freeze the patch embedding + the first N of AST's 12 blocks (0 = full fine-tune, our default)")
    ap.add_argument("--topup", type=int, default=0,
                    help="Kim et al. 'Mixed-N': synthetic only fills each minority class up to N cycles in total "
                         "(official train: crackle 1215, wheeze 501, both 363 -> N=500 adds 0 / 0 / 137)")
    ap.add_argument("--da", choices=["dat", "dat5", "sgscl", "sgscl5"], default="dat",
                    help="domain adaptation: dat = sg_exp S1 (linear device head + GRL); dat5 = S1 with a 5th 'synthetic' class in "
                         "the device head, every synthetic cycle labelled with it (AFT-style real/synthetic alignment, "
                         "Kim et al. 2023); sgscl = sg_exp S2 (device "
                         "SupCon + GRL + stop-grad), synthetic samples left out of it; sgscl5 = S2 with synthetic "
                         "samples as a fifth domain (aligns real and synthetic, like Kim et al. 2023 AFT)")
    a = ap.parse_args()
    assert a.run or a.control in ("noise", "realaug", "bgonly") or a.no_synth, "--run is required for GAN and shuffle arms"
    devices = a.synth_devices.split(",")
    global REJECT_MAX, BG_NOISE, DCT, FILTER, TRUNC, SOFTW, DEV_NOISE
    ZEROPAD, FORCE_CKPT, REJECT_MAX, BG_NOISE, DCT = a.zeropad, a.ckpt, a.reject_max, a.bg_noise, a.dct
    FILTER, TRUNC, SOFTW, DEV_NOISE = a.filter, a.trunc, a.soft_weight, a.dev_noise
    run_name = ("spec_" + "_".join(r.replace("pc_", "") for r in a.run.split(","))) if a.run and "," in a.run else a.run
    name = "base" if a.no_synth else {None: (run_name or "") + ("_at%dk" % (a.ckpt // 1000) if a.ckpt else ""), "noise": "ctrl_noise", "shuffle": "%s_shuf" % a.run,
                                         "realaug": "realaug", "allnormal": "%s_allnorm" % a.run, "bgonly": "ctrl_bgonly"}[a.control]
    if not a.no_synth and devices != TEST_DEVICES:
        name += "_dev-" + "".join(d[0] for d in devices)
    if not a.no_synth and a.synth_per_class != 1000:
        name += "_pc%d" % a.synth_per_class
    name += ("_cfg%g" % a.cfg if a.diff else "") + ("_hfr%d" % a.hf_real if a.hf_real else "") + "_lp2k" * a.lowpass2k
    name += ("_f%d" % a.filter if a.filter else "") + ("_t%g" % a.trunc if a.trunc != 1.0 else "") + "_sw" * a.soft_weight \
        + "_dn" * a.dev_noise
    name += "_bgn" * (a.bg_noise and not a.no_synth) + "_mix" * a.mixup + "_pmcl" * a.patchmix + "_zp" * a.zeropad +("_clean" + ("" if a.clean_dir == "icbhi" else "-" + a.clean_dir) if a.clean else "") + "_roll" * a.roll \
        + ("_seg%g" % a.seg if a.seg != 8.0 else "") + {"dat": "", "dat5": "_dat5", "sgscl": "_sgscl", "sgscl5": "_sgscl5"}[a.da] \
        + ("_top%d" % a.topup if a.topup else "") \
        + ("_fz%d" % a.freeze_blocks if a.freeze_blocks else "") + ("_val%d" % round(100 * a.val_frac) if a.val_frac else "") + a.suffix
    exp = "S1" if a.da in ("dat", "dat5") else "S2"
    # sg_exp.run_experiment names its folder <EXP>_<gan> and appends _<N>pc when synth_per_class != 1000
    run_dir = "%s_%s" % (exp, name) + ("" if a.synth_per_class == 1000 else "_%dpc" % a.synth_per_class)

    import sg_exp as S
    tp = S.tp
    S.OUT_ROOT = _P.sub("ast_runs")
    tp.CACHE_DIR = _P.sub("cache")
    os.makedirs(S.OUT_ROOT, exist_ok=True)
    tp.setup_logging(os.path.join(S.OUT_ROOT, "_setup"))
    if a.seg != 8.0:                                    # first: the clean / zero-pad datasets subclass it
        install_seg(S, a.seg)
    if a.mixup:
        install_mixup(S)
    if a.patchmix:                                      # after --mixup: wraps whatever F is installed
        install_patchmix(S)
    if a.soft_weight:
        install_softweight(S)
    if a.zeropad:
        install_zeropad(S)
    if a.clean:
        install_clean(S, a.clean_dir)
    if a.lowpass2k:
        install_lowpass(S)
    if a.roll:                                          # after --mixup: roll first, then SpecAugment (+ Mixup)
        install_roll(S)
    if a.da == "sgscl":
        install_sgscl_mask(S)
    if a.hf_real:                                       # HF-Type-1 as a new DAT domain (HF steth files = Litt3200)
        S.DEVICES = S.DEVICES + ["HF-Type-1"]
        S.DEV_ID = dict(S.DEV_ID, **{"HF-Type-1": len(S.DEVICES) - 1})
    if a.da == "dat5":                                  # 5-way domain head: 4 stethoscopes + "synthetic"
        S.DEVICES = S.DEVICES + ["SYNTH"]
        S.DEV_ID = dict(S.DEV_ID, SYNTH=len(S.DEVICES) - 1)
    recs, cycles = tp.build_dataset(smoke_stride=1)
    tp.ensure_wave_cache(recs)
    ctx = tp.Ctx(recs, cycles)
    frames = int(tp.frames_for(S.SEG_SEC))
    per_class = a.synth_per_class
    if a.topup:
        real = np.bincount([c["label"] for c in cycles if c["split"] == "train"], minlength=4)
        per_class = {k: max(0, a.topup - int(real[k])) for k in (1, 2, 3)}

    cache = {}

    def add_synth(train_ds, frames_, args, seed):
        if a.no_synth:
            json.dump(dict(control="none"), open(os.path.join(S.OUT_ROOT, run_dir, "gan_checkpoint.json"), "w"))
            return train_ds, 0
        if seed not in cache:
            if a.control:
                x, y, devs, info = control_set(a.control, per_class, seed, frames_, a.run, devices)
            else:
                if "," in a.run:                        # one specialist GAN per class: "crackle_run,wheeze_run,both_run"
                    global FORCE_CKPT
                    runs, cks = a.run.split(","), [int(c) for c in a.ckpts.split(",")]
                    xs_, ys_, devs = [], [], []
                    for k, run, ck in zip((1, 2, 3), runs, cks):
                        n_k = per_class[k] if isinstance(per_class, dict) else per_class
                        if not n_k:
                            continue
                        FORCE_CKPT = ck
                        xk, yk, dk, _, _ = synthesize(run, {1: 0, 2: 0, 3: 0, k: n_k}, seed, frames_, [None])
                        xs_.append(xk); ys_.append(yk); devs += dk
                    FORCE_CKPT = a.ckpt
                    x, y, it, ev = np.concatenate(xs_), np.concatenate(ys_), 0, None
                    info = dict(runs=runs, ckpts=cks, per_class=per_class)
                else:
                    x, y, devs, it, ev = (synthesize_diff(a.run, per_class, seed, frames_, a.cfg) if a.diff else
                                          synthesize_dct(a.run, per_class, seed, frames_) if a.dct else
                                          synthesize(a.run, per_class, seed, frames_, devices))
                    info = dict(run=a.run, checkpoint=it, val_eval=ev, per_class=per_class, dct=a.dct)
            if a.da == "dat5":
                devs = ["SYNTH"] * len(y)
            d = np.array([S.DEV_ID[v] if v else -1 for v in devs])
            cache[seed] = SynthDS(x, y, d)
            if a.dct and a.run and "," not in a.run:    # option C with held-out canvases: drop them from the real set
                ccfg = json.load(open(os.path.join(ROOT, "DCT_GAN", "runs", a.run, "config.json")))
                if ccfg.get("canvas_holdout"):
                    keys = {(r["rec_id"], round(r["start"], 2)) for r in G.load_meta()
                            if r["gan_split"] == "val" and r["label"] == 0}
                    n0 = len(train_ds.ids)
                    train_ds.ids = [i for i in train_ds.ids if (ctx.recs[ctx.cycles[i]["ridx"]]["rec_id"],
                                                                round(ctx.cycles[i]["start"], 2)) not in keys]
                    tp.log.info("[gan-synth] canvas hold-out: %d real normal cycles removed from the AST training set",
                                n0 - len(train_ds.ids))
            tp.log.info("[gan-synth] %s %s: +%d synthetic, per class %s, device labels %s", name, info, len(y),
                        np.bincount(y, minlength=4).tolist(),
                        {k: int((d == v).sum()) for k, v in list(S.DEV_ID.items()) + [("none", -1)]})
            json.dump(info, open(os.path.join(S.OUT_ROOT, run_dir, "gan_checkpoint.json"), "w"), indent=1)
        return torch.utils.data.ConcatDataset([train_ds, cache[seed]]), len(cache[seed])

    S.add_gan_synth = add_synth
    if a.hf_real:
        import hfreal

        def add_with_hf(train_ds, frames_, args_, seed):
            ds, n = add_synth(train_ds, frames_, args_, seed)
            hx, hy, hd = hfreal.load(a.hf_real, frames_)
            tp.log.info("[hf-real] +%d HF_Lung cycles as real training data, per class %s, devices %s", len(hy),
                        np.bincount(hy, minlength=4).tolist(), {d: hd.count(d) for d in set(hd)})
            return torch.utils.data.ConcatDataset([ds, SynthDS(hx, hy, np.array([S.DEV_ID[d] for d in hd]))]), n

        S.add_gan_synth = add_with_hf
    args = argparse.Namespace(epochs=a.epochs, seeds=a.seeds, freeze_blocks=a.freeze_blocks, num_workers=a.num_workers,
                              gan=name, synth_per_class=a.synth_per_class, smoke=False, val_frac=a.val_frac,
                              no_register=True)
    os.makedirs(os.path.join(S.OUT_ROOT, run_dir), exist_ok=True)
    tp.log.info("ast_gan variant %s | exp=%s da=%s mixup=%s zeropad=%s synth_devices=%s", name, exp, a.da, a.mixup,
                a.zeropad, devices)
    S.run_experiment(exp, ctx, args, frames)
    from register_result import register
    d = os.path.join(_P.OUT, "ast_runs", run_dir)
    info = json.load(open(os.path.join(d, "gan_checkpoint.json")))
    seeds = "1 seed: %d" % a.seeds[0] if len(a.seeds) == 1 else "%d seeds" % len(a.seeds)
    if a.no_synth:
        what = "no synthetic data"
    else:
        what = {None: ("one specialist GAN per class %s @%s" % (a.run, a.ckpts)) if "," in (a.run or "") else
                ("mel diffusion %s @%dk CFG w=%g" % (a.run, info.get("checkpoint", 0) // 1000, a.cfg) if a.diff else "new %sGAN %s @%dk" % ("1D DCT " if a.dct else "", a.run, info.get("checkpoint", 0) // 1000)),
                "noise": "CONTROL class-mean Gaussian noise samples",
                "shuffle": "CONTROL new GAN %s @%dk with shuffled minority labels" % (a.run, info.get("checkpoint", 0) // 1000),
                "realaug": "Kim-style re-synthesis without a generator: augmented copies of real cycles",
                "allnormal": "CONTROL GAN %s fakes all labelled normal" % a.run,
                "bgonly": "CONTROL real background noise only, minority labels"}[a.control]
        amount = "top-up to %d/class (+%s)" % (a.topup, "/".join(str(per_class[k]) for k in (1, 2, 3))) if a.topup \
            else "%d/class" % a.synth_per_class
        what += " (%s%s)" % (amount, "" if devices == TEST_DEVICES else ", devices " + "/".join(devices))
    extras = [x for x, on in (("judge filter: %dx generated, most class-confident kept" % a.filter, a.filter),
                              ("truncation psi %g" % a.trunc, a.trunc != 1.0),
                              ("fakes weighted by judge confidence", a.soft_weight),
                              ("test device per fake: DAT label + device-matched noise", a.dev_noise),
                              ("real background noise on synthetic (bgnoise.py, Round 15 method)", a.bg_noise and not a.no_synth),
                              ("Mixup/SpecMix", a.mixup),
                              ("Patch-Mix contrastive learning (Bae et al. 2023, weight %g, tau %g)" % (PMCL["w"], PMCL["tau"]), a.patchmix),("zero-pad 8 s", a.zeropad),
                              ("CLEANED audio train+test (%s)" % ("band-pass + spectral denoise + z-score" if a.clean_dir == "icbhi" else
                                                  "spectral denoise only, full band, no z-score" if a.clean_dir == "icbhi_dn" else a.clean_dir), a.clean),
                              ("random circular time shift of training windows", a.roll),
                              ("+ %d HF_Lung cycles as real training data (rhonchi/stridor dropped, ICBHI class ratio, HF devices in DAT)" % a.hf_real, a.hf_real),
                              ("ICBHI train+test band-limited to 2 kHz like HF (16k->4k->16k)", a.lowpass2k),
                              ("AST input %g s (%d frames)" % (a.seg, int(tp.frames_for(a.seg))), a.seg != 8.0),
                              ("first %d AST blocks frozen" % a.freeze_blocks, a.freeze_blocks),
                              ("trained on %d%% of the train patients (patient-wise %d%% validation hold-out, logits saved)" % (100 - round(100 * a.val_frac), round(100 * a.val_frac)), a.val_frac)) if on]
    cmd = "GAN_FINETUNING/ast_gan.py" + (" --no-synth" if a.no_synth else "") \
        + (" --control %s" % a.control if a.control else "") + (" --run %s" % a.run if a.run else "") \
        + (" --synth-devices %s" % a.synth_devices if devices != TEST_DEVICES else "") \
        + (" --synth-per-class %d" % a.synth_per_class if a.synth_per_class != 1000 else "") \
        + " --mixup" * a.mixup + " --patchmix" * a.patchmix + " --zeropad" * a.zeropad + (" --clean" + ("" if a.clean_dir == "icbhi" else " --clean-dir " + a.clean_dir) if a.clean else "") + " --roll" * a.roll \
        + (" --seg %g" % a.seg if a.seg != 8.0 else "") + (" --da %s" % a.da if a.da != "dat" else "") \
        + (" --freeze-blocks %d" % a.freeze_blocks if a.freeze_blocks else "") \
        + (" --val-frac %g" % a.val_frac if a.val_frac else "") \
        + (" --topup %d" % a.topup if a.topup else "") \
        + (" --suffix %s" % a.suffix if a.suffix else "") + (" --ckpt %d" % a.ckpt if a.ckpt else "") \
        + (" --ckpts %s" % a.ckpts if a.ckpts else "") + (" --filter %d" % a.filter if a.filter else "") \
        + (" --trunc %g" % a.trunc if a.trunc != 1.0 else "") + " --soft-weight" * a.soft_weight + " --dev-noise" * a.dev_noise \
        + (" --reject-max %g" % a.reject_max if a.reject_max else "") + " --bg-noise" * a.bg_noise + " --dct" * a.dct + (" --diff --cfg %g" % a.cfg if a.diff else "") + (" --hf-real %d" % a.hf_real if a.hf_real else "") + " --lowpass2k" * a.lowpass2k
    base = {"dat": "DAT · device-adversarial, plain CE",
            "dat5": "DAT · 4 devices + 'synthetic' as 5th domain (AFT-style alignment), plain CE",
            "sgscl": "SG-SCL · device SupCon + GRL + stop-grad, plain CE (synthetic left out of the SupCon)",
            "sgscl5": "SG-SCL · device SupCon + GRL + stop-grad, plain CE (synthetic as a 5th domain)"}[a.da]
    register("GF_%s_%s" % (exp, name), "AST stethoscope-DA + GAN_FINETUNING",
             "%s + %s%s [%s]" % (base, what, "".join(" + " + e for e in extras), seeds),
             cmd + " --seeds %s" % " ".join(map(str, a.seeds)), d + "/", d + "/final_summary.json")
    tp.log.info("registered GF_%s_%s in results_summary/all_runs.csv", exp, name)


if __name__ == "__main__":
    main()
