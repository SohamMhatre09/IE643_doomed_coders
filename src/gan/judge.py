"""
judge.py -- the frozen "judge" for the evaluation suite (Step 2).

AST (AudioSet-pretrained, local ast_model/) fine-tuned on gan_split=train real cycles only (4 classes,
weighted CE, 512-frame window).  The held-out val patients are never seen, so its accuracy on val is the
real-data reference for class fidelity.  Provides:
  * predict(x)  -> class probabilities          (class fidelity of synthetic samples)
  * embed(x)    -> 768-d (CLS+dist)/2 embedding  (FD, precision/recall, probe, nearest neighbours)
x is RAW log-mel [N,512,128] (the gfdata / GAN scale); AST normalisation is applied inside.

Train once:  $PY GAN_FINETUNING/judge.py      -> GAN_FINETUNING/eval/judge.pt + judge_report.md
"""
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import gfdata as G                                  # noqa: E402

import na_paths as _P
EVAL_DIR = os.path.join(_P.OUT, "gan_eval")
JUDGE_PT = os.path.join(EVAL_DIR, "judge.pt")
EPOCHS, BATCH, LR_BB, LR_HEAD, SEED = 8, 16, 5e-5, 5e-4, 0


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        import train_optimal_pipeline as tp
        tp.FREEZE_BLOCKS = 0
        self.m = tp.build_model(G.WIN)
        for p in self.m.parameters():
            p.requires_grad = True

    def forward(self, x):                                # x: raw log-mel [B,T,F]
        h = self.m.audio_spectrogram_transformer(input_values=G.ast_normalize(x)).last_hidden_state
        z = (h[:, 0] + h[:, 1]) / 2.0
        return z, self.m.classifier(z)


def real_set(split, labels=(0, 1, 2, 3), rng=None):
    """All real cycles of one gan_split as (x [N,512,128] float32, y)."""
    rows = [r for r in G.load_meta() if r["gan_split"] == split and r["label"] in labels]
    x = np.stack([G.crop_pad(G.get_fbank(r), rng=rng)[0] for r in rows])
    return x, np.array([r["label"] for r in rows]), rows


_J = None


def load_judge(device="cuda"):
    global _J
    if _J is None:
        _J = Judge()
        _J.load_state_dict(torch.load(JUDGE_PT, map_location="cpu"))
        _J = _J.to(device).eval()
    return _J


@torch.no_grad()
def run(x, device="cuda", bs=32):
    """-> (embeddings [N,768], probs [N,4]) for raw log-mel x [N,512,128]."""
    j = load_judge(device)
    zs, ps = [], []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(np.asarray(x[i:i + bs], dtype=np.float32), device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            z, lg = j(xb)
        zs.append(z.float().cpu().numpy())
        ps.append(lg.float().softmax(-1).cpu().numpy())
    return np.concatenate(zs), np.concatenate(ps)


def train():
    import train_optimal_pipeline as tp
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    os.makedirs(EVAL_DIR, exist_ok=True)
    rows = [r for r in G.load_meta() if r["gan_split"] == "train"]
    y = np.array([r["label"] for r in rows])
    xv, yv, _ = real_set("val")
    dev = "cuda"
    net = Judge().to(dev)
    head = list(net.m.classifier.parameters())
    hid = {id(p) for p in head}
    bb = [p for p in net.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{"params": bb, "lr": LR_BB}, {"params": head, "lr": LR_HEAD}], weight_decay=0.01)
    steps = EPOCHS * (len(rows) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [LR_BB, LR_HEAD], total_steps=steps, pct_start=0.1)
    cnt = np.bincount(y, minlength=4)
    cw = torch.tensor(cnt.sum() / (4 * cnt), dtype=torch.float32, device=dev)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(len(rows))
        tot = 0.0
        for b in range(len(rows) // BATCH):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            xb = np.stack([G.crop_pad(G.get_fbank(rows[i]), rng=rng)[0] for i in idx])
            xb = torch.as_tensor(xb, device=dev)
            for k in range(len(xb)):                    # light SpecAugment
                f0, fw = rng.randint(0, 100), rng.randint(0, 24)
                t0, tw = rng.randint(0, 400), rng.randint(0, 96)
                xb[k, :, f0:f0 + fw] = xb[k].min()
                xb[k, t0:t0 + tw] = xb[k].min()
            with torch.autocast("cuda", dtype=torch.float16):
                _, lg = net(xb)
            loss = F.cross_entropy(lg.float(), torch.as_tensor(y[idx], device=dev), weight=cw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        net.eval()
        global _J
        _J = net
        _, p = run(xv)
        m, _ = tp.compute_metrics(yv, p.argmax(1))
        print("epoch %d loss %.4f | val AS %.2f SE %.2f SP %.2f" % (ep + 1, tot / (len(rows) // BATCH), m["AS"], m["SE"], m["SP"]), flush=True)
    torch.save(net.state_dict(), JUDGE_PT)
    _, p = run(xv)
    pred = p.argmax(1)
    rec = [100 * np.mean(pred[yv == k] == k) for k in range(4)]
    m, cm = tp.compute_metrics(yv, pred)
    with open(os.path.join(EVAL_DIR, "judge_report.md"), "w") as f:
        f.write("# Judge (AST fine-tuned on gan_split=train real, %d epochs)\n\n" % EPOCHS)
        f.write("Real val patients (never seen): AS %.2f SE %.2f SP %.2f\n\n" % (m["AS"], m["SE"], m["SP"]))
        f.write("| class | n | recall |\n|---|---|---|\n")
        for k in range(4):
            f.write("| %s | %d | %.1f |\n" % (G.CLASS_NAMES[k], (yv == k).sum(), rec[k]))
        f.write("\nconfusion (rows true, cols pred):\n\n```\n%s\n```\n" % np.array2string(np.asarray(cm)))
    print(open(os.path.join(EVAL_DIR, "judge_report.md")).read())


if __name__ == "__main__":
    train()
