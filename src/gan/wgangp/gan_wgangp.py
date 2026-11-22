"""
WGAN-GP (Gulrajani et al. 2017).
Same Wasserstein critic as WGAN, but the Lipschitz constraint is a soft penalty
lambda * (||grad_x D(x_hat)|| - 1)^2 on random interpolations x_hat between real and fake.
Paper settings: Adam lr 1e-4, betas (0, 0.9), 5 critic steps, lambda 10, no batch-norm / no SN in critic.
"""
import torch
from gan_common import Method, run


class WGANGP(Method):
    name = "wgangp"
    sn, projection, aux = False, True, False
    n_critic = 5
    lr_g = lr_d = 1e-4
    betas = (0.0, 0.9)
    lam = 10.0

    def d_loss(self, D, xr, xf, y):
        ar, af = self.augment(xr), self.augment(xf)
        dr, _ = D(ar, y)
        df, _ = D(af, y)
        eps = torch.rand(xr.size(0), 1, 1, 1, device=xr.device)
        xh = (eps * ar + (1 - eps) * af).detach().requires_grad_(True)
        dh, _ = D(xh, y)
        g, = torch.autograd.grad(dh.sum(), xh, create_graph=True)
        gp = ((g.flatten(1).norm(dim=1) - 1) ** 2).mean()
        wdist = dr.mean() - df.mean()
        return -wdist + self.lam * gp, dict(wdist=float(wdist), gp=float(gp))

    def g_loss(self, D, xf, y):
        return -D(self.augment(xf), y)[0].mean()


if __name__ == "__main__":
    run(WGANGP)
