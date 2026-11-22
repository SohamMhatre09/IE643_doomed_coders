"""
azimuth.py -- azimuthal integration of the 2D Fourier power spectrum (Durall et al., CVPR 2020, "Watch Your
Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions").

For an image x [H, W]: P = |FFT2(x)|^2, centred; the 1D spectrum is the mean of log P over rings of equal radius.
Our "images" are log-mel windows [128 mel, 512 frames], not square, so the radius uses normalised frequencies
(each axis scaled to [-0.5, 0.5] cycles/sample): r = sqrt(fy^2 + fx^2) in [0, 0.707], cut into N_BINS rings up to
r = 0.5 (the inscribed circle, as in Durall's code). Up-convolution artefacts show up at the high-frequency end.

Durall's spectral loss: each sample's 1D spectrum is min-max normalised to [0, 1]; the generator gets
BCE(fake spectrum, mean real spectrum). Everything here is differentiable (torch), so the same code measures and trains.
Also: 1D spectra along each axis separately (time / frequency), since the axes mean different things.
"""
import torch
import torch.nn.functional as F

N_BINS = 64


class Azimuth(torch.nn.Module):
    def __init__(self, H=128, W=512, n_bins=N_BINS):
        super().__init__()
        fy = torch.fft.fftshift(torch.fft.fftfreq(H))
        fx = torch.fft.fftshift(torch.fft.fftfreq(W))
        r = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
        idx = (r / 0.5 * n_bins).long()
        keep = idx < n_bins
        M = torch.zeros(H * W, n_bins)
        M[torch.arange(H * W)[keep.flatten()], idx.flatten()[keep.flatten()]] = 1.0
        M = M / M.sum(0, keepdim=True)
        self.register_buffer("M", M)
        self.n_bins = n_bins

    def forward(self, x):
        """x [B, H, W] (or [B,1,H,W]) -> 1D log-power spectrum [B, n_bins] (ring means of log |FFT2|^2)."""
        if x.dim() == 4:
            x = x[:, 0]
        x = x - x.mean((1, 2), keepdim=True)
        P = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1)).abs().pow(2)
        return torch.log(P + 1e-8).flatten(1) @ self.M

    @staticmethod
    def axis_spectra(x):
        """-> (time spectrum [B, W//2+1], frequency spectrum [B, H//2+1]): mean log power of 1D FFTs along each axis."""
        if x.dim() == 4:
            x = x[:, 0]
        x = x.float() - x.float().mean((1, 2), keepdim=True)
        t = torch.log(torch.fft.rfft(x, dim=2).abs().pow(2) + 1e-8).mean(1)
        f = torch.log(torch.fft.rfft(x, dim=1).abs().pow(2) + 1e-8).mean(2)
        return t, f


def minmax(s):
    lo, hi = s.min(1, keepdim=True).values, s.max(1, keepdim=True).values
    return (s - lo) / (hi - lo + 1e-8)


def spectral_loss(az, fake, real_ref):
    """Durall's spectral loss: BCE between each fake's normalised 1D spectrum and the (normalised) real reference."""
    p = minmax(az(fake)).clamp(1e-4, 1 - 1e-4)
    return F.binary_cross_entropy(p, real_ref.expand_as(p))
