from deepinv.transform import Transform

import torch


class Translate(Transform):
    def _get_params(self, x):
        N = x.shape[0] * self.n_trans
        H, W = x.shape[-2:]
        displacement_h = torch.rand((N,), device=x.device) * H
        displacement_w = torch.rand((N,), device=x.device) * W
        return {"displacement": (displacement_h, displacement_w)}

    def _transform(self, x, displacement, **kwargs):
        H, W = x.shape[-2:]
        delta_h, delta_w = displacement
        s = x.shape[-2:]
        x = torch.fft.rfft2(x)
        h_freqs, w_freqs = torch.meshgrid(
            torch.fft.fftfreq(H, device=x.device),
            torch.fft.rfftfreq(W, device=x.device),
            indexing="ij",
        )
        x *= torch.exp(-2j * torch.pi * (h_freqs * delta_h + w_freqs * delta_w))
        return torch.fft.irfft2(x, s=s)
