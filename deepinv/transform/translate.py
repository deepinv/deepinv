from deepinv.transform.base import Transform

import torch


class Translate(Transform):
    r"""
    Continuous Fourier-based 2D translations.

    This implements the translation of a 2D vector :math:`x \in \mathbb R^{H \times W}` by a displacement :math:`(\Delta_H, \Delta_W) \in \mathbb R^2` as the 2D vector :math:`y \in \mathbb R^{H \times W}` defined by

    .. math::

        Y_{k,l} = e^{-2i\pi \left( \frac{k}{H}\Delta_{H} + \frac{l}{W}\Delta_{W} \right)} X_{k,l},


    where :math:`X` and :math:`Y` are the discrete Fourier transform of :math:`x` and :math:`y` respectively. It expects an input of size :math:`(N, C, H, W)` and computes the translations independently on the batch and channel dimensions.

    For efficient whole pixel translations, see :class:`deepinv.transform.Shift`.

    See :class:`deepinv.transform.Transform` for further details and examples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate translation parameters.

        :param torch.Tensor x: input image
        :return dict: keyword args of translation parameters
        """
        N = x.shape[0] * self.n_trans
        H, W = x.shape[-2:]
        displacement_h = H * torch.rand((N,), device=x.device, generator=self.rng)
        displacement_w = W * torch.rand((N,), device=x.device, generator=self.rng)
        return {"delta_h": displacement_h, "delta_w": displacement_w}

    def _transform(
        self,
        x: torch.Tensor,
        delta_h: torch.Tensor,
        delta_w: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Translate image given translation parameters.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor delta_h: vertical displacement
        :param torch.Tensor delta_w: horizontal displacement
        :return: torch.Tensor: transformed image.
        """
        H, W = x.shape[-2:]
        s = x.shape[-2:]
        x = torch.fft.rfft2(x)
        h_freqs, w_freqs = torch.meshgrid(
            torch.fft.fftfreq(H, device=x.device),
            torch.fft.rfftfreq(W, device=x.device),
            indexing="ij",
        )
        x *= torch.exp(-2j * torch.pi * (h_freqs * delta_h + w_freqs * delta_w))
        return torch.fft.irfft2(x, s=s)
