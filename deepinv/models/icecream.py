from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from deepinv.models.base import Reconstructor
from deepinv.models.unet import UNet


class Icecream(Reconstructor):
    r"""
    Icecream reconstruction network for cryo-electron tomography.

    Icecream :footcite:p:`kishore2026icecream` is a self-supervised method jointly denoising tomograms and correcting their missing wedge.
    A 3D network :math:`f` is trained with :class:`deepinv.loss.IcecreamLoss` on sub-tomograms of two tomograms reconstructed from independent halves of the dose (e.g. even and odd frames).
    See the `original code <https://github.com/swing-research/Icecream>`_.

    The network is applied to each channel independently: a pair :math:`y=(y_1, y_2)` stacked along channels gives :math:`(f(y_1), f(y_2))`.
    Use :meth:`reconstruct` for full tomograms.

    :param torch.nn.Module backbone: 3D network mapping volumes of shape `(B, 1, D, H, W)` to the same shape. Default: bias-free 3D :class:`deepinv.models.UNet` with `channels_per_scale` feature maps, as in the paper.
    :param list[int] channels_per_scale: feature maps of the default backbone. Default: `[64, 128, 256, 512]`.
    :param torch.device, str device: device of the model.

    |sep|

    :Example:

        >>> import torch
        >>> import deepinv as dinv
        >>> model = dinv.models.Icecream(channels_per_scale=[4, 8])
        >>> physics = dinv.physics.MissingWedge(img_size=(16, 16, 16), angle_range=(-60, 60))
        >>> y = torch.randn(1, 2, 16, 16, 16)  # pair of sub-tomograms
        >>> model(y, physics).shape
        torch.Size([1, 2, 16, 16, 16])
    """

    def __init__(
        self,
        backbone: torch.nn.Module | None = None,
        channels_per_scale: list[int] | None = None,
        device: torch.device | str = "cpu",
    ):
        super().__init__(device=device)
        if backbone is None:
            backbone = UNet(
                in_channels=1,
                out_channels=1,
                residual=False,
                bias=False,
                batch_norm=False,
                dim=3,
                channels_per_scale=(
                    [64, 128, 256, 512]
                    if channels_per_scale is None
                    else channels_per_scale
                ),
            )
        self.backbone = backbone
        self.to(device)

    def forward(self, y: Tensor, physics=None, **kwargs) -> Tensor:
        r"""
        Applies the network to each channel of `y` independently.

        :param torch.Tensor y: (sub-)tomograms of shape `(B, C, D, H, W)`.
        :param deepinv.physics.MissingWedge physics: missing-wedge operator (unused).
        :return: (:class:`torch.Tensor`) estimates of shape `(B, C, D, H, W)`.
        """
        B, C = y.shape[:2]
        out = self.backbone(y.reshape(B * C, 1, *y.shape[2:]))
        return out.reshape(B, C, *y.shape[2:])

    @torch.no_grad()
    def reconstruct(
        self,
        y: Tensor,
        physics,
        stride: int | None = None,
        batch_size: int = 4,
        double_pass: bool = True,
    ) -> Tensor:
        r"""
        Reconstructs full tomograms from overlapping sub-tomograms.

        Each sub-tomogram :math:`v` is processed as :math:`f(\forw{f(v)})` if `double_pass=True` (as in the original code), or :math:`f(v)` otherwise.
        The central parts (half the size along each dimension) are averaged, as well as the channels (e.g. the two half tomograms).

        :param torch.Tensor y: tomograms of shape `(B, C, D, H, W)`.
        :param deepinv.physics.MissingWedge physics: missing-wedge operator whose size defines the sub-tomograms.
        :param int stride: distance between sub-tomograms. Default: half their size.
        :param int batch_size: number of sub-tomograms processed simultaneously.
        :param bool double_pass: if `True`, applies the network twice, re-imposing the missing wedge in between.
        :return: (:class:`torch.Tensor`) tomograms of shape `(B, 1, D, H, W)`.
        """
        size = physics.volume_size
        if len(set(size)) != 1:
            raise ValueError("Icecream expects cubic sub-tomograms.")
        n = size[0]
        stride = n // 2 if stride is None else int(stride)
        pad = n // 4
        if stride > n - 2 * pad:
            raise ValueError(
                f"The stride must be at most {n - 2 * pad} so that the central parts of the sub-tomograms cover the volume."
            )
        window = torch.zeros(size, device=y.device)
        window[pad : n - pad, pad : n - pad, pad : n - pad] = 1

        was_training = self.training
        self.eval()
        B, C, D, H, W = y.shape
        out = torch.zeros(B, C, D, H, W, device=y.device)
        for b in range(B):
            for c in range(C):
                out[b, c] = self._reconstruct_volume(
                    y[b, c], physics, n, stride, pad, window, batch_size, double_pass
                )
        self.train(was_training)
        return out.mean(dim=1, keepdim=True)

    def _reconstruct_volume(
        self, vol, physics, n, stride, pad, window, batch_size, double_pass
    ):
        shape = vol.shape
        # pad so that the borders fall in central parts and the sub-tomograms tile the volume
        extra = []
        for s in shape:
            total = s + 2 * pad
            extra.append(pad + (-(total - n) % stride) if total > n else n - s - pad)
        vol = F.pad(vol, (pad, extra[2], pad, extra[1], pad, extra[0]))
        starts = [range(0, s - n + 1, stride) for s in vol.shape]
        corners = [(i, j, k) for i in starts[0] for j in starts[1] for k in starts[2]]

        acc = torch.zeros_like(vol)
        weight = torch.zeros_like(vol)
        for chunk in range(0, len(corners), batch_size):
            batch = corners[chunk : chunk + batch_size]
            crops = torch.stack(
                [vol[i : i + n, j : j + n, k : k + n] for i, j, k in batch]
            )[:, None]
            est = self(crops, physics)
            if double_pass:
                est = self(physics.A(est), physics)
            for (i, j, k), e in zip(batch, est[:, 0], strict=True):
                acc[i : i + n, j : j + n, k : k + n] += e * window
                weight[i : i + n, j : j + n, k : k + n] += window
        acc = acc / weight.clamp(min=1)
        return acc[pad : pad + shape[0], pad : pad + shape[1], pad : pad + shape[2]]
