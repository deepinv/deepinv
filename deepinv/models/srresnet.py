"""
Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from .base import Reconstructor

if TYPE_CHECKING:
    from deepinv.physics import Physics


class SRResNet(Reconstructor):
    r"""
    SRResNet super-resolution network.

    Convolutional super-resolution architecture introduced in :footcite:t:`ledig2017photo`
    as the generator of SRGAN. The network applies a feature-extraction conv, a stack of
    residual blocks (Conv-Norm-Activation-Conv-Norm with an additive skip), a long skip
    connection from the feature-extraction output, and finally a sequence of
    :class:`torch.nn.PixelShuffle`-based upsampling stages followed by a wide output
    convolution.

    The total upsampling factor is ``upscale`` and must be a power of two; the network
    contains :math:`\log_2(\text{upscale})` upsampling stages, each doubling the spatial
    resolution.

    The model is registered as a :class:`Reconstructor <deepinv.models.Reconstructor>`:
    its ``forward`` takes a low-resolution measurement ``y`` and returns a
    high-resolution estimate. The ``physics`` argument is accepted for API compatibility
    but is not used by this network.

    .. note::
        The defaults correspond to the network configuration in :footcite:t:`ledig2017photo`.

    :param int num_blocks: number of residual blocks in the trunk. Default: 16
    :param int im_c: number of image channels (used for both input and output). Default: 3
    :param int feats: number of feature channels in the trunk. Default: 64
    :param int upscale: upsampling factor. Must be a power of two. Default: 4
    :param type[torch.nn.Module] actv: activation layer class, instantiated with no
        arguments. Default: :class:`torch.nn.ReLU`.
    :param str norm: normalization layer, can be one of ('instance_norm', 'batch_norm', 'layer_norm', None).
    :param int final_kernel_size: kernel size of the final output convolution. Must be odd. Default: 9.
    :param bool final_relu: enforce non-negativity of output by performing a relu after final conv. Default: False
    """

    def __init__(
        self,
        num_blocks: int = 16,
        im_c: int = 3,
        feats: int = 64,
        upscale: int = 4,
        actv: type[nn.Module] = nn.PReLU,
        norm: Optional[str] = "batch_norm",
        final_kernel_size: int = 9,
        final_relu: bool = False,
    ):
        super().__init__()
        if upscale < 1 or (upscale & (upscale - 1)) != 0:
            raise ValueError(
                f"upscale must be a power of two (e.g. 2, 4, 8), got {upscale}"
            )
        if final_kernel_size % 2 == 0:
            raise ValueError(f"final_kernel_size must be odd, got {final_kernel_size}")
        if norm not in ("batch_norm", "instance_norm", "layer_norm", None):
            raise ValueError(
                f"norm must be one of (batch_norm, instance_norm, layer_norm, None), got {norm}"
            )
        norm = {
            "batch_norm": nn.BatchNorm2d,
            "instance_norm": nn.InstanceNorm2d,
            "layer_norm": _LayerNorm2d,
            None: nn.Identity,
        }[norm]
        self.fe = nn.Sequential(nn.Conv2d(im_c, feats, 9, 1, 4), actv())
        self.blocks = nn.Sequential(
            *[_Block(feats, actv, norm) for _ in range(num_blocks)]
        )
        self.block = nn.Sequential(nn.Conv2d(feats, feats, 3, 1, 1), norm(feats))

        self.upsampling = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(feats, feats * 4, 3, 1, 1), nn.PixelShuffle(2), actv()
                )
                for _ in range(int(math.log2(upscale)))
            ]
        )
        p = (final_kernel_size - 1) // 2
        self.final_conv = nn.Sequential(
            *(
                [nn.Conv2d(feats, im_c, final_kernel_size, 1, p)]
                + ([nn.ReLU()] if final_relu else [])
            )
        )

    def forward(
        self, y: torch.Tensor, physics: Physics | None = None, **kwargs
    ) -> torch.Tensor:
        r"""
        Apply the super-resolution network to a low-resolution input.

        :param torch.Tensor y: low-resolution input image, of shape ``(B, im_c, H, W)``.
        :param deepinv.physics.Physics physics: forward operator (not used).

        :returns: (:class:`torch.Tensor`) high-resolution estimate, of shape
            ``(B, im_c, upscale * H, upscale * W)``.
        """
        lf = self.fe(y)
        x = self.blocks(lf)
        x = lf + self.block(x)
        x = self.upsampling(x)
        return self.final_conv(x)


class _Block(nn.Module):
    def __init__(self, feats: int, actv: type[nn.Module], norm: type[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(feats, feats, 3, 1, 1),
            norm(feats),
            actv(),
            nn.Conv2d(feats, feats, 3, 1, 1),
            norm(feats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layers(x)
        return x


class _LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W), normalize over C
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
