# The decoder is adapted from Transfer CLIP, https://github.com/alwaysuu/CLIPDenoising
# Copyright (c) 2024 Jun Cheng. Released under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Denoiser
from .utils import get_weights_url, load_state_dict_from_url, test_pad


class FrozenCLIPDenoiser(Denoiser):
    r"""
    Frozen CLIP prior denoiser.

    Denoiser of :footcite:t:`diazdelgado2026frozen`, :math:`\denoiser{x}{\sigma} = G(E(x), x)`, where :math:`E` is the frozen
    CLIP RN50 image encoder :footcite:p:`radford2021learning`, whose features at strides 2, 4, 8 and 16 are mapped back to the image
    by a light U-Net decoder :math:`G` adapted from Transfer CLIP :footcite:p:`cheng2024transfer`. Only :math:`G` (11M parameters) is trainable.

    It is the prior of an unrolled plug-and-play ADMM for Poisson inverse problems, trained without ground truth
    with :class:`deepinv.loss.R2RLoss` and :class:`deepinv.loss.EILoss`.
    See :ref:`sphx_glr_auto_examples_unfolded_demo_frozen_clip_poisson.py`.

    The model is blind (``sigma`` is ignored) and takes RGB images in :math:`[0, 1]`, without CLIP normalization.
    Inputs are padded to a multiple of 16. Requires `timm <https://github.com/huggingface/pytorch-image-models>`_.

    .. note::

        As in the original implementation, the frozen encoder follows the train/eval mode of the module:
        in train mode, its batch-norm layers use batch statistics and update their running statistics.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> model = dinv.models.FrozenCLIPDenoiser(pretrained=None)
    >>> x = torch.rand(1, 3, 40, 40)
    >>> model(x).shape
    torch.Size([1, 3, 40, 40])

    :param str, None pretrained: ``"download"`` for the weights of :footcite:t:`diazdelgado2026frozen`
        (Poisson demosaicing with gain 0.01, trained with R2R and EI), a path to local weights,
        or ``None`` for random weights (encoder included, for testing only).
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param float feature_aug: in train mode, the encoder features are multiplied by :math:`1 + k \cdot` ``feature_aug`` :math:`\cdot \omega`,
        :math:`\omega \sim \mathcal{N}(0, 1)`, with :math:`k = 1, \dots, 4` from finest to coarsest scale :footcite:p:`cheng2024transfer`.
    :param torch.device, str device: device of the model.
    """

    def __init__(
        self,
        pretrained: str | None = "download",
        feature_aug: float = 0.02,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:  # pragma: no cover
            raise ImportError(
                "FrozenCLIPDenoiser requires timm. Install it with `pip install timm`."
            )

        self.feature_aug = feature_aug
        # CLIP weights are not downloaded here: the pretrained file also stores the encoder,
        # whose batch-norm running statistics were updated during training.
        self.encoder = timm.create_model(
            "resnet50_clip", features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.encoder.requires_grad_(False)

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.LeakyReLU(0.2),
            )

        self.decoder = nn.ModuleList(
            [
                conv_block(1024, 512),
                conv_block(512 + 512, 256),
                conv_block(256 + 256, 128),
                conv_block(128 + 64, 64),
                conv_block(64 + 3, 64),
            ]
        )
        self.last = nn.Conv2d(64, 3, 3, padding=1, bias=False)

        if pretrained is not None:
            if pretrained == "download":
                # TODO: upload the weights to HuggingFace and check this URL.
                name = "frozen_clip_demosaicing_poisson_0.01.pth"
                url = get_weights_url(model_name="frozen_clip", file_name=name)
                ckpt = load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt, strict=True)
            self.eval()

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma=None, **kwargs) -> torch.Tensor:
        r"""
        Denoise an image.

        :param torch.Tensor x: noisy RGB image of shape (B, 3, H, W).
        :param sigma: ignored.
        """
        return test_pad(self._forward, x, modulo=16)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)  # strides 2, 4, 8, 16
        if self.training and self.feature_aug > 0:
            feats = [
                f * (1 + k * self.feature_aug * torch.randn_like(f))
                for k, f in enumerate(feats, start=1)
            ]
        z = self.decoder[0](feats[-1])
        for block, skip in zip(
            self.decoder[1:], [*reversed(feats[:-1]), x], strict=True
        ):
            z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=True)
            z = block(torch.cat([z, skip], dim=1))
        return self.last(z)
