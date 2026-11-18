# Code adapted from Transfer CLIP by Cheng et al., at: https://github.com/alwaysuu/CLIPDenoising
# Copyright (c) 2024 Jun Cheng. MIT License, see the LICENSE file in this folder.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepinv.models.base import Denoiser
from deepinv.models.utils import get_weights_url, load_state_dict_from_url, test_pad


class TransferCLIP(Denoiser):
    r"""
    Transfer CLIP denoiser network.

    The network architecture is based on the paper :footcite:t:`cheng2024transfer`. It is an asymmetric encoder-decoder
    :math:`\denoiser{x}{\sigma} = G(E(x), x)`, where :math:`E` is the frozen CLIP RN50 image encoder :footcite:p:`radford2021learning`
    and :math:`G` is a light U-Net decoder that maps the features of :math:`E` at strides 2, 4, 8 and 16, together with the noisy image,
    back to the image. Only :math:`G` (11M parameters) is trainable. Trained on Gaussian noise with progressive feature augmentation,
    the network generalizes to unseen noise distributions.

    Code adapted from https://github.com/alwaysuu/CLIPDenoising.

    Frozen CLIP Priors :footcite:p:`diazdelgado2026frozen` uses this network as the prior of an unrolled plug-and-play ADMM
    for Poisson inverse problems, and trains its decoder without ground truth with :class:`deepinv.loss.R2RLoss` and :class:`deepinv.loss.EILoss`.
    See :ref:`sphx_glr_auto_examples_unfolded_demo_transfer_clip.py` for both uses.

    The network is blind (``sigma`` is ignored) and takes RGB images in :math:`[0, 1]`, without CLIP normalization.
    Inputs are padded to a multiple of 16. Requires `timm <https://github.com/huggingface/pytorch-image-models>`_.

    .. note::

        As in the original implementation, the frozen encoder follows the train/eval mode of the module:
        in train mode, its batch-norm layers use batch statistics and update their running statistics.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> model = dinv.models.TransferCLIP(pretrained=None)
    >>> x = torch.rand(1, 3, 40, 40)
    >>> model(x).shape
    torch.Size([1, 3, 40, 40])

    :param str, None pretrained: Pretrained weights to load:

        - ``"download"`` (default): Transfer CLIP weights :footcite:p:`cheng2024transfer`, trained on CBSD432 with Gaussian noise of standard deviation 15/255.
        - ``"poisson_demosaicing"``: Frozen CLIP Priors weights :footcite:p:`diazdelgado2026frozen`, trained from the Transfer CLIP weights
          for Poisson demosaicing with gain 0.01.
        - A path to the user's own pretrained weights.
        - ``None``: Random weights, including the encoder. For testing only.

        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param float aug_level: Level of the progressive feature augmentation. In train mode, the encoder features are multiplied by
        :math:`1 + k \cdot` ``aug_level`` :math:`\cdot \omega`, with :math:`\omega \sim \mathcal{N}(0, 1)` and :math:`k = 1, \dots, 4`
        from the finest to the coarsest scale. Transfer CLIP uses 0.025 and Frozen CLIP Priors uses 0.02.
    :param torch.device, str device: Device to put the model on.
    """

    def __init__(
        self,
        pretrained: str | None = "download",
        aug_level: float = 0.025,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:  # pragma: no cover
            raise ImportError(
                "TransferCLIP requires timm. Install it with `pip install timm`."
            )

        self.aug_level = aug_level
        # CLIP weights are not downloaded here: the pretrained files also store the encoder,
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

        weights = {
            "download": "transfer_clip_gaussian_sigma15.pth",
            "poisson_demosaicing": "frozen_clip_demosaicing_poisson_0.01.pth",
        }
        if pretrained in weights:
            # TODO: upload the weights to HuggingFace and check this URL.
            name = weights[pretrained]
            url = get_weights_url(model_name="transfer_clip", file_name=name)
            ckpt = load_state_dict_from_url(
                url, map_location=lambda storage, loc: storage, file_name=name
            )
        elif pretrained is not None:
            ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
        if pretrained is not None:
            self.load_state_dict(ckpt, strict=True)
            self.eval()

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma=None, **kwargs) -> torch.Tensor:
        r"""
        Denoise an image.

        :param torch.Tensor x: Noisy RGB image of shape (B, 3, H, W).
        :param sigma: Ignored, the network is blind.
        :return: (:class:`torch.Tensor`) Denoised image of shape (B, 3, H, W).
        """
        return test_pad(self._forward, x, modulo=16)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)  # strides 2, 4, 8, 16
        if self.training and self.aug_level > 0:
            feats = [
                f * (1 + k * self.aug_level * torch.randn_like(f))
                for k, f in enumerate(feats, start=1)
            ]
        z = self.decoder[0](feats[-1])
        for block, skip in zip(
            self.decoder[1:], [*reversed(feats[:-1]), x], strict=True
        ):
            z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=True)
            z = block(torch.cat([z, skip], dim=1))
        return self.last(z)
