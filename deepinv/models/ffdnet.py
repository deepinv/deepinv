from __future__ import annotations
import torch
import torch.nn as nn

from .base import Denoiser
from .drunet import weights_init_drunet


class FFDNet(Denoiser):
    r"""
    FFDNet denoiser network.

    The network architecture is based on the paper :footcite:t:`zhang2018ffdnet`.
    and consists of a ``PixelUnshuffle`` downsampling operation, a series of 3x3 convolutional layers
    (similar to DnCNN), followed by a ``PixelShuffle`` upsampling operation to get back to the original shape.

    The network takes into account the noise level of the input image, which is encoded as an additional input channel.

    :param int n_conv_layers: Number of convolutional layers used. Default: 15
    :param int nf: Number of channels per convolutional layer. Default: 64
    :param int img_channels: Number of channels of your input image. Default: 1 (greyscale)
    :param bool residual_denoising: Whether to use a residual connection between input image and the network output. Default: False
    :param str norm: normalization to use in the convolutional layers. Choose from instance_norm, batch_norm, or None (no norm). Default: batch_norm
    :param bool orthogonal_init: Apply orthogonal initialization to the convolutional weights. Ignored if pretrained not None. Default: True
    :param bool last_conv_bias: Set the learnable bias on or off on the final convolution. Default: False
    :param str pretrained: Load pretrained weights from a checkpoint. Default: None
    :param torch.device, str device: Device to put the model on.
    """

    def __init__(
        self,
        n_conv_layers: int = 15,
        nf: int = 64,
        img_channels: int = 1,
        residual_denoising: bool = False,
        norm: str | None = "batch_norm",
        orthogonal_init: bool = True,
        last_conv_bias: bool = False,
        pretrained: str | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        downsample_factor = 2
        self.downsampler = nn.PixelUnshuffle(downsample_factor)
        self.upsampler = nn.PixelShuffle(downsample_factor)
        if norm not in ["instance_norm", "batch_norm", None]:  # pragma: no cover
            raise ValueError(
                f"norm must be one of (instance_norm, batch_norm, None), but got {norm}"
            )
        norm = {
            "instance_norm": nn.InstanceNorm2d,
            "batch_norm": nn.BatchNorm2d,
            None: nn.Identity(),
        }[norm]
        blocks = []
        blocks.append(
            nn.Sequential(
                nn.Conv2d((img_channels + 1) * downsample_factor**2, nf, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(n_conv_layers - 2):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(nf, nf, 3, padding=1),
                    norm(nf),
                    nn.ReLU(inplace=True),
                )
            )
        blocks.append(
            nn.Conv2d(
                nf,
                (img_channels) * downsample_factor**2,
                3,
                padding=1,
                bias=last_conv_bias,
            )
        )
        self.blocks = nn.Sequential(*blocks)
        self.residual_denoising = residual_denoising
        if orthogonal_init:
            self.apply(weights_init_drunet)  # DRUNet also applies orthogonal init.
        if pretrained is not None:
            if pretrained == "download":
                raise ValueError(
                    'Received pretrained "download", but FFDNet has no downloadable weights.'
                )
            state = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(state, strict=True)
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        r"""
        Run the denoiser on image with noise level :math:`\sigma`.

        :param torch.Tensor x: noisy image
        :param float, torch.Tensor sigma: noise level. If ``sigma`` is a float, it is used for all images in the batch.
            If ``sigma`` is a tensor, it must be of shape ``(batch_size,)``.
        """
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:  # pragma: no cover
            raise ValueError(
                f"FFDNet requires H,W both divisble by 2. Got tensor of shape {tuple(x.shape)}"
            )
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.full(
                    (x.size(0), 1, x.size(2), x.size(3)),
                    sigma,
                    device=x.device,
                    dtype=x.dtype,
                )
        else:
            noise_level_map = torch.full(
                (x.size(0), 1, *x.shape[2:]), sigma, dtype=x.dtype, device=x.device
            )
        if self.residual_denoising:
            noisy_img = x
        x = torch.cat((x, noise_level_map), 1)
        x = self.downsampler(x)
        x = self.blocks(x)
        x = self.upsampler(x)
        if self.residual_denoising:
            x = x + noisy_img
        return x
