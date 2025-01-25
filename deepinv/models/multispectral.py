import torch
import torch.nn as nn

from deepinv.utils.tensorlist import TensorList
from deepinv.physics.forward import Physics
from deepinv.physics.noise import PoissonNoise
from deepinv.physics.blur import Downsampling, Blur
from deepinv.physics.remote_sensing import Pansharpen


class ResNet(nn.Module):
    """Simple implementation of ResNet with ReLU activation

    :param int hidden_channels: number of hidden channels, defaults to 32
    :param int num_blocks: numer of ResNet blocks, defaults to 4
    :param bool batch_norm: whether to add batchnorm layers, defaults to True
    :param bool relu_before_addition: perform activation before residual addition, defaults to False
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        batch_norm: bool = True,
        relu_before_addition=False,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.relu_before_addition = relu_before_addition
        self.blocks = nn.ModuleList(
            [self._make_block(hidden_channels) for _ in range(num_blocks)]
        )

    def _make_block(self, hidden_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_channels) if self.batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_channels) if self.batch_norm else nn.Identity(),
            nn.ReLU() if self.relu_before_addition else nn.Identity(),
        )

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
            x = nn.ReLU()(x) if not self.relu_before_addition else x
        return x


class PanNet(nn.Module):
    """PanNet architecture for pan-sharpening.

    PanNet neural network from Yang et al. `PanNet: A Deep Network Architecture for Pan-Sharpening <https://ieeexplore.ieee.org/document/8237455/>`_, ICCV 2017.

    Takes input measurements as a :class:`deepinv.utils.TensorList` with elements (MS, PAN),
    where MS is the low-resolution multispectral image of shape (B, C, H, W) and PAN is the
    high-resolution panchromatic image of shape (B, 1, H*r, W*r) where r is the pan-sharpening factor.

    :param torch.nn.Module backbone_net: Backbone neural network, e.g. ResNet. If ``None``, defaults to a simple ResNet.
    :param tuple[int] hrms_shape: shape of high-resolution multispectral images (C,H,W), defaults to (4,900,900)
    :param int scale_factor: pansharpening downsampling ratio HR/LR, defaults to 4
    :param int highpass_kernel_size: square kernel size for extracting high-frequency features, defaults to 5
    :param str device: torch device, defaults to "cpu"
    """

    def __init__(
        self,
        backbone_net: nn.Module = None,
        hrms_shape: tuple = (4, 900, 900),
        scale_factor: int = 4,
        highpass_kernel_size: int = 5,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.hrms_shape = hrms_shape
        self.device = device
        self.upsampler = self.create_sampler("up", self.hrms_shape)

        self.boxblur = Blur(
            filter=torch.tensor(1.0 / highpass_kernel_size**2, device=device).expand(
                1, 1, highpass_kernel_size, highpass_kernel_size
            ),
            padding="reflect",
            device=device,
        ).A

        if backbone_net is None:
            backbone_net = ResNet()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=hrms_shape[0] + 1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            backbone_net,
            nn.Conv2d(
                in_channels=32,
                out_channels=hrms_shape[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def create_sampler(
        self, direction: str, hr_shape: tuple, noise_gain: float = 0.0
    ) -> Physics:
        """Helper function for downsampling/upsampling images (useful for reduced-resolution training with Wald's protocol).

        :param str direction: down or up
        :param tuple hr_shape: HRMS input shape (C,H,W)
        :param float noise_gain: noise applied to downsampling ONLY, defaults to 0.
        :return dinv.physics.Physics: deepinv sampler
        """
        sampler = Downsampling(
            img_size=hr_shape,
            factor=self.scale_factor,
            filter="bicubic",
            device=self.device,
        )

        if noise_gain > 0:
            sampler.noise_model = PoissonNoise(gain=noise_gain, clip_positive=True)

        return sampler if direction == "down" else sampler.A_adjoint

    def highpass(self, x):  # high-pass with box filter as per Yang et al.
        return x - self.boxblur(x)  # kornia.filters.BoxBlur((5, 5))(x)

    def forward(self, y: TensorList, physics: Pansharpen, *args, **kwargs):
        r"""
        Evaluate the pansharpening model

        :param deepinv.utils.TensorList y: (MS,PAN) images
        :param deepinv.physics.Pansharpen physics: Pansharpening operator
        """
        lr, pan = y

        lr_highpass = self.highpass(lr)
        pan_highpass = self.highpass(pan)

        lr_highpass_up = self.upsampler(lr_highpass)  # note fixed upsampler

        ms = torch.cat([pan_highpass, lr_highpass_up], dim=1)

        output = self.net(ms) + self.upsampler(lr)

        return output
