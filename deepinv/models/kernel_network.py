import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_weights_url
import warnings


class KernelIdentificationNetwork(nn.Module):
    r"""
    Space varying blur kernel estimation network.

    U-Net proposed by :footcite:t:`carbajal2023blind`, estimating
    the parameters of :class:`deepinv.physics.SpaceVaryingBlur` forward model, i.e., blur kernels and corresponding spatial multipliers (weights).

    Current implementation supports blur kernels of size 33x33 (default) and 65x65, and 1 or 3 input channels.


    Code adapted from https://github.com/GuillermoCarbajal/J-MKPD with permission from the author.

    Images are assumed to be in range [0, 1] before being passed to the network, and to be **non-gamma corrected** (i.e., linear RGB).
    If your blurry image has been gamma-corrected (e.g., standard sRGB images), consider applying an inverse gamma correction (e.g., raising to the power of 2.2)
    before passing it to the network for better results.

    :param int filters: number of blur kernels to estimate, defaults to 25.
    :param int blur_kernel_size: size of the blur kernels to estimate, defaults to 33. Only 33 and 65 are currently supported.
    :param bool bilinear: whether to use bilinear upsampling or transposed convolutions, defaults to False.
    :param bool no_softmax: whether to apply softmax to the estimated kernels, defaults to False.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with default parameters).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param str, torch.device device: device to use, defaults to 'cpu'.

    |sep|

    Example usage:

    >>> import deepinv as dinv
    >>> import torch
    >>> device = dinv.utils.get_freer_gpu(verbose=False) if torch.cuda.is_available() else "cpu"
    >>> kernel_estimator = dinv.models.KernelIdentificationNetwork(device=device)
    >>> physics = dinv.physics.SpaceVaryingBlur(device=device, padding="constant")
    >>> y = torch.randn(1, 3, 128, 128).to(device)  # random blurry image for demonstration
    >>> with torch.no_grad():
    ...     params = kernel_estimator(y)  # this outputs {"filters": ..., "multipliers": ...}
    >>> physics.update(**params) # update physics with estimated kernels
    >>> print(params["filters"].shape, params["multipliers"].shape)
    torch.Size([1, 1, 25, 33, 33]) torch.Size([1, 1, 25, 128, 128])


    """

    def __init__(
        self,
        filters=25,
        blur_kernel_size=33,
        bilinear=False,
        no_softmax=False,
        pretrained="download",
        device="cpu",
    ):
        super(KernelIdentificationNetwork, self).__init__()

        self.no_softmax = no_softmax
        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blur_kernel_size = blur_kernel_size
        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.K = filters

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.feat = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up1 = Up(1024, 1024, 512, bilinear)
        self.up2 = Up(512, 512, 256, bilinear)
        self.up3 = Up(256, 256, 128, bilinear)
        self.up4 = Up(128, 128, 64, bilinear)
        self.up5 = Up(64, 64, 64, bilinear)

        self.masks_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, self.K, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        self.feat5_gap = PooledSkip(2)
        self.feat4_gap = PooledSkip(4)
        self.feat3_gap = PooledSkip(8)
        self.feat2_gap = PooledSkip(16)
        self.feat1_gap = PooledSkip(32)

        self.kernel_up1 = Up(1024, 1024, 512, bilinear)
        self.kernel_up2 = Up(512, 512, 256, bilinear)
        self.kernel_up3 = Up(256, 256, 256, bilinear)
        self.kernel_up4 = Up(256, 128, 128, bilinear)
        self.kernel_up5 = Up(128, 64, 64, bilinear)
        if self.blur_kernel_size > 33:
            self.kernel_up6 = Up(64, 0, 64, bilinear)

        self.kernels_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, self.K, kernel_size=3, padding=1),
        )
        self.kernel_softmax = nn.Softmax(dim=2)

        # load pretrained weights
        if pretrained is not None:
            if pretrained == "download":
                if self.K == 25:
                    file_name = "carbajal_kernel_identification_network.pth"
                    url = get_weights_url(
                        model_name="kernel_identification", file_name=file_name
                    )
                    ckpt = torch.hub.load_state_dict_from_url(
                        url,
                        map_location=lambda storage, loc: storage,
                        file_name=file_name,
                        check_hash=True,
                        weights_only=True,
                    )
                    self.load_state_dict(ckpt, strict=True)
                else:
                    warnings.warn(
                        "Pretrained weights not available for the specified configuration. Proceeding without loading weights."
                    )
            else:
                self.load_state_dict(torch.load(pretrained))

        self.to(device)

    def forward(self, x):
        r"""
        Forward pass of the kernel estimation network.

        :param x: input blurry image of shape (N, C, H, W) with values in [0, 1]. Assumed to be non-gamma corrected (i.e., linear RGB).
        :return: dictionary with estimated blur kernels and spatial multipliers:
            -  ``'filters'``: estimated blur kernels of shape (N, 1, K, blur_kernel_size, blur_kernel_size)
            -  ``'multipliers'``: estimated spatial multipliers of shape (N, 1, K, H, W)
        """
        x = x - 0.5  # normalize input to [−0.5,0.5]
        # Encoder
        if x.shape[1] == 3:
            x1 = self.inc_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        feat6_gap = x6_feat.mean((2, 3), keepdim=True)
        feat5_gap = self.feat5_gap(x5_feat)
        feat4_gap = self.feat4_gap(x4_feat)
        feat3_gap = self.feat3_gap(x3_feat)
        feat2_gap = self.feat2_gap(x2_feat)
        feat1_gap = self.feat1_gap(x1_feat)

        k1 = self.kernel_up1(feat6_gap, feat5_gap)
        k2 = self.kernel_up2(k1, feat4_gap)
        k3 = self.kernel_up3(k2, feat3_gap)
        k4 = self.kernel_up4(k3, feat2_gap)
        k5 = self.kernel_up5(k4, feat1_gap)

        if self.blur_kernel_size == 65:
            k6 = self.kernel_up6(k5)
            k = self.kernels_end(k6)
        else:
            k = self.kernels_end(k5)

        N, F, H, W = k.shape  # H and W should be one
        k = k.view(N, self.K, self.blur_kernel_size * self.blur_kernel_size)

        if self.no_softmax:
            k = F.leaky_relu(k)
        else:
            k = self.kernel_softmax(k)

        k = k.view(N, self.K, self.blur_kernel_size, self.blur_kernel_size)

        # Decoder
        x7 = self.up1(x6_feat, x5_feat)
        x8 = self.up2(x7, x4_feat)
        x9 = self.up3(x8, x3_feat)
        x10 = self.up4(x9, x2_feat)
        x11 = self.up5(x10, x1_feat)
        logits = self.masks_end(x11)

        # change from corr to conv
        k = torch.flip(k, [2, 3])
        return {"filters": k.unsqueeze(1), "multipliers": logits.unsqueeze(1)}


class Down(nn.Module):
    """double conv and then downscaling with maxpool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.down_sampling = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.feat = nn.Sequential(
            nn.Conv2d(
                feat_channels + out_channels, out_channels, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x1 = self.double_conv(x1)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 is not None:

            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        feat = self.feat(x)
        return feat


class PooledSkip(nn.Module):
    def __init__(self, output_spatial_size):
        super().__init__()

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2, 3), keepdim=True)
        return global_avg_pooling.repeat(
            1, 1, self.output_spatial_size, self.output_spatial_size
        )
