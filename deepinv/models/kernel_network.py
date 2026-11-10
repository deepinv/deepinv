from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_weights_url, load_state_dict_from_url
from deepinv.utils.patch_extractor import image_to_patches
import warnings


class KernelIdentificationNetwork(nn.Module):
    r"""
    Space varying blur kernel estimation network.

    U-Net proposed by :footcite:t:`carbajal2023blind`, estimating
    the parameters of :class:`deepinv.physics.SpaceVaryingBlur` forward model, i.e., blur kernels and corresponding spatial multipliers (masks).

    .. note::

        The estimated parameters should therefore be plugged into
        ``deepinv.physics.SpaceVaryingBlur(mask_first=False, padding="circular")``.

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

    :Examples:

        >>> import deepinv as dinv
        >>> import torch
        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> kernel_estimator = dinv.models.KernelIdentificationNetwork(device=device)
        >>> physics = dinv.physics.SpaceVaryingBlur(device=device, padding="circular", mask_first=False)
        >>> y = torch.randn(1, 3, 128, 128).to(device)  # random blurry image for demonstration
        >>> with torch.no_grad():
        ...     params = kernel_estimator(y)  # this outputs {"filters": ..., "multipliers": ...}
        >>> physics.update(**params) # update physics with estimated kernels
        >>> print(params["filters"].shape, params["multipliers"].shape)
        torch.Size([1, 1, 25, 33, 33]) torch.Size([1, 1, 25, 128, 128])

    """

    def __init__(
        self,
        filters: int = 25,
        blur_kernel_size: int = 33,
        bilinear: bool = False,
        no_softmax: bool = False,
        pretrained: str = "download",
        device: torch.device | str = "cpu",
    ):
        super(KernelIdentificationNetwork, self).__init__()
        if blur_kernel_size not in [33, 65]:
            raise ValueError(
                f"Only blur_kernel_size of 33 and 65 are currently supported, got {blur_kernel_size}."
            )
        if blur_kernel_size == 65 and pretrained == "download":
            raise ValueError(
                "Pretrained weights are not available for blur_kernel_size=65. Please set pretrained to None or provide your own pretrained weights."
            )

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
                    ckpt = load_state_dict_from_url(
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r"""
        Forward pass of the kernel estimation network.

        :param x: input blurry image of shape (N, C, H, W) with values in [0, 1]. Assumed to be non-gamma corrected (i.e., linear RGB).
        :return: dictionary with estimated blur kernels and spatial multipliers of the
            :math:`y = \sum_k w_k \odot (h_k \star x)` model:
            -  ``'filters'``: estimated blur kernels :math:`h_k` of shape (N, 1, K, blur_kernel_size, blur_kernel_size)
            -  ``'multipliers'``: estimated spatial masks :math:`w_k` of shape (N, 1, K, H, W)
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

        N = k.shape[0]  # H and W should be one
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


class DeepBlur(nn.Module):
    r"""
    PSF identification network for fluorescence microscopy.

    Estimates the point spread function (PSF) :math:`h` of a blurry and noisy image :math:`y = \noise{h \star x}`,
    assuming the blur is spatially invariant over the input.
    The network is a residual encoder followed by an atrous spatial pyramid pooling (ASPP) bottleneck and a regression head.
    The estimated PSF is nonnegative and sums to one (softmax with a learned temperature).

    The pretrained weights estimate :math:`71 \times 71` PSFs from grayscale patches of :math:`128 \times 128` to :math:`256 \times 256` pixels.
    They were trained on images blurred by optical PSFs with random aberrations (Zernike polynomials 3 to 36) and corrupted by
    Poisson-Gaussian noise, where the clean images are natural images partly turned into sparse, fluorescence-like structures.

    For space-varying blur, :meth:`forward_tiled` estimates one PSF per tile, which parametrizes
    :class:`deepinv.physics.TiledSpaceVaryingBlur`. See
    :ref:`sphx_glr_auto_examples_blind-inverse-problems_demo_tiled_blind_deconvolution_microscopy.py`.

    .. note::

        Channels are processed independently, and each input (or tile) is min-max normalized to :math:`[0, 1]`,
        so the intensity range of :math:`y` does not matter.

    :param int psf_size: size of the estimated square PSF. Pretrained weights are only available for `psf_size=71`.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights are initialized at random
        using PyTorch's default initialization. If ``pretrained='download'``, the weights are downloaded from the
        `MAMBO server <https://mambo-utoulouse.fr/weights>`_.
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param str, torch.device device: device to use.

    |sep|

    :Examples:

        Estimate a PSF per tile and use them in a space-varying blur operator:

        >>> import torch
        >>> import deepinv as dinv
        >>> model = dinv.models.DeepBlur(pretrained=None)
        >>> y = torch.rand(1, 1, 256, 256)
        >>> with torch.no_grad():
        ...     filters = model.forward_tiled(y, patch_size=128, stride=64)
        >>> filters.shape  # 3 x 3 tiles
        torch.Size([1, 1, 9, 71, 71])
        >>> physics = dinv.physics.TiledSpaceVaryingBlur(filters=filters, patch_size=128 + 71 - 1, stride=64)
        >>> physics.A_adjoint(y).shape
        torch.Size([1, 1, 326, 326])

    """

    def __init__(
        self,
        psf_size: int = 71,
        pretrained: str | None = "download",
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if pretrained == "download" and psf_size != 71:
            raise ValueError(
                f"Pretrained weights are only available for psf_size=71, got {psf_size}. "
                "Set pretrained to None or to the path of your own weights."
            )
        self.psf_size = psf_size
        nc = [64, 128, 256, 512, 768, 1024]

        # Encoder, downsampling by 16
        self.initial = _ConvBlock(1, nc[0], 3, padding=1)
        self.rb0 = _ResidualBlock(nc[0], nc[1], stride=2)
        self.rb1 = _ResidualBlock(nc[1], nc[2], stride=2)
        self.rb2 = _ResidualBlock(nc[2], nc[3], stride=2)
        self.rb3 = _ResidualBlock(nc[3], nc[4], stride=2)
        self.rb4 = _ResidualBlock(nc[4], nc[5], stride=1)
        self.aspp_bridge = _ASPP(nc[5], rates=(2, 4, 8))

        # Regression head: 8 x 8 candidate PSFs, mixed by a 1x1 convolution
        self.dec1 = _ConvBlock(nc[5], nc[4], 3, padding=1, norm="group")
        self.dec2 = _ConvBlock(nc[4], nc[3], 3, padding=1, norm="group")
        self.head_conv = nn.Sequential(
            nn.Conv2d(nc[3], psf_size**2, 3, stride=2, padding=1), nn.GELU()
        )
        self.mixing = nn.Conv2d(64, 1, kernel_size=1)
        self.temperature = nn.Parameter(torch.tensor(0.1))

        if pretrained is not None:
            if pretrained == "download":
                state_dict = load_state_dict_from_url(
                    "https://mambo-utoulouse.fr/weights/z3_z36/checkpoint.pth",
                    map_location="cpu",
                    file_name="deepblur_z3_z36.pth",
                    weights_only=True,
                )
            else:
                state_dict = torch.load(
                    pretrained, map_location="cpu", weights_only=True
                )
            # The original implementation stores the mixing convolution in a Sequential
            state_dict = {
                k.replace("mixing.1.", "mixing."): v for k, v in state_dict.items()
            }
            self.load_state_dict(state_dict)

        self.to(device)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        r"""
        Estimates the PSF of each channel of a blurry image.

        :param torch.Tensor y: blurry image of shape `(B, C, H, W)`, ideally with :math:`128 \leq H, W \leq 256`.
        :return: (:class:`torch.Tensor`) PSFs of shape `(B, C, psf_size, psf_size)`, nonnegative and summing to one.
        """
        B, C, H, W = y.shape
        y = y.reshape(B * C, 1, H, W)
        y_min = y.amin(dim=(-2, -1), keepdim=True)
        y_range = y.amax(dim=(-2, -1), keepdim=True) - y_min
        y = (y - y_min) / torch.where(y_range < 1e-8, 1.0, y_range)

        x = self.initial(y)
        x = self.rb4(self.rb3(self.rb2(self.rb1(self.rb0(x)))))
        x = self.aspp_bridge(x)
        x = self.head_conv(self.dec2(self.dec1(x)))
        if x.shape[-2:] != (8, 8):
            x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)

        # (B * C, psf_size**2, 8, 8) -> (B * C, 64, psf_size, psf_size)
        x = x.reshape(B * C, self.psf_size, self.psf_size, 64).permute(0, 3, 1, 2)
        x = self.mixing(x) / self.temperature
        h = torch.softmax(x.flatten(1), dim=1)
        return h.reshape(B, C, self.psf_size, self.psf_size)

    def forward_tiled(
        self,
        y: torch.Tensor,
        patch_size: int | tuple[int, int] = 256,
        stride: int | tuple[int, int] | None = None,
        batch_size: int = 16,
    ) -> torch.Tensor:
        r"""
        Estimates one PSF per overlapping tile of an image with space-varying blur.

        The tiles are those of :func:`deepinv.utils.image_to_patches`: the image is zero-padded at the bottom and right to fit an integer number of tiles.
        The PSFs parametrize :class:`deepinv.physics.TiledSpaceVaryingBlur` with patch size `patch_size + psf_size - 1` and the same `stride`,
        since this operator uses valid convolutions, i.e. its input is `psf_size - 1` pixels larger than :math:`y`.

        :param torch.Tensor y: blurry image of shape `(B, C, H, W)`.
        :param int, tuple[int, int] patch_size: tile size, ideally between 128 and 256.
        :param int, tuple[int, int] stride: stride between tiles. If `None`, `patch_size // 2`.
        :param int batch_size: number of tiles processed at once.
        :return: (:class:`torch.Tensor`) PSFs of shape `(B, C, K, psf_size, psf_size)`, where `K` is the number of tiles.
        """
        patches = image_to_patches(y, patch_size, stride)
        B, C, n_rows, n_cols = patches.shape[:4]
        patches = patches.reshape(-1, 1, *patches.shape[-2:])
        h = torch.cat([self(p) for p in patches.split(batch_size)])
        return h.reshape(B, C, n_rows * n_cols, self.psf_size, self.psf_size)


class _ConvBlock(nn.Module):
    r"""Convolution, instance or group normalization, and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        norm: str = "instance",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if norm == "instance"
            else nn.GroupNorm(32, out_channels)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _ResidualBlock(nn.Module):
    r"""Residual block with instance normalization and a strided convolution shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class _ASPP(nn.Module):
    r"""Atrous spatial pyramid pooling: parallel dilated convolutions, concatenated and projected."""

    def __init__(self, channels: int, rates: tuple[int, ...] = (2, 4, 8)):
        super().__init__()
        self.convs = nn.ModuleList(
            [_ConvBlock(channels, channels, 1, norm="group")]
            + [
                _ConvBlock(channels, channels, 3, padding=r, dilation=r, norm="group")
                for r in rates
            ]
        )
        self.project = _ConvBlock(len(self.convs) * channels, channels, 1, norm="group")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))
