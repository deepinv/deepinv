"""
## Multi-Stage Progressive Image Restoration
MDPRNet model
Code adapted from Zamir et al., at: https://github.com/swz30/MPRNet/

See deepinv/models/third_party/LICENSE for license details.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepinv.models import Denoiser, Reconstructor
from deepinv.models.utils import (
    conv_nd,
    fix_dim,
    get_weights_url,
    load_state_dict_from_url,
)
from deepinv.physics import Physics


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = False,
    stride: int = 1,
    dim: int = 2,
) -> nn.Module:
    return conv_nd(dim)(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def upsample(dim: int, scale_factor: float) -> nn.Upsample:
    return nn.Upsample(
        scale_factor=scale_factor,
        mode={2: "bilinear", 3: "trilinear"}[dim],
        align_corners=False,
    )


def split_patches(x: torch.Tensor, dims: list[int]) -> list[torch.Tensor]:
    """Split tensor ``x`` into ``2 ** len(dims)`` equal patches, halved along each dim in ``dims``."""
    patches = [x]
    for d in dims:
        patches = [p for patch in patches for p in patch.chunk(2, d)]
    return patches


def merge_patches(
    feats: list[list[torch.Tensor]], dims: list[int]
) -> list[torch.Tensor]:
    """Inverse of :func:`split_patches` for a list of per-patch feature lists."""
    for d in reversed(dims):
        feats = [
            [torch.cat((a, b), d) for a, b in zip(feats[i], feats[i + 1], strict=True)]
            for i in range(0, len(feats), 2)
        ]
    return feats[0]


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(
        self, channel: int, reduction: int = 16, bias: bool = False, dim: int = 2
    ):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[dim](1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            conv(channel, channel // reduction, 1, bias=bias, dim=dim),
            nn.ReLU(inplace=True),
            conv(channel // reduction, channel, 1, bias=bias, dim=dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        bias: bool,
        act: nn.Module,
        dim: int = 2,
    ):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dim=dim))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias, dim=dim))

        self.CA = CALayer(n_feat, reduction, bias=bias, dim=dim)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int, bias: bool, dim: int = 2):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias, dim=dim)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias, dim=dim)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias, dim=dim)

    def forward(
        self, x: torch.Tensor, x_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


## U-Net


class Encoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        scale_unetfeats: int,
        csff: bool,
        dim: int = 2,
    ):
        super(Encoder, self).__init__()

        self.encoder_level1 = [
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim)
            for _ in range(2)
        ]
        self.encoder_level2 = [
            CAB(
                n_feat + scale_unetfeats,
                kernel_size,
                reduction,
                bias=bias,
                act=act,
                dim=dim,
            )
            for _ in range(2)
        ]
        self.encoder_level3 = [
            CAB(
                n_feat + (scale_unetfeats * 2),
                kernel_size,
                reduction,
                bias=bias,
                act=act,
                dim=dim,
            )
            for _ in range(2)
        ]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_unetfeats, dim=dim)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats, dim=dim)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = conv_nd(dim)(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = conv_nd(dim)(
                n_feat + scale_unetfeats,
                n_feat + scale_unetfeats,
                kernel_size=1,
                bias=bias,
            )
            self.csff_enc3 = conv_nd(dim)(
                n_feat + (scale_unetfeats * 2),
                n_feat + (scale_unetfeats * 2),
                kernel_size=1,
                bias=bias,
            )

            self.csff_dec1 = conv_nd(dim)(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = conv_nd(dim)(
                n_feat + scale_unetfeats,
                n_feat + scale_unetfeats,
                kernel_size=1,
                bias=bias,
            )
            self.csff_dec3 = conv_nd(dim)(
                n_feat + (scale_unetfeats * 2),
                n_feat + (scale_unetfeats * 2),
                kernel_size=1,
                bias=bias,
            )

    def forward(
        self,
        x: torch.Tensor,
        encoder_outs: list[torch.Tensor] | None = None,
        decoder_outs: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = (
                enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])
            )

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = (
                enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])
            )

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = (
                enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])
            )

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        scale_unetfeats: int,
        dim: int = 2,
    ):
        super(Decoder, self).__init__()

        self.decoder_level1 = [
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim)
            for _ in range(2)
        ]
        self.decoder_level2 = [
            CAB(
                n_feat + scale_unetfeats,
                kernel_size,
                reduction,
                bias=bias,
                act=act,
                dim=dim,
            )
            for _ in range(2)
        ]
        self.decoder_level3 = [
            CAB(
                n_feat + (scale_unetfeats * 2),
                kernel_size,
                reduction,
                bias=bias,
                act=act,
                dim=dim,
            )
            for _ in range(2)
        ]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(
            n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim
        )
        self.skip_attn2 = CAB(
            n_feat + scale_unetfeats,
            kernel_size,
            reduction,
            bias=bias,
            act=act,
            dim=dim,
        )

        self.up21 = SkipUpSample(n_feat, scale_unetfeats, dim=dim)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats, dim=dim)

    def forward(self, outs: list[torch.Tensor]) -> list[torch.Tensor]:
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]


##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels: int, s_factor: int, dim: int = 2):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            upsample(dim, 0.5),
            conv_nd(dim)(
                in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels: int, s_factor: int, dim: int = 2):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            upsample(dim, 2),
            conv_nd(dim)(
                in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels: int, s_factor: int, dim: int = 2):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(
            upsample(dim, 2),
            conv_nd(dim)(
                in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = x + y
        return x


## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        num_cab: int,
        dim: int = 2,
    ):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim)
            for _ in range(num_cab)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size, dim=dim))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res += x
        return res


class ORSNet(nn.Module):
    def __init__(
        self,
        n_feat: int,
        scale_orsnetfeats: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        scale_unetfeats: int,
        num_cab: int,
        dim: int = 2,
    ):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(
            n_feat + scale_orsnetfeats,
            kernel_size,
            reduction,
            act,
            bias,
            num_cab,
            dim=dim,
        )
        self.orb2 = ORB(
            n_feat + scale_orsnetfeats,
            kernel_size,
            reduction,
            act,
            bias,
            num_cab,
            dim=dim,
        )
        self.orb3 = ORB(
            n_feat + scale_orsnetfeats,
            kernel_size,
            reduction,
            act,
            bias,
            num_cab,
            dim=dim,
        )

        self.up_enc1 = UpSample(n_feat, scale_unetfeats, dim=dim)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats, dim=dim)

        self.up_enc2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats, dim=dim),
            UpSample(n_feat, scale_unetfeats, dim=dim),
        )
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats, dim=dim),
            UpSample(n_feat, scale_unetfeats, dim=dim),
        )

        self.conv_enc1 = conv_nd(dim)(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_enc2 = conv_nd(dim)(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_enc3 = conv_nd(dim)(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )

        self.conv_dec1 = conv_nd(dim)(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_dec2 = conv_nd(dim)(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )
        self.conv_dec3 = conv_nd(dim)(
            n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_outs: list[torch.Tensor],
        decoder_outs: list[torch.Tensor],
    ) -> torch.Tensor:
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = (
            x
            + self.conv_enc2(self.up_enc1(encoder_outs[1]))
            + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        )

        x = self.orb3(x)
        x = (
            x
            + self.conv_enc3(self.up_enc2(encoder_outs[2]))
            + self.conv_dec3(self.up_dec2(decoder_outs[2]))
        )

        return x


class MPRNet(Reconstructor, Denoiser):
    r"""
    MPRNet restoration model.

    MPRNet is a blind restoration model that was proposed in :footcite:t:`zamir2021multi`.

    There are pretrained weights available for two tasks, which can be loaded via ``pretrained``:

        - Real-world image denoising on SIDD: ``pretrained='denoising'`` with the default architecture.

        - Image deblurring: ``pretrained='deblurring'`` with ``n_feat=96`` and default values for the other arguments.

    .. note::

        All spatial dimensions of the input (height and width, and depth for 3D
        networks) must be divisible by 8, otherwise an error is raised.

    :param int in_channels: number of channels in the input image.
    :param int out_channels: number of channels in the output image.
    :param int n_feat: number of feature maps in the intermediate layers.
    :param int scale_unetfeats: number of feature maps added at each level of the U-Net
        encoder and decoder stages.
    :param int scale_orsnetfeats: number of feature maps added to the original resolution
        stage.
    :param int num_cab: number of channel attention blocks in each original resolution
        block.
    :param int kernel_size: kernel size of the convolutional layers.
    :param int reduction: reduction factor of the channel attention layers.
    :param bool bias: use learnable biases in the convolutional layers.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='denoising'`` or ``pretrained='deblurring'``, the weights
        will be downloaded from an online repository, which requires the architecture of the original paper (for
        ``'deblurring'``, set ``n_feat=96``). Finally, ``pretrained`` can also be set as a path to the user's own
        pretrained weights. See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param torch.device, str device: device to put the model on.
    :param str, int dim: whether to build a 2D or 3D network (if str, can be "2", "2d",
        "3D", etc.).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        n_feat: int = 80,
        scale_unetfeats: int = 48,
        scale_orsnetfeats: int = 32,
        num_cab: int = 8,
        kernel_size: int = 3,
        reduction: int = 4,
        bias: bool = False,
        pretrained: str | None = None,
        device: torch.device | str = "cpu",
        dim: str | int = 2,
    ):
        super(MPRNet, self).__init__()

        self.dim = fix_dim(dim)
        dim = self.dim

        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(
            conv(in_channels, n_feat, kernel_size, bias=bias, dim=dim),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim),
        )
        self.shallow_feat2 = nn.Sequential(
            conv(in_channels, n_feat, kernel_size, bias=bias, dim=dim),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim),
        )
        self.shallow_feat3 = nn.Sequential(
            conv(in_channels, n_feat, kernel_size, bias=bias, dim=dim),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act, dim=dim),
        )

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            csff=False,
            dim=dim,
        )
        self.stage1_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, dim=dim
        )

        self.stage2_encoder = Encoder(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            csff=True,
            dim=dim,
        )
        self.stage2_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, dim=dim
        )

        self.stage3_orsnet = ORSNet(
            n_feat,
            scale_orsnetfeats,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            num_cab,
            dim=dim,
        )

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, dim=dim)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, dim=dim)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias, dim=dim)
        self.concat23 = conv(
            n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias, dim=dim
        )
        self.tail = conv(
            n_feat + scale_orsnetfeats, out_channels, kernel_size, bias=bias, dim=dim
        )

        if pretrained is not None:
            if pretrained in ("denoising", "deblurring"):
                if self.dim != 2:  # pragma: no cover
                    raise ValueError(
                        f"Downloadable MPRNet weights are 2D only, got dim={self.dim}."
                    )
                expected = {
                    "in_channels": 3,
                    "out_channels": 3,
                    "n_feat": 80 if pretrained == "denoising" else 96,
                    "scale_unetfeats": 48,
                    "scale_orsnetfeats": 32,
                    "num_cab": 8,
                    "kernel_size": 3,
                    "reduction": 4,
                    "bias": False,
                }
                actual = {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "n_feat": n_feat,
                    "scale_unetfeats": scale_unetfeats,
                    "scale_orsnetfeats": scale_orsnetfeats,
                    "num_cab": num_cab,
                    "kernel_size": kernel_size,
                    "reduction": reduction,
                    "bias": bias,
                }
                mismatches = [
                    f"{k}: expected {v}, got {actual[k]}"
                    for k, v in expected.items()
                    if actual[k] != v
                ]
                if mismatches:  # pragma: no cover
                    raise ValueError(
                        f"pretrained='{pretrained}' requires the original "
                        f"architecture: {', '.join(mismatches)}."
                    )
                file_name = f"model_{pretrained}.pth"
                url = get_weights_url(model_name="MPRNet", file_name=file_name)
                ckpt = load_state_dict_from_url(
                    url,
                    map_location=lambda storage, loc: storage,
                    file_name=file_name,
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            self.load_state_dict(ckpt, strict=True)
            self.eval()

        if device is not None:
            self.to(device)

    def forward(
        self, x: torch.Tensor, physics: Physics | None = None, **kwargs
    ) -> torch.Tensor:
        if x.dim() != self.dim + 2:
            raise ValueError(
                f"Expected a {self.dim + 2}-dim input (B, C, ...), got {x.dim()}."
            )

        # Original-resolution Image for Stage 3
        H = x.size(-2)

        if any(s % 8 != 0 for s in x.shape[2:]):
            raise ValueError(
                f"MPRNet requires all spatial dimensions divisible by 8, "
                f"but the input has spatial size {tuple(x.shape[2:])}."
            )

        # Multi-Patch Hierarchy: Split Image into non-overlapping patches

        # Two Patches for Stage 2, split along H
        x2top_img = x[..., 0 : int(H / 2), :]
        x2bot_img = x[..., int(H / 2) : H, :]

        # Stage 1 patches: each Stage 2 patch is split along the remaining spatial
        # dims (W in 2D, D and W in 3D)
        split_dims = [3] if self.dim == 2 else [2, 4]

        ##-------------- Stage 1---------------------

        ## Compute Shallow Features and process all patches with the Encoder of Stage 1
        feat1_top = merge_patches(
            [
                self.stage1_encoder(self.shallow_feat1(p))
                for p in split_patches(x2top_img, split_dims)
            ],
            split_dims,
        )
        feat1_bot = merge_patches(
            [
                self.stage1_encoder(self.shallow_feat1(p))
                for p in split_patches(x2bot_img, split_dims)
            ],
            split_dims,
        )

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Apply Supervised Attention Module (SAM)
        x2top_samfeats, _ = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, _ = self.sam12(res1_bot[0], x2bot_img)

        ##-------------- Stage 2---------------------

        ## Compute Shallow Features
        x2top = self.shallow_feat2(x2top_img)
        x2bot = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [
            torch.cat((k, v), -2) for k, v in zip(feat2_top, feat2_bot, strict=False)
        ]

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, _ = self.sam23(res2[0], x)

        ##-------------- Stage 3---------------------

        ## Compute Shallow Features
        x3 = self.shallow_feat3(x)

        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))

        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)

        stage3_img = self.tail(x3_cat)

        return stage3_img + x
