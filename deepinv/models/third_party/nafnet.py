# This file is adapted from https://github.com/megvii-research/NAFNet/
# Copyright (c) 2022 megvii-model
# See THIRD_PARTY_LICENSES for the full text.
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepinv.models import Denoiser, Reconstructor
from deepinv.models.utils import get_weights_url, load_state_dict_from_url
from deepinv.physics import Physics


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # this normalizes over channels.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        c: int,
        DW_Expand: int = 2,
        FFN_Expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(Reconstructor, Denoiser):
    r"""
    NAFNet restoration model.

    NAFNet is a blind restoration model that was proposed in :footcite:t:`chen2022simple`..

    There are pretrained weights available for two tasks (each with ``width=32`` and
    ``width=64`` variants), which can be loaded via ``pretrained``:

        - Image deblurring on GoPro: ``pretrained='deblurring'`` with ``width=32`` or ``64``,
          ``middle_blk_num=1``, ``enc_blk_nums=[1, 1, 1, 28]`` and ``dec_blk_nums=[1, 1, 1, 1]``.

        - Real-world image denoising on SIDD: ``pretrained='denoising'`` with ``width=32`` or
          ``64``, ``middle_blk_num=12``, ``enc_blk_nums=[2, 2, 4, 8]`` and
          ``dec_blk_nums=[2, 2, 2, 2]``.

    :param int img_channel: number of channels in the input and output image. Default: 3
    :param int width: the base width of the network, doubled at each downsampling stage. Default: 16
    :param int middle_blk_num: number of blocks in the middle (bottleneck) stage. Default: 1
    :param list[int], tuple[int, ...] enc_blk_nums: number of blocks per encoder stage; its
        length sets the number of downsampling levels. Default: ``(1, 1, 1, 1)``.
    :param list[int], tuple[int, ...] dec_blk_nums: number of blocks per decoder stage; its
        length must match ``enc_blk_nums``. Default: ``(1, 1, 1, 1)``.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights
        will be initialized at random using Pytorch's default initialization. If
        ``pretrained='deblurring'`` or ``pretrained='denoising'``, the weights will be downloaded
        from an online repository, which requires the architecture of the original paper (see
        above). Finally, ``pretrained`` can also be set as a path
        to the user's own pretrained weights. See :ref:`pretrained-weights <pretrained-weights>`
        for more details.
    :param torch.device, str device: device to put the model on. Default: ``"cpu"``.
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums: list[int] | tuple[int, ...] = (1, 1, 1, 1),
        dec_blk_nums: list[int] | tuple[int, ...] = (1, 1, 1, 1),
        pretrained: str | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

        if pretrained is not None:
            if pretrained in ("deblurring", "denoising"):
                dataset = "GoPro" if pretrained == "deblurring" else "SIDD"
                expected = {
                    "deblurring": {
                        "middle_blk_num": 1,
                        "enc_blk_nums": [1, 1, 1, 28],
                        "dec_blk_nums": [1, 1, 1, 1],
                    },
                    "denoising": {
                        "middle_blk_num": 12,
                        "enc_blk_nums": [2, 2, 4, 8],
                        "dec_blk_nums": [2, 2, 2, 2],
                    },
                }[pretrained]
                actual = {
                    "middle_blk_num": middle_blk_num,
                    "enc_blk_nums": enc_blk_nums,
                    "dec_blk_nums": dec_blk_nums,
                }
                mismatches = [
                    f"{k}: expected {v}, got {actual[k]}"
                    for k, v in expected.items()
                    if list(actual[k]) != v
                ]
                if width not in (32, 64):
                    mismatches.append(f"width: expected 32 or 64, got {width}")
                if mismatches:  # pragma: no cover
                    raise ValueError(
                        f"pretrained='{pretrained}' requires the original "
                        f"architecture: {', '.join(mismatches)}."
                    )
                file_name = f"NAFNet-{dataset}-width{width}.pth"
                url = get_weights_url(model_name="NAFNet", file_name=file_name)
                ckpt = load_state_dict_from_url(
                    url,
                    map_location=lambda storage, loc: storage,
                    file_name=file_name,
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            if isinstance(ckpt, dict) and "params" in ckpt:
                ckpt = ckpt["params"]
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            self.load_state_dict(ckpt, strict=True)
            self.eval()

        if device is not None:
            self.to(device)

    def forward(
        self, x: torch.Tensor, physics: Physics | None = None, **kwargs: Any
    ) -> torch.Tensor:
        r"""
        Run NAFNet on the noisy or degraded image. The noise level or physics are ignored.

        If the image shape cannot be processed, the image will be padded before being given to the network.
        After going through the network, it will be cropped back to the original shape.

        :param torch.Tensor x: Image to be restored.
        :param deepinv.physics.Physics physics: forward operator (not used).

        :returns: (:class:`torch.Tensor`) restored image estimate.
        """
        _, _, H, W = x.shape
        inp = self.check_image_size(x)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs, strict=True):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(
            self.decoders, self.ups, encs[::-1], strict=True
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
