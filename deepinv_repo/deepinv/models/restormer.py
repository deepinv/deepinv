r"""Define the neural network architecture of the Restormer.

Model specialized in restoration tasks including deraining, single-image motion deblurring,
defocus deblurring and image denoising for high-resolution images. Code adapted from
https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py.

Restormer: Efficient Transformer for High-Resolution Image Restoration
Authors: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
Paper: https://arxiv.org/abs/2111.09881
Code: https://github.com/swz30/Restormer
"""

import numbers
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import get_weights_url, test_pad
from .base import Denoiser


class Restormer(Denoiser):
    r"""
    Restormer denoiser network.

    This network architecture was proposed in the paper:
    `Restormer: Efficient Transformer for High-Resolution Image Restoration <https://arxiv.org/abs/2111.09881>`_

    By default, the model is a denoising network with pretrained weights. For other tasks such as deraining, some arguments needs to be adapted.

    :param int in_channels: number of channels of the input.
    :param int out_channels: number of channels of the output.
    :param int dim: number of channels after the first conv operation (``in_channel``, H, W) -> (``dim``, H, W).
        ``dim`` corresponds to ``C`` in the figure.
    :param list num_blocks: number of ``TransformerBlock`` for each level of scale in the encoder-decoder stage with a total of 4-level of scales.
        ``num_blocks = [L1, L2, L3, L4]`` with L1 ≤ L2 ≤ L3 ≤ L4.
    :param int num_refinement_blocks: number of ``TransformerBlock`` in the refinement stage after the decoder stage.
        Corresponds to ``Lr`` in the figure.
    :param list heads: number of heads in ``TransformerBlock`` for each level of scale in the encoder-decoder stage and in the refinement stage.
        At same scale, all `TransformerBlock` have the same number of heads. The number of heads for the refinement block is ``heads[0]``.
    :param float ffn_expansion_factor: corresponds to :math:`\eta` in GDFN.
    :param bool bias: Add bias or not in each of the Attention and Feedforward layers inside of the ``TransformerBlock``.
    :param str LayerNorm_type: Add bias or not in each of the LayerNorm inside of the ``TransformerBlock``.
        ``LayerNorm_type = 'BiasFree' / 'WithBias'``.
    :param bool dual_pixel_task: Should be true if dual-pixel defocus deblurring is enabled, false for single-pixel deblurring and other tasks.
    :param None, torch.device device: Instruct our module to be either on cpu or on gpu. Default to ``None``, which suggests working on cpu.
    :param None, str pretrained: Default to ``'denoising'``.
        If ``pretrained = 'denoising' / 'denoising_gray' / 'denoising_color' / 'denoising_real' / 'deraining' / 'defocus_deblurring'``,
        will download weights from the HuggingFace Hub.
        If ``pretrained = '\*.pth'``, will load weights from a local pth file.
    :param bool train: training or testing mode.

    .. note::
        To obtain good performance on a broad range of noise levels, even with limited noise levels during training, it is recommended to remove all additive constants by setting :
        ``LayerNorm_type='BiasFree'`` and ``bias=False``
        (`Robust And Interpretable Bling Image Denoising Via Bias-Free Convolutional Neural Networks <https://arxiv.org/abs/1906.05478>`_).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: List[int] = [4, 6, 6, 8],
        num_refinement_blocks: int = 4,
        heads: List[int] = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "BiasFree",
        dual_pixel_task: bool = False,
        pretrained: Optional[str] = "denoising",
        device: Optional[torch.device] = None,
    ) -> None:
        super(Restormer, self).__init__()

        # stores the filename of pretrained weights, used later in the code to download the pth file from the HuggingFace Hub
        weights_pth_filename = None
        # When loading pretrained weights from HuggingFace Hub, we check if our model is compatible with the weights.
        if pretrained is not None:
            if pretrained == "denoising":
                self.is_standard_denoising_network(
                    in_channels,
                    out_channels,
                    dim,
                    num_blocks,
                    num_refinement_blocks,
                    heads,
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    dual_pixel_task,
                )
                if in_channels == 1:
                    weights_pth_filename = "gaussian_gray_denoising_blind.pth"
                elif in_channels == 3:
                    weights_pth_filename = "gaussian_color_denoising_blind.pth"
            elif "denoising_real" in pretrained:
                self.is_standard_denoising_network(
                    in_channels,
                    out_channels,
                    dim,
                    num_blocks,
                    num_refinement_blocks,
                    heads,
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    dual_pixel_task,
                )
                assert (
                    in_channels == 3
                ), f"Real denoising / EXPECTED in_channels == 3, INSTEAD of {in_channels}"
                weights_pth_filename = "real_denoising.pth"
            elif "denoising_gray" in pretrained:
                self.is_standard_denoising_network(
                    in_channels,
                    out_channels,
                    dim,
                    num_blocks,
                    num_refinement_blocks,
                    heads,
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    dual_pixel_task,
                )
                assert (
                    in_channels == 1
                ), f"Real denoising / EXPECTED in_channels == 1, INSTEAD of {in_channels}"
                weights_pth_filename = "gaussian_gray_denoising_blind.pth"
            elif "denoising_color" in pretrained:
                self.is_standard_denoising_network(
                    in_channels,
                    out_channels,
                    dim,
                    num_blocks,
                    num_refinement_blocks,
                    heads,
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    dual_pixel_task,
                )
                assert (
                    in_channels == 3
                ), f"Color denoising / EXPECTED in_channels == 3, INSTEAD of {in_channels}"
                weights_pth_filename = "gaussian_color_denoising_blind.pth"
            elif pretrained == "deraining":
                self.is_standard_deraining_network(
                    in_channels,
                    out_channels,
                    dim,
                    num_blocks,
                    num_refinement_blocks,
                    heads,
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    dual_pixel_task,
                )
                weights_pth_filename = "deraining.pth"
            elif pretrained == "defocus_deblurring":
                self.is_standard_deblurring_network(
                    in_channels,
                    out_channels,
                    dim,
                    num_blocks,
                    num_refinement_blocks,
                    heads,
                    ffn_expansion_factor,
                    bias,
                    LayerNorm_type,
                    dual_pixel_task,
                )
                if dual_pixel_task:
                    assert (
                        in_channels == 6
                    ), f"Dual defocus deblurring / EXPECTED in_channels == 6, INSTEAD of {in_channels}"
                    weights_pth_filename = "dual_pixel_defocus_deblurring.pth"
                else:
                    assert (
                        in_channels == 3
                    ), f"Single defocus deblurring / EXPECTED in_channels == 3, INSTEAD of {in_channels}"
                    weights_pth_filename = "single_image_defocus_deblurring.pth"

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(
            int(dim * 2**1)
        )  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2**1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(
            int(dim * 2**1),
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

        # we don't check if our model is a good fit for the weights from the .pth file
        if pretrained and pretrained.endswith(".pth") and os.path.exists(pretrained):
            print(f"Loading from local file {pretrained}")
            ckpt_restormer = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            self.load_state_dict(ckpt_restormer["params"], strict=True)
            self.eval()
        elif weights_pth_filename is not None:
            print(f"Loading from {weights_pth_filename}")
            url = get_weights_url(
                model_name="restormer", file_name=weights_pth_filename
            )
            ckpt_restormer = torch.hub.load_state_dict_from_url(
                url,
                map_location=lambda storage, loc: storage,
                file_name=weights_pth_filename,
            )
            self.load_state_dict(ckpt_restormer["params"], strict=True)
            self.eval()
        elif pretrained is not None:
            raise ValueError(f"pretrained value error, {pretrained}")

        if device is not None:
            self.to(device)

    def forward_restormer(self, x):
        r"""
        Run the Restormer network on the input image.

        The input shape is expected to be divisible by 8.

        :param torch.Tensor x: input image
        """
        # expected : x.shape = (B, C, H, W)
        assert (
            x.shape[-2] % 8 == 0 and x.shape[-1] % 8 == 0
        ), f"Image spatial dim is not divisible by 8. Spatial dim : ({x.shape[-2]},{x.shape[-1]})"

        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + x

        return out_dec_level1

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image
        """
        if self.training:
            out = self.forward_restormer(x)
        else:
            out = test_pad(self.forward_restormer, x, modulo=8)
        return out

    def is_standard_denoising_network(
        self,
        in_channels,
        out_channels,
        dim,
        num_blocks,
        num_refinement_blocks,
        heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        dual_pixel_task,
    ):
        """Check if model params are the params used to pre-trained the standard network for denoising."""
        assert (
            in_channels == 1 or in_channels == 3
        ), f"Standard denoising / EXPECTED in_channels == 1 or 3, INSTEAD of {in_channels}"
        assert (
            out_channels == in_channels
        ), f"Standard denoising / EXPECTED out_channels == in_channels, INSTEAD of {out_channels}"
        self._is_standard_network(
            dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias
        )
        assert (
            LayerNorm_type == "BiasFree"
        ), f"Standard denoising / EXPECTED LayerNorm_type == 'BiasFree', INSTEAD of {LayerNorm_type}"
        assert (
            dual_pixel_task == False
        ), f"Standard denoising / EXPECTED dual_pixel_task == False, INSTEAD of {dual_pixel_task}"

    def is_standard_deraining_network(
        self,
        in_channels,
        out_channels,
        dim,
        num_blocks,
        num_refinement_blocks,
        heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        dual_pixel_task,
    ):
        """Check if model params are the params used to pre-trained the standard network for deraining."""
        assert (
            in_channels == 3
        ), f"Standard deraining / EXPECTED in_channels == 3, INSTEAD of {in_channels}"
        assert (
            out_channels == 3
        ), f"Standard deraining / EXPECTED out_channels == 3, INSTEAD of {out_channels}"
        self._is_standard_network(
            dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias
        )
        assert (
            LayerNorm_type == "WithBias"
        ), f"Standard deraining / EXPECTED LayerNorm_type == 'WithBias', INSTEAD of {LayerNorm_type}"
        assert (
            dual_pixel_task == False
        ), f"Standard deraining / EXPECTED dual_pixel_task == False, INSTEAD of {dual_pixel_task}"

    def is_standard_deblurring_network(
        self,
        in_channels,
        out_channels,
        dim,
        num_blocks,
        num_refinement_blocks,
        heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        dual_pixel_task,
    ):
        """Check if model params are the params used to pre-trained the standard network for deblurring."""
        assert (
            in_channels == 3 or in_channels == 6
        ), f"Standard deblurring / EXPECTED in_channels == 3 or 6, INSTEAD of {in_channels}"
        assert (
            out_channels == 3
        ), f"Standard deblurring / EXPECTED out_channels == 3, INSTEAD of {out_channels}"
        self._is_standard_network(
            dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias
        )
        assert (
            LayerNorm_type == "WithBias"
        ), f"Standard deblurring / EXPECTED LayerNorm_type == 'WithBias', INSTEAD of {LayerNorm_type}"

    def _is_standard_network(
        self, dim, num_blocks, num_refinement_blocks, heads, ffn_expansion_factor, bias
    ):
        """The pre-trained networks for denoising, for deraining, and for deblurring have some params with same values,
        so when trying to load the pre-trained weights from one of these networks, we check first that our model params
        have these values to avoid mismatch between our model and the weights.
        """
        assert (
            dim == 48
        ), f"Standard restormer architecture / EXPECTED dim == 48, INSTEAD of {dim}"
        assert num_blocks == [
            4,
            6,
            6,
            8,
        ], f"Standard restormer architecture / EXPECTED num_blocks == [4,6,6,8], INSTEAD of {num_blocks}"
        assert (
            num_refinement_blocks == 4
        ), f"Standard restormer architecture / EXPECTED num_refinement_blocks == 4, INSTEAD of {num_refinement_blocks}"
        assert heads == [
            1,
            2,
            4,
            8,
        ], f"Standard restormer architecture / EXPECTED heads == [1,2,4,8], INSTEAD of {heads}"
        assert (
            ffn_expansion_factor == 2.66
        ), f"Standard restormer architecture / EXPECTED ffn_expansion_factor == 2.66, INSTEAD of {ffn_expansion_factor}"
        assert (
            bias == False
        ), f"Standard restormer architecture / EXPECTED bias == False, INSTEAD of {bias}"


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # type: ignore

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # type: ignore

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)
