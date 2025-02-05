
# Unet copied from "A Variational Perspective on Diffusion-Based Generative Models and Score Matching", NeurIPS 2021, Huang et al
# https://github.com/CW-Huang/sdeflow-light
# which is modified from the "Denoising Diffusion Probabilistic Models", Ho et al

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


from .base import Denoiser


def get_weights_from_drive(pretrained_name):
    if pretrained_name == 'celeba':
        drive_id = '1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6'
    elif pretrained_name == 'afhq_cat':
        drive_id = '1FpD3cYpgtM8-KJ3Qk48fcjtr1Ne_IMOF'
    return f"https://drive.google.com/uc?id={drive_id}"


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


def group_norm(out_ch):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=out_ch,
        eps=1e-6,
        affine=True)
# TODO: change init


class FlowUNet(Denoiser):
    def __init__(self,
                 input_channels,
                 input_height,
                 ch=32,
                 output_channels=None,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks=6,
                 attn_resolutions=(16, 8),
                 dropout=0.,
                 resamp_with_conv=True,
                 act=Swish(),
                 normalize=group_norm,
                 pretrained="download",
                 device='cuda'
                 ):
        super().__init__(device=device)
        self.input_channels = input_channels
        self.input_height = input_height
        self.ch = ch
        self.output_channels = output_channels = input_channels if output_channels is None else output_channels
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv
        self.act = act
        self.normalize = normalize
        self.device = device

        # init
        self.num_resolutions = num_resolutions = len(ch_mult)
        in_ht = input_height
        in_ch = input_channels
        temb_ch = ch * 4
        assert in_ht % 2 ** (num_resolutions -
                             1) == 0, "input_height doesn't satisfy the condition"

        # Timestep embedding
        self.temb_net = TimestepEmbedding(
            embedding_dim=ch,
            hidden_dim=temb_ch,
            output_dim=temb_ch,
            act=act,
        )

        # Downsampling
        self.begin_conv = conv2d(in_ch, ch)
        unet_chs = [ch]
        in_ht = in_ht
        in_ch = ch
        down_modules = []
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch,
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize,
                )
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(
                        out_ch, normalize=normalize)
                unet_chs += [out_ch]
                in_ch = out_ch
            # Downsample
            if i_level != num_resolutions - 1:
                block_modules['{}b_downsample'.format(i_level)] = downsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht //= 2
                unet_chs += [out_ch]
            # convert list of modules to a module list, and append to a list
            down_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.down_modules = nn.ModuleList(down_modules)

        # Middle
        mid_modules = []
        mid_modules += [ResidualBlock(in_ch,
                                      temb_ch=temb_ch,
                                      out_ch=in_ch,
                                      dropout=dropout,
                                      act=act,
                                      normalize=normalize)]
        mid_modules += [SelfAttention(in_ch, normalize=normalize)]
        mid_modules += [ResidualBlock(in_ch,
                                      temb_ch=temb_ch,
                                      out_ch=in_ch,
                                      dropout=dropout,
                                      act=act,
                                      normalize=normalize)]
        self.mid_modules = nn.ModuleList(mid_modules)

        # Upsampling
        up_modules = []
        for i_level in reversed(range(num_resolutions)):
            # Residual blocks for this resolution
            block_modules = {}
            out_ch = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block_modules['{}a_{}a_block'.format(i_level, i_block)] = \
                    ResidualBlock(
                        in_ch=in_ch + unet_chs.pop(),
                        temb_ch=temb_ch,
                        out_ch=out_ch,
                        dropout=dropout,
                        act=act,
                        normalize=normalize)
                if in_ht in attn_resolutions:
                    block_modules['{}a_{}b_attn'.format(i_level, i_block)] = SelfAttention(
                        out_ch, normalize=normalize)
                in_ch = out_ch
            # Upsample
            if i_level != 0:
                block_modules['{}b_upsample'.format(i_level)] = upsample(
                    out_ch, with_conv=resamp_with_conv)
                in_ht *= 2
            # convert list of modules to a module list, and append to a list
            up_modules += [nn.ModuleDict(block_modules)]
        # conver to a module list
        self.up_modules = nn.ModuleList(up_modules)
        assert not unet_chs

        # End
        self.end_conv = nn.Sequential(
            normalize(in_ch),
            self.act,
            conv2d(in_ch, output_channels, init_scale=0.),
        )

        if pretrained is not None:
            if pretrained == "download":
                name = "celeba"
                # TODO: handle different datasets
                # TODO: check architecture
                url = get_weights_from_drive(name)
                # ckpt = torch.hub.load_state_dict_from_url(
                #     url, map_location=lambda storage, loc: storage, file_name=name
                # )
                # TODO: fix import from drive
                import gdown
                gdown.download(url,  './model_final_celeba.pt')
                ckpt = torch.load('./model_final_celeba.pt',
                                  map_location=torch.device(self.device))
            else:
                ckpt = torch.load('./model_final_celeba.pt',
                                  map_location=torch.device(self.device))
                # ckpt = torch.load(
                #     pretrained, map_location=lambda storage, loc: storage)

            self.load_state_dict(ckpt, strict=True)
            self.to(self.device)
            self.eval()

    # noinspection PyMethodMayBeStatic
    def _compute_cond_module(self, module, x, temp):
        for m in module:
            x = m(x, temp)
        return x

    # noinspection PyArgumentList,PyShadowingNames
    def forward(self, x, temp):
        # Init
        B, C, H, W = x.size()

        # Timestep embedding
        temb = self.temb_net(temp)
        assert list(temb.shape) == [B, self.ch * 4]

        # Downsampling
        hs = [self.begin_conv(x)]
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            block_modules = self.down_modules[i_level]
            for i_block in range(self.num_res_blocks):
                resnet_block = block_modules['{}a_{}a_block'.format(
                    i_level, i_block)]
                h = resnet_block(hs[-1], temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(
                        i_level, i_block)]
                    h = attn_block(h, temb)
                hs.append(h)
            # Downsample
            if i_level != self.num_resolutions - 1:
                downsample = block_modules['{}b_downsample'.format(i_level)]
                hs.append(downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self._compute_cond_module(self.mid_modules, h, temb)

        # Upsampling
        for i_idx, i_level in enumerate(reversed(range(self.num_resolutions))):
            # Residual blocks for this resolution
            block_modules = self.up_modules[i_idx]
            for i_block in range(self.num_res_blocks + 1):
                resnet_block = block_modules['{}a_{}a_block'.format(
                    i_level, i_block)]
                h = resnet_block(torch.cat([h, hs.pop()], axis=1), temb)
                if h.size(2) in self.attn_resolutions:
                    attn_block = block_modules['{}a_{}b_attn'.format(
                        i_level, i_block)]
                    h = attn_block(h, temb)
            # Upsample
            if i_level != 0:
                upsample = block_modules['{}b_upsample'.format(i_level)]
                h = upsample(h)
        assert not hs

        # End
        h = self.end_conv(h)
        assert list(h.size()) == [x.size(
            0), self.output_channels, x.size(2), x.size(3)]
        return h


def upsample(in_ch, with_conv):
    up = nn.Sequential()
    up.add_module('up_nn', nn.Upsample(scale_factor=2, mode='nearest'))
    if with_conv:
        up.add_module('up_conv', conv2d(
            in_ch, in_ch, kernel_size=(3, 3), stride=1))
    return up


def downsample(in_ch, with_conv):
    if with_conv:
        down = conv2d(in_ch, in_ch, kernel_size=(3, 3), stride=2)
    else:
        down = nn.AvgPool2d(2, 2)
    return down


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_ch,
            temb_ch,
            out_ch=None,
            conv_shortcut=False,
            dropout=0.,
            normalize=group_norm,
            act=Swish()):
        super().__init__()
        self.in_ch = in_ch
        self.temb_ch = temb_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.act = act

        self.temb_proj = dense(temb_ch, out_ch)
        self.norm1 = normalize(
            in_ch) if normalize is not None else nn.Identity()
        self.conv1 = conv2d(in_ch, out_ch)
        self.norm2 = normalize(
            out_ch) if normalize is not None else nn.Identity()
        self.dropout = nn.Dropout2d(
            p=dropout) if dropout > 0. else nn.Identity()
        self.conv2 = conv2d(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = conv2d(in_ch, out_ch)
            else:
                self.shortcut = conv2d(
                    in_ch, out_ch, kernel_size=(1, 1), padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        # forward conv1
        h = x
        h = self.act(self.norm1(h))
        h = self.conv1(h)

        # add in timestep embedding
        h = h + self.temb_proj(self.act(temb))[:, :, None, None]

        # forward conv2
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # shortcut
        x = self.shortcut(x)

        # combine and return
        assert x.shape == h.shape
        return x + h


class SelfAttention(nn.Module):
    """
    copied modified from https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py#L29
    copied modified from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py#L66
    """

    def __init__(self, in_channels, normalize=group_norm):
        super().__init__()
        self.in_channels = in_channels
        self.attn_q = conv2d(in_channels, in_channels,
                             kernel_size=1, stride=1, padding=0)
        self.attn_k = conv2d(in_channels, in_channels,
                             kernel_size=1, stride=1, padding=0)
        self.attn_v = conv2d(in_channels, in_channels,
                             kernel_size=1, stride=1, padding=0)
        self.proj_out = conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            init_scale=0.)
        self.softmax = nn.Softmax(dim=-1)
        if normalize is not None:
            self.norm = normalize(in_channels)
        else:
            self.norm = nn.Identity()

    # noinspection PyUnusedLocal
    def forward(self, x, temp=None):
        """ t is not used """
        _, C, H, W = x.size()

        h = self.norm(x)
        q = self.attn_q(h).view(-1, C, H * W)
        k = self.attn_k(h).view(-1, C, H * W)
        v = self.attn_v(h).view(-1, C, H * W)

        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        attn = self.softmax(attn)

        h = torch.bmm(v, attn.permute(0, 2, 1))
        h = h.view(-1, C, H, W)
        h = self.proj_out(h)

        assert h.shape == x.shape
        return x + h


def _calculate_correct_fan(tensor, mode):
    """
    copied and modified from https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L337
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out', 'fan_avg']
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(
                mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    r"""Fills the input `Tensor` with values according to the method
    desrcibed in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: multiplier to the dispersion
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in')
    """
    fan = _calculate_correct_fan(tensor, mode)
    # gain = calculate_gain(nonlinearity, a)
    var = gain / max(1., fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0 * var)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def variance_scaling_init_(tensor, scale):
    return kaiming_uniform_(
        tensor,
        gain=1e-10 if scale == 0 else scale,
        mode='fan_avg')


def dense(in_channels, out_channels, init_scale=1.):
    lin = nn.Linear(in_channels, out_channels)
    variance_scaling_init_(lin.weight, scale=init_scale)
    nn.init.zeros_(lin.bias)
    return lin


def conv2d(
        in_planes,
        out_planes,
        kernel_size=(
            3,
            3),
    stride=1,
    dilation=1,
    padding=1,
    bias=True,
    padding_mode='zeros',
        init_scale=1.):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode=padding_mode)
    variance_scaling_init_(conv.weight, scale=init_scale)
    if bias:
        nn.init.zeros_(conv.bias)
    return conv


def get_sinusoidal_positional_embedding(
        timesteps: torch.LongTensor,
        embedding_dim: int):
    """
    Copied and modified from
        https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90

    From Fairseq in
        https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py#L15
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the desrciption in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.size()) == 1
    timesteps = timesteps.to(torch.get_default_dtype())
    device = timesteps.device

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(
        half_dim, dtype=torch.float, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # bsz x embd
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), "constant", 0)
    assert list(emb.size()) == [timesteps.size(0), embedding_dim]
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=Swish()):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb
