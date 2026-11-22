# Code Implementation of the MambaIR Model
# This is released under Apache 2.0 at https://github.com/csguoh/MambaIR/
# And has been modified to fit the deepinv api.
from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .base import Denoiser, Reconstructor
from .swinir import PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep
from .utils import DropPath, load_state_dict_from_url, trunc_normal_

if TYPE_CHECKING:
    from deepinv.physics import Physics


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    :param int num_feat: Channel number of intermediate features.
    :param int squeeze_factor: Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(
        self,
        num_feat: int,
        is_light_sr: bool = False,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
    ) -> None:
        super().__init__()
        if is_light_sr:  # we use dilated-conv & DWConv for lightSR for a large ERF
            compress_ratio = 2
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 1, 1, 0),
                nn.Conv2d(
                    num_feat // compress_ratio,
                    num_feat // compress_ratio,
                    3,
                    1,
                    1,
                    groups=num_feat // compress_ratio,
                ),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 1, 1, 0),
                nn.Conv2d(
                    num_feat, num_feat, 3, 1, padding=2, groups=num_feat, dilation=2
                ),
                ChannelAttention(num_feat, squeeze_factor),
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                ChannelAttention(num_feat, squeeze_factor),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class SS2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        except ImportError as e:
            raise ImportError(
                "mamba_ssm is required to use MambaIR. "
                "Install it with `pip install mamba-ssm` (requires CUDA) "
            ) from e

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=4, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs: Any,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(
        d_state: int,
        d_inner: int,
        copies: int = 1,
        device: torch.device | str | None = None,
        merge: bool = True,
    ) -> nn.Parameter:
        from einops import repeat

        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        return A_log

    @staticmethod
    def D_init(
        d_inner: int,
        copies: int = 1,
        device: torch.device | str | None = None,
        merge: bool = True,
    ) -> nn.Parameter:
        from einops import repeat

        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        return D

    def forward_core(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat(
            [x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1
        )  # (1, 4, 192, 3136)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        expand: float = 2.0,
        is_light_sr: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate,
            **kwargs,
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = (
            x * self.skip_scale2
            + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous())
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """The Basic MambaIR Layer in one Residual State Space Group.

    :param int dim: Number of input channels.
    :param tuple[int, int] input_resolution: Input resolution.
    :param int depth: Number of blocks.
    :param float mlp_ratio: Ratio of mlp hidden dim to embedding dim.
    :param float | list[float] drop_path: Stochastic depth rate. Default: 0.0.
    :param type[nn.Module] norm_layer: Normalization layer. Default: nn.LayerNorm.
    :param downsample: Downsample layer at the end of the layer. Default: None.
    :param bool use_checkpoint: Whether to use checkpointing to save memory. Default: False.
    :param bool is_light_sr: Whether this is a lightweight SR variant. Default: False.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        drop_path: float | list[float] = 0.0,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: Callable[..., nn.Module] | None = None,
        use_checkpoint: bool = False,
        is_light_sr: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    d_state=d_state,
                    expand=self.mlp_ratio,
                    input_resolution=input_resolution,
                    is_light_sr=is_light_sr,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class MambaIR(Reconstructor, Denoiser):
    r"""MambaIR restoration network.

    The MambaIR model was introduced by :footcite:t:`guo2024mambair`. This code is adapted from the official
    implementation by the authors.

    MambaIR can be used for:

    - **Image denoising**: set ``upscale=1`` and ``upsampler=''``. The model then acts as a
      :class:`Denoiser <deepinv.models.Denoiser>`
    - **Super-resolution**: set ``upscale=2/3/4`` with ``upsampler='pixelshuffle'`` (classical) or
      ``upsampler='pixelshuffledirect'`` (lightweight).

    .. note::
        This model requires the optional ``mamba-ssm`` package, which can be installed with
        ``pip install mamba-ssm`` (requires a CUDA-capable GPU and a compatible PyTorch build).

    :param int | tuple[int, int] img_size: Input image size. Default 64.
    :param int | tuple[int, int] patch_size: Patch size. Default: 1.
    :param int in_chans: Number of input image channels. Default: 3.
    :param int embed_dim: Patch embedding dimension. Default: 96.
    :param tuple[int, ...] depths: Depth of each RSSG. Default: (6, 6, 6, 6).
    :param float drop_rate: Dropout rate. Default: 0.
    :param int d_state: Number of hidden state in the state space model. Default: 16.
    :param float mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 2.0.
    :param float drop_path_rate: Stochastic depth rate. Default: 0.1.
    :param type[nn.Module] norm_layer: Normalization layer. Default: nn.LayerNorm.
    :param bool patch_norm: If True, add normalization after patch embedding. Default: True.
    :param bool use_checkpoint: Whether to use checkpointing to save memory. Default: False.
    :param int upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising. Default: 2.
    :param float img_range: Image range. 1. or 255. Default: 1.0.
    :param str upsampler: The reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'' (empty for denoising). Default: ''.
    :param str resi_connection: The convolutional block before residual connection. '1conv'/'3conv'. Default: '1conv'.
    :param str | None pretrained: Use a pretrained network. If ``pretrained=None``, the weights will be
        initialized at random using PyTorch's default initialization. Otherwise, ``pretrained`` selects
        which pretrained weights to download from the authors' HuggingFace repository
        (https://huggingface.co/cguoh/MambaIR/tree/main/MambaIRv1_ckpt). Only the arguments that differ
        from their defaults are listed below:

        - ``pretrained='denoising_15'`` / ``'denoising_25'`` / ``'denoising_50'``: color Gaussian denoising
          at noise level 15/25/50 (0-255 scale). Non-default args: ``upscale=1``, ``img_size=128``,
          ``embed_dim=180``, ``depths=(6, 6, 6, 6, 6, 6)``, ``mlp_ratio=1.2``.
        - ``pretrained='sr_x2'`` / ``'sr_x3'`` / ``'sr_x4'``: classical super-resolution. Non-default args:
          ``upscale=2/3/4``, ``embed_dim=180``, ``depths=(6, 6, 6, 6, 6, 6)``, ``upsampler='pixelshuffle'``.
        - ``pretrained='lightsr_x2'`` / ``'lightsr_x3'`` / ``'lightsr_x4'``: lightweight super-resolution.
          Non-default args: ``upscale=2/3/4``, ``embed_dim=60``, ``d_state=10``, ``mlp_ratio=1.2``,
          ``upsampler='pixelshuffledirect'``.

        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights file (``.pth``).
        Default: ``None``.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 64,
        patch_size: int | tuple[int, int] = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        drop_rate: float = 0.0,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "",
        resi_connection: str = "1conv",
        pretrained: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler == "pixelshuffledirect" else False
        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 6-layer
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state=d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr=self.is_light_sr,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

        if pretrained is not None:
            base_url = (
                "https://huggingface.co/cguoh/MambaIR/resolve/main/MambaIRv1_ckpt/"
            )
            emsg = "pretrained={0!r} requires {1}={2!r}, got {1}={3!r}"
            download_keys = (
                "denoising_15",
                "denoising_25",
                "denoising_50",
                "sr_x2",
                "sr_x3",
                "sr_x4",
                "lightsr_x2",
                "lightsr_x3",
                "lightsr_x4",
            )
            if pretrained in download_keys:
                if in_chans != 3:  # pragma: no cover
                    raise ValueError(
                        f"pretrained={pretrained!r} requires in_chans=3, got {in_chans}"
                    )
                if img_range != 1.0:  # pragma: no cover
                    raise ValueError(
                        emsg.format(pretrained, "img_range", 1.0, img_range)
                    )
                if resi_connection != "1conv":  # pragma: no cover
                    raise ValueError(
                        emsg.format(
                            pretrained, "resi_connection", "1conv", resi_connection
                        )
                    )

                if pretrained.startswith("denoising_"):
                    level = int(pretrained.rsplit("_", 1)[1])
                    if upscale != 1:  # pragma: no cover
                        raise ValueError(emsg.format(pretrained, "upscale", 1, upscale))
                    if img_size != 128:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "img_size", 128, img_size)
                        )
                    if embed_dim != 180:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "embed_dim", 180, embed_dim)
                        )
                    if list(depths) != [6, 6, 6, 6, 6, 6]:  # pragma: no cover
                        raise ValueError(
                            emsg.format(
                                pretrained, "depths", [6, 6, 6, 6, 6, 6], list(depths)
                            )
                        )
                    if d_state != 16:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "d_state", 16, d_state)
                        )
                    if mlp_ratio != 1.2:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "mlp_ratio", 1.2, mlp_ratio)
                        )
                    if upsampler != "":  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "upsampler", "", upsampler)
                        )
                    weights_url = base_url + f"ColorDN_MambaIR_level{level}.pth"
                elif pretrained.startswith("sr_"):
                    scale = int(pretrained.rsplit("_", 1)[1][1:])
                    if upscale != scale:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "upscale", scale, upscale)
                        )
                    if img_size != 64:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "img_size", 64, img_size)
                        )
                    if embed_dim != 180:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "embed_dim", 180, embed_dim)
                        )
                    if list(depths) != [6, 6, 6, 6, 6, 6]:  # pragma: no cover
                        raise ValueError(
                            emsg.format(
                                pretrained, "depths", [6, 6, 6, 6, 6, 6], list(depths)
                            )
                        )
                    if d_state != 16:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "d_state", 16, d_state)
                        )
                    if mlp_ratio != 2.0:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "mlp_ratio", 2.0, mlp_ratio)
                        )
                    if upsampler != "pixelshuffle":  # pragma: no cover
                        raise ValueError(
                            emsg.format(
                                pretrained, "upsampler", "pixelshuffle", upsampler
                            )
                        )
                    weights_url = base_url + f"MambaIR_classicSRx{scale}.pth"
                elif pretrained.startswith("lightsr_"):
                    scale = int(pretrained.rsplit("_", 1)[1][1:])
                    if upscale != scale:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "upscale", scale, upscale)
                        )
                    if img_size != 64:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "img_size", 64, img_size)
                        )
                    if embed_dim != 60:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "embed_dim", 60, embed_dim)
                        )
                    if list(depths) != [6, 6, 6, 6]:  # pragma: no cover
                        raise ValueError(
                            emsg.format(
                                pretrained, "depths", [6, 6, 6, 6], list(depths)
                            )
                        )
                    if d_state != 10:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "d_state", 10, d_state)
                        )
                    if mlp_ratio != 1.2:  # pragma: no cover
                        raise ValueError(
                            emsg.format(pretrained, "mlp_ratio", 1.2, mlp_ratio)
                        )
                    if upsampler != "pixelshuffledirect":  # pragma: no cover
                        raise ValueError(
                            emsg.format(
                                pretrained, "upsampler", "pixelshuffledirect", upsampler
                            )
                        )
                    weights_url = base_url + f"MambaIR_lightSR{scale}.pth"
                else:  # pragma: no cover
                    raise ValueError(
                        f"Unknown pretrained key {pretrained!r}. "
                        f"Expected one of {download_keys} or a path to a .pth file."
                    )
                pretrained_weights = load_state_dict_from_url(
                    weights_url, map_location=lambda storage, loc: storage
                )
            else:
                pretrained_weights = torch.load(
                    pretrained, map_location=lambda storage, loc: storage
                )
            # Unwrap checkpoint: official MambaIR weights use a "params" key,
            # deepinv trainer checkpoints use "state_dict". If neither is
            # present, assume the file is already a raw state_dict.
            for key in ("params", "state_dict"):
                if key in pretrained_weights:
                    pretrained_weights = pretrained_weights[key]
                    break

            self.load_state_dict(pretrained_weights, strict=True)
            self.eval()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(
        self,
        x: torch.Tensor,
        physics: Physics | None = None,
        sigma: float | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        r"""
        Run the model on the input image.

        :param torch.Tensor x: input image, of shape B, C, H, W.
        :param deepinv.physics.Physics physics: forward operator (not used).
        :param float, torch.Tensor sigma: noise level (not used).
        :return: (:class:`torch.Tensor`) restored image.
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    :param int dim: Number of input channels.
    :param tuple[int, int] input_resolution: Input resolution.
    :param int depth: Number of blocks.
    :param int d_state: Number of hidden state in the state space model. Default: 16.
    :param float mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.0.
    :param float | list[float] drop_path: Stochastic depth rate. Default: 0.0.
    :param type[nn.Module] norm_layer: Normalization layer. Default: nn.LayerNorm.
    :param downsample: Downsample layer at the end of the layer. Default: None.
    :param bool use_checkpoint: Whether to use checkpointing to save memory. Default: False.
    :param int | tuple[int, int] | None img_size: Input image size.
    :param int | tuple[int, int] | None patch_size: Patch size.
    :param str resi_connection: The convolutional block before residual connection. Default: '1conv'.
    :param bool is_light_sr: Whether this is a lightweight SR variant. Default: False.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: Callable[..., nn.Module] | None = None,
        use_checkpoint: bool = False,
        img_size: int | tuple[int, int] | None = None,
        patch_size: int | tuple[int, int] | None = None,
        resi_connection: str = "1conv",
        is_light_sr: bool = False,
    ) -> None:
        super().__init__()

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
        )

        # build the last conv layer in each residual state space group
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return (
            self.patch_embed(
                self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
            )
            + x
        )
