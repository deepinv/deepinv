# This file is taken (with only mild modifications) from the PanFormer repository:
# https://github.com/zhysora/PanFormer/blob/main/models/panformer.py &
# https://github.com/zhysora/PanFormer/blob/main/models/common/modules.py
# -----------------------------------------------------------------------------------
# PanFormer: a Transformer Based Model for Pan-sharpening, https://arxiv.org/abs/2203.02916
# Written by: Zhou, Huanyu & Liu, Qingjie & Wang, Yunhong
# -----------------------------------------------------------------------------------

import numpy as np
import torch
from torch import nn, einsum
from typing import Callable, Any

# ------------ Replacement "rearrange" function ----------
# PanFormer code relies on one function from the einops library, not supported by deepinv

def rearrange(tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
    """
    Replacement of einops "rearrange" function only using Pytorch. 
    Only handles patterns needed in the rest of the file.

    Args:
        tensor (torch.Tensor): to be reshaped and permuted.
        pattern (str): einops string pattern (4 supported)
        **kwargs: Specific dimensions
            h1 (int): height dimension 
            h2 (int): height dimension 2
            h (int): number of attention heads
            w_h (int): window height
            w_w (int): window width
            nw_h (int): number of windows along height
            nw_w (int): number of windows along width
    """
    pattern_clean = pattern.replace(" ", "")

    if pattern_clean == "(h1w1)(h2w2)->h1w1h2w2":
        if "h1" not in kwargs:
            raise ValueError("Required argument 'h1'.")
        
        h1 = kwargs["h1"]
        h2 = kwargs.get("h2", h1)
        
        if tensor.shape[0] % h1 != 0 or tensor.shape[1] % h2 != 0:
            raise ValueError(f"Dimensions {tensor.shape[:2]} not divisible h1={h1}, h2={h2}")
            
        w1 = tensor.shape[0] // h1
        w2 = tensor.shape[1] // h2
        return tensor.view(h1, w1, h2, w2)

    elif pattern_clean == "h1w1h2w2->(h1w1)(h2w2)":
        h1, w1, h2, w2 = tensor.shape
        return tensor.contiguous().view(h1 * w1, h2 * w2)

    elif pattern_clean == "b(nw_hw_h)(nw_ww_w)(hd)->bh(nw_hnw_w)(w_hw_w)d":
        b, H, W, total_dim = tensor.shape
        
        for key in ["h", "w_h", "w_w"]:
            if key not in kwargs:
                raise ValueError(f"Required argument: {key}.")
                
        h = kwargs["h"]
        w_h = kwargs["w_h"]
        w_w = kwargs["w_w"]
        
        if H % w_h != 0 or W % w_w != 0 or total_dim % h != 0:
            raise ValueError("Spatial or channel dimensions are not multiple of the window or head sizes")
            
        nw_h = H // w_h
        nw_w = W // w_w
        d = total_dim // h

        x = tensor.view(b, nw_h, w_h, nw_w, w_w, h, d)
        x = x.permute(0, 5, 1, 3, 2, 4, 6).contiguous()
        return x.view(b, h, nw_h * nw_w, w_h * w_w, d)

    elif pattern_clean == "bh(nw_hnw_w)(w_hw_w)d->b(nw_hw_h)(nw_ww_w)(hd)":
        b, h, num_windows, window_area, d = tensor.shape
        #
        for key in ["w_h", "w_w", "nw_h", "nw_w"]:
            if key not in kwargs:
                raise ValueError(f"Required argument: {key}.")
        
        w_h = kwargs["w_h"]
        w_w = kwargs["w_w"]
        nw_h = kwargs["nw_h"]
        nw_w = kwargs["nw_w"]

        if nw_h * nw_w != num_windows or w_h * w_w != window_area:
            raise ValueError(f"Dimension mismatch: nw_h*nw_w={nw_h*nw_w} (expected {num_windows})")

        x = tensor.view(b, h, nw_h, nw_w, w_h, w_w, d)
        x = x.permute(0, 2, 4, 3, 5, 1, 6).contiguous()
        return x.view(b, nw_h * w_h, nw_w * w_w, h * d)

    else:
        raise NotImplementedError(f"Unsupported pattern: {pattern}")



# ----------- PanFormer ---------------
# Code from cited repository

def conv3x3(in_channels: int, out_channels:int, stride: int=1, padding: int=1, *args, **kwargs)->nn.Conv2d:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                     stride=stride, padding=padding, *args, **kwargs)

class CyclicShift(nn.Module):
    """
    Shifts a feature map circularly.
    """
    def __init__(self, displacement:int):
        """
        Args:
            displacement: number of pixels to shift along both height and width.
        """
        super().__init__()
        self.displacement = displacement

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

class Residual(nn.Module):
    """
    Wrapper for (x+f(x)), f a function.
    """
    def __init__(self, fn:Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x:torch.Tensor, **kwargs)->torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Wrapper for layer normalization
    """
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x:torch.Tensor, **kwargs)->torch.Tensor:
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    """
    MLP with GELU activation
    """
    def __init__(self, dim:int, hidden_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.net(x)


def create_mask(window_size:int, displacement:int, upper_lower:bool, left_right:bool)->torch.Tensor:
    """
    Creates an attention mask to prevent information from leaking across boundaries
    Will be used when CyclicShift is applied
    """
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size:int)->torch.Tensor:
    """
    Computes the 2D distances between pixels within a window
    """
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


    
class WindowAttention(nn.Module):
    """
    Layer that computes attention within window
    """
    def __init__(self, dim:int, heads:int, head_dim:int, shifted:bool, window_size:int, relative_pos_embedding: bool, cross_attn:bool):
        """
        Args:
            dim: input channel dimension
            heads: number of attention heads
            head_dim: dimension of one attention head
            shifted: to apply shift
            window_size: spatial size of window
            relative_pos_embedding: to inject spatial distance biases into the attention weights
            cross_attn: to activate cross attention between two inputs x and y
        """
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.cross_attn = cross_attn

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        if not self.cross_attn:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x:torch.Tensor, y:torch.Tensor=None)->torch.Tensor:
        if self.shifted:
            x = self.cyclic_shift(x)
            if self.cross_attn:
                y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads
        if not self.cross_attn:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head] * 3
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            qkv = (self.to_q(y), kv[0], kv[1])

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
            # [N, num_heads, num_win, win_area, hidden_dim/num_heads]


        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q * k / sqrt(d)

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head]
        out = self.to_out(out)
        # [N, H//downscaling_factor, W//downscaling_factor, dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    """
    Encapsulation of one complete Transformer layer
    """
    def __init__(self, dim:int, heads:int, head_dim:int, mlp_dim:int, shifted:bool, window_size:int, relative_pos_embedding:bool, cross_attn:bool):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x:torch.Tensor, y:torch.Tensor=None)->torch.Tensor:
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x


class PatchMerging(nn.Module):
    """
    Performs pooling: reduces spatial resolution while increasing channel count
    """
    def __init__(self, in_channels:int, out_channels:int, downscaling_factor:int):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]



class SwinModule(nn.Module):
    def __init__(self, in_channels:int, hidden_dimension:int, layers:int, downscaling_factor: int, num_heads:int, head_dim:int, window_size:int,
                 relative_pos_embedding:bool, cross_attn:bool):
        r"""
        Args:
            in_channels: number of input channels
            hidden_dimension: hidden layer dimension 
            layers: number of SwinBlocks; must be a multiple of 2, consisting of consecutive regular blocks and shifted blocks
            downscaling_factor: spatial downsampling factor for height and width
            num_heads: number of attention heads in the multi-head attention
            head_dim:  dimension of each individual attention head
            window_size: window size; the attention computation is restricted within this local window
        """
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
            ]))

    def forward(self, x:torch.Tensor, y: torch.Tensor=None)->torch.Tensor:
        if y is None:
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
        else:
            x = self.patch_partition(x)
            y = self.patch_partition(y)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x, y)
                x = shifted_block(x, y)
            return x.permute(0, 3, 1, 2)


class CrossSwinTransformer(nn.Module):
    def __init__(self, cfg:Any, n_feats:int=64, n_heads:int=4, head_dim:int=16, win_size:int=4,
                 n_blocks:int=3, cross_module:list=['pan', 'ms'], cat_feat:list=['pan', 'ms'], sa_fusion:bool=False):

        """
        Args:
            cfg: configuration namespace; attributes:
                - ms_chans(int): number of bands in the multispectral input
                - norm_input(bool): if True, output is clamped to [0,1], else clamped using bit depth
                - bit_depth(int): dynamic range of sensor
            n_feats, n_heads, head_dim, win_size, n_blocks: structural hyperparameters
            cross_module: fusion direction
            cat_feat: features to concatenate before reconstruction
            sa_fusion: to use self-attention fusion
        """
        super().__init__()
        self.cfg = cfg
        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion

        pan_encoder = [
            SwinModule(in_channels=1, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        ms_encoder = [
            SwinModule(in_channels=cfg.ms_chans, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]

        if 'ms' in self.cross_module:
            self.ms_cross_pan = nn.ModuleList()
            for _ in range(n_blocks):
                self.ms_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            ms_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                         downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                         window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(n_blocks):
                self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            pan_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                          downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                          window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        self.HR_tail = nn.Sequential(
            conv3x3(n_feats * len(cat_feat), n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats),
            nn.ReLU(True), conv3x3(n_feats, cfg.ms_chans))

        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

    def forward(self, pan: torch.Tensor, ms: torch.Tensor)->torch.Tensor:
        """
        Retrieves high resolution multispectral images
        Args:
            pan: high resolution panchromatic image (B, 1, H, W)
            ms: low resolutioon multispectral image (B, self.cfg.ms_chans, H/4, W/4)
        """
        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms)

        last_pan_feat = pan_feat
        last_ms_feat = ms_feat
        for i in range(self.n_blocks):
            if 'pan' in self.cross_module:
                pan_cross_ms_feat = self.pan_cross_ms[i](last_pan_feat, last_ms_feat)
            if 'ms' in self.cross_module:
                ms_cross_pan_feat = self.ms_cross_pan[i](last_ms_feat, last_pan_feat)
            if 'pan' in self.cross_module:
                last_pan_feat = pan_cross_ms_feat
            if 'ms' in self.cross_module:
                last_ms_feat = ms_cross_pan_feat

        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(last_pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(last_ms_feat)

        output = self.HR_tail(torch.cat(cat_list, dim=1))

        if self.cfg.norm_input:
            output = torch.clamp(output, 0, 1)
        else:
            output = torch.clamp(output, 0, 2 ** self.cfg.bit_depth - .5)

        return output

