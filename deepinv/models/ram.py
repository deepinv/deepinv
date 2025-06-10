import torch
import torch.nn as nn
import torch.nn.functional as F

import deepinv as dinv
from deepinv.physics import Physics, LinearPhysics, Downsampling
from deepinv.utils.tensorlist import TensorList

from huggingface_hub import hf_hub_download

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class RAM(nn.Module):
    r"""
    RAM model.

    This model (proposed in `this paper <https://arxiv.org/abs/2503.08915>`_) is a convolutional neural network (CNN)
    designed for image reconstruction tasks.

    :param in_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param device: Device to which the model should be moved. If None, the model will be created on the default device.
    :param pretrained: If True, the model will be initialized with pretrained weights.
    """

    def __init__(
        self,
        in_channels=[1, 2, 3],
        device=None,
        pretrained=True,
    ):
        super(RAM, self).__init__()

        nc = [64, 128, 256, 512]  # number of channels in the network
        self.in_channels = in_channels
        self.fact_realign = torch.nn.Parameter(torch.tensor([1.0], device=device))

        self.separate_head = isinstance(in_channels, list)

        if isinstance(in_channels, list):
            in_channels_first = []
            for i in range(len(in_channels)):
                in_channels_first.append(in_channels[i] + 2)

        # check if in_channels is a list
        self.m_head = InHead(in_channels_first, nc[0])

        self.m_down1 = BaseEncBlock(
            nc[0], nc[0], img_channels=in_channels, decode_upscale=1
        )
        self.m_down2 = BaseEncBlock(
            nc[1], nc[1], img_channels=in_channels, decode_upscale=2
        )
        self.m_down3 = BaseEncBlock(
            nc[2], nc[2], img_channels=in_channels, decode_upscale=4
        )
        self.m_body = BaseEncBlock(
            nc[3], nc[3], img_channels=in_channels, decode_upscale=8
        )
        self.m_up3 = BaseEncBlock(
            nc[2], nc[2], img_channels=in_channels, decode_upscale=4
        )
        self.m_up2 = BaseEncBlock(
            nc[1], nc[1], img_channels=in_channels, decode_upscale=2
        )
        self.m_up1 = BaseEncBlock(
            nc[0], nc[0], img_channels=in_channels, decode_upscale=1
        )

        self.pool1 = downsample_strideconv(nc[0], nc[1], bias=False, mode="2")
        self.pool2 = downsample_strideconv(nc[1], nc[2], bias=False, mode="2")
        self.pool3 = downsample_strideconv(nc[2], nc[3], bias=False, mode="2")
        self.up3 = upsample_convtranspose(nc[3], nc[2], bias=False, mode="2")
        self.up2 = upsample_convtranspose(nc[2], nc[1], bias=False, mode="2")
        self.up1 = upsample_convtranspose(nc[1], nc[0], bias=False, mode="2")

        self.m_tail = OutTail(nc[0], in_channels)

        # load pretrained weights from hugging face
        if pretrained:
            self.load_state_dict(
                torch.load(
                    hf_hub_download(repo_id="mterris/ram", filename="ram.pth.tar"),
                    map_location=device,
                )
            )

        if device is not None:
            self.to(device)

    def constant2map(self, value, x):
        r"""
        Converts a constant value to a map of the same size as the input tensor x.

        :params float value: constant value
        :params torch.Tensor x: input tensor
        """
        if isinstance(value, torch.Tensor):
            if value.ndim > 0:
                value_map = value.view(x.size(0), 1, 1, 1)
                value_map = value_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                value_map = torch.ones(
                    (x.size(0), 1, x.size(2), x.size(3)), device=x.device
                ) * value[None, None, None, None].to(x.device)
        else:
            value_map = (
                torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                * value
            )
        return value_map

    def base_conditioning(self, x, sigma, gamma):
        noise_level_map = self.constant2map(sigma, x)
        gamma_map = self.constant2map(gamma, x)
        return torch.cat((x, noise_level_map, gamma_map), 1)

    def realign_input(self, x, physics, y):
        r"""
        Realign the input x based on the measurements y and the physics model.
        Applies the proximity operator of the L2 norm with respect to the physics model.

        :params torch.Tensor x: Input tensor
        :params deepinv.physics.Physics physics: Physics model
        :params torch.Tensor y: Measurements
        """
        if hasattr(physics, "factor"):
            f = physics.factor
        elif hasattr(physics, "base") and hasattr(physics.base, "factor"):
            f = physics.base.factor
        elif (
            hasattr(physics, "base")
            and hasattr(physics.base, "base")
            and hasattr(physics.base.base, "factor")
        ):
            f = physics.base.base.factor
        else:
            f = 1.0

        sigma = 1e-6  # default value
        if hasattr(physics.noise_model, "sigma"):
            sigma = physics.noise_model.sigma
        if (
            hasattr(physics, "base")
            and hasattr(physics.base, "noise_model")
            and hasattr(physics.base.noise_model, "sigma")
        ):
            sigma = physics.base.noise_model.sigma
        if (
            hasattr(physics, "base")
            and hasattr(physics.base, "base")
            and hasattr(physics.base.base, "noise_model")
            and hasattr(physics.base.base.noise_model, "sigma")
        ):
            sigma = physics.base.base.noise_model.sigma

        if isinstance(y, TensorList):
            num = y[0].reshape(y[0].shape[0], -1).abs().mean(1)
        else:
            num = y.reshape(y.shape[0], -1).abs().mean(1)

        snr = num / (sigma + 1e-4)  # SNR equivariant
        gamma = 1 / (1e-4 + 1 / (snr * f**2))
        gamma = gamma[(...,) + (None,) * (x.dim() - 1)]
        model_input = physics.prox_l2(x, y, gamma=gamma * self.fact_realign)

        return model_input

    def forward_unet(self, x0, sigma=None, gamma=None, physics=None, y=None):
        r"""
        Forward pass of the UNet model.

        :params torch.Tensor x0: init image
        :params float sigma: Gaussian noise level
        :params float gamma: Poisson noise gain
        :params deepinv.physics.Physics physics: physics measurement operator
        :params torch.Tensor y: measurements
        """
        img_channels = x0.shape[1]
        physics = MultiScaleLinearPhysics(physics, x0.shape[-3:], device=x0.device)

        if self.separate_head and img_channels not in self.in_channels:
            raise ValueError(
                f"Input image has {img_channels} channels, but the network only have heads for {self.in_channels} channels."
            )

        if y is not None:
            x0 = self.realign_input(x0, physics, y)

        x0 = self.base_conditioning(x0, sigma, gamma)

        x1 = self.m_head(x0)

        x1_ = self.m_down1(x1, physics=physics, y=y, img_channels=img_channels, scale=0)
        x2 = self.pool1(x1_)

        x3_ = self.m_down2(x2, physics=physics, y=y, img_channels=img_channels, scale=1)
        x3 = self.pool2(x3_)

        x4_ = self.m_down3(x3, physics=physics, y=y, img_channels=img_channels, scale=2)
        x4 = self.pool3(x4_)

        x = self.m_body(x4, physics=physics, y=y, img_channels=img_channels, scale=3)

        x = self.up3(x + x4)
        x = self.m_up3(x, physics=physics, y=y, img_channels=img_channels, scale=2)

        x = self.up2(x + x3)
        x = self.m_up2(x, physics=physics, y=y, img_channels=img_channels, scale=1)

        x = self.up1(x + x2)
        x = self.m_up1(x, physics=physics, y=y, img_channels=img_channels, scale=0)

        x = self.m_tail(x + x1, img_channels)

        return x

    def forward(self, y=None, physics=None):
        r"""
        Reconstructs a signal estimate from measurements y
        :param torch.tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        """
        if physics is None:
            physics = dinv.physics.Denoising(
                noise_model=dinv.physics.GaussianNoise(sigma=0.0), device=y.device
            )

        x_temp = physics.A_adjoint(y)
        pad = (-x_temp.size(-2) % 8, -x_temp.size(-1) % 8)
        physics = Pad(physics, pad)

        x_in = physics.A_adjoint(y)

        sigma = (
            physics.noise_model.sigma if hasattr(physics.noise_model, "sigma") else 1e-3
        )
        gamma = (
            physics.noise_model.gain if hasattr(physics.noise_model, "gain") else 1e-3
        )

        out = self.forward_unet(x_in, sigma=sigma, gamma=gamma, physics=physics, y=y)

        out = physics.remove_pad(out)

        return out


### --------------- MODEL ---------------
class BaseEncBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        nb=4,
        img_channels=None,
        decode_upscale=None,
    ):
        super(BaseEncBlock, self).__init__()
        self.enc = nn.ModuleList(
            [
                ResBlock(
                    in_channels,
                    out_channels,
                    bias=bias,
                    img_channels=img_channels,
                    decode_upscale=decode_upscale,
                )
                for _ in range(nb)
            ]
        )

    def forward(self, x, physics=None, y=None, img_channels=None, scale=0):
        for i in range(len(self.enc)):
            x = self.enc[i](
                x, physics=physics, y=y, img_channels=img_channels, scale=scale
            )
        return x


def krylov_embeddings(y, p, factor, v=None, N=4, x_init=None):
    r"""
    Efficient Krylov subspace embedding computation with parallel processing.

    :params torch.Tensor y: Input tensor.
    :params p: An object with A and A_adjoint methods (linear operator).
    :params float factor: Scaling factor.
    :params torch.Tensor v: Precomputed values to subtract from Krylov sequence. Defaults to None.
    :params int N: Number of Krylov iterations. Defaults to 4.
    :params torch.Tensor x_init: Initial guess. Defaults to None.
    """

    if x_init is None:
        x = p.A_adjoint(y)
    else:
        x = x_init.clone()  # Extract the first img_channels

    norm = factor**2  # Precompute normalization factor
    AtA = lambda u: p.A_adjoint(p.A(u)) * norm  # Define the linear operator

    v = v if v is not None else torch.zeros_like(x)

    out = x.clone()
    # Compute Krylov basis
    x_k = x.clone()
    for i in range(N - 1):
        x_k = AtA(x_k) - v
        out = torch.cat([out, x_k], dim=1)

    return out


class MeasCondBlock(nn.Module):
    r"""
    Measurement conditioning block for the RAM model.

    :param out_channels: Number of output channels.
    :param img_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param decode_upscale: Upscaling factor for the decoding convolution.
    :param N: Number of Krylov iterations.
    :param depth_encoding: Depth of the encoding convolution.
    :param c_mult: Multiplier for the number of channels.
    """

    def __init__(
        self,
        out_channels=64,
        img_channels=None,
        decode_upscale=None,
        N=4,
        depth_encoding=1,
        c_mult=1,
    ):
        super(MeasCondBlock, self).__init__()

        self.separate_head = isinstance(img_channels, list)

        assert img_channels is not None, "decode_dimensions should be provided"
        assert decode_upscale is not None, "decode_upscale should be provided"

        self.N = N
        self.c_mult = c_mult
        self.relu_encoding = nn.ReLU(inplace=False)
        self.decoding_conv = Tails(
            out_channels, img_channels, depth=1, scale=1, bias=False, c_mult=self.c_mult
        )
        self.encoding_conv = Heads(
            img_channels,
            out_channels,
            depth=depth_encoding,
            scale=1,
            bias=False,
            c_mult=self.c_mult * N,
            c_add=N,
            relu_in=False,
            skip_in=True,
        )

        self.gain = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.gain_gradx = torch.nn.Parameter(torch.tensor([1e-2]), requires_grad=True)
        self.gain_grady = torch.nn.Parameter(torch.tensor([1e-2]), requires_grad=True)
        self.gain_pinvx = torch.nn.Parameter(torch.tensor([1e-2]), requires_grad=True)
        self.gain_pinvy = torch.nn.Parameter(torch.tensor([1e-2]), requires_grad=True)

    def forward(self, x, y, physics, img_channels=None, scale=1):
        physics.set_scale(scale)
        dec = self.decoding_conv(x, img_channels)
        factor = 2 ** (scale)
        meas_y = krylov_embeddings(y, physics, factor, N=self.N)
        meas_dec = krylov_embeddings(
            y, physics, factor, N=self.N, x_init=dec[:, :img_channels, ...]
        )
        for c in range(1, self.c_mult):
            meas_cur = krylov_embeddings(
                y,
                physics,
                factor,
                N=self.N,
                x_init=dec[:, img_channels * c : img_channels * (c + 1)],
            )
            meas_dec = torch.cat([meas_dec, meas_cur], dim=1)
        meas = torch.cat([meas_y, meas_dec], dim=1)
        cond = self.encoding_conv(meas)
        emb = self.relu_encoding(cond)
        return emb


class ResBlock(nn.Module):
    r"""
    Convolutional residual block.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Size of the convolution kernel.
    :param stride: Stride of the convolution.
    :param padding: Padding for the convolution.
    :param bias: Whether to use bias in the convolution.
    :param img_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param decode_upscale: Upscaling factor for the decoding convolution.
    :param head: Whether this is a head block.
    :param tail: Whether this is a tail block.
    :param N: Number of Krylov iterations.
    :param c_mult: Multiplier for the number of channels.
    :param depth_encoding: Depth of the encoding convolution.
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        img_channels=None,
        decode_upscale=None,
        head=False,
        tail=False,
        N=2,
        c_mult=2,
        depth_encoding=2,
    ):
        super(ResBlock, self).__init__()

        if not head and not tail:
            assert (
                in_channels == out_channels
            ), "Only support in_channels==out_channels."
        self.separate_head = isinstance(img_channels, list)
        self.is_head = head
        self.is_tail = tail

        if self.is_head:
            self.head = InHead(img_channels, out_channels, input_layer=True)

        if not self.is_head and not self.is_tail:
            self.conv1 = conv(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias,
                "C",
            )
            self.nl = nn.ReLU(inplace=True)
            self.conv2 = conv(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias,
                "C",
            )

        self.gain = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.PhysicsBlock = MeasCondBlock(
            out_channels=out_channels,
            c_mult=c_mult,
            img_channels=img_channels,
            decode_upscale=decode_upscale,
            N=N,
            depth_encoding=depth_encoding,
        )

    def forward(self, x, physics=None, y=None, img_channels=None, scale=0):
        u = self.conv1(x)
        u = self.nl(u)
        u_2 = self.conv2(u)
        emb_grad = self.PhysicsBlock(
            u, y, physics, img_channels=img_channels, scale=scale
        )
        u_1 = self.gain * emb_grad
        return x + u_2 + u_1


class InHead(torch.nn.Module):
    def __init__(
        self, in_channels_list, out_channels, mode="", bias=False, input_layer=False
    ):
        super(InHead, self).__init__()
        self.in_channels_list = in_channels_list
        self.input_layer = input_layer
        for i, in_channels in enumerate(in_channels_list):
            conv = AffineConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias,
                mode=mode,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
            )
            setattr(self, f"conv{i}", conv)

    def forward(self, x):
        in_channels = x.size(1) - 1 if self.input_layer else x.size(1)

        # find index
        i = self.in_channels_list.index(in_channels)
        x = getattr(self, f"conv{i}")(x)

        return x


class OutTail(torch.nn.Module):
    def __init__(self, in_channels, out_channels_list, mode="", bias=False):
        super(OutTail, self).__init__()
        self.in_channels = in_channels
        self.out_channels_list = out_channels_list
        for i, out_channels in enumerate(out_channels_list):
            conv = AffineConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias,
                mode=mode,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
            )
            setattr(self, f"conv{i}", conv)

    def forward(self, x, out_channels):
        i = self.out_channels_list.index(out_channels)
        x = getattr(self, f"conv{i}")(x)

        return x


class Heads(torch.nn.Module):
    def __init__(
        self,
        in_channels_list,
        out_channels,
        depth=2,
        scale=1,
        bias=True,
        mode="bilinear",
        c_mult=1,
        c_add=0,
        relu_in=False,
        skip_in=False,
    ):
        super(Heads, self).__init__()
        self.in_channels_list = [c * (c_mult + c_add) for c in in_channels_list]
        self.scale = scale
        self.mode = mode
        for i, in_channels in enumerate(self.in_channels_list):
            setattr(
                self,
                f"head{i}",
                HeadBlock(
                    in_channels,
                    out_channels,
                    depth=depth,
                    bias=bias,
                    relu_in=relu_in,
                    skip_in=skip_in,
                ),
            )

        if self.mode == "":
            self.nl = torch.nn.ReLU(inplace=False)
            if self.scale != 1:
                for i, in_channels in enumerate(in_channels_list):
                    setattr(
                        self,
                        f"down{i}",
                        downsample_strideconv(
                            in_channels, in_channels, bias=False, mode=str(self.scale)
                        ),
                    )

    def forward(self, x):
        in_channels = x.size(1)
        i = self.in_channels_list.index(in_channels)

        if self.scale != 1:
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(
                    x, scale_factor=1 / self.scale, mode="bilinear", align_corners=False
                )
            else:
                x = getattr(self, f"down{i}")(x)
                x = self.nl(x)

        # find index
        x = getattr(self, f"head{i}")(x)

        return x


class Tails(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels_list,
        depth=2,
        scale=1,
        bias=True,
        mode="bilinear",
        c_mult=1,
        relu_in=False,
        skip_in=False,
    ):
        super(Tails, self).__init__()
        self.out_channels_list = out_channels_list
        self.scale = scale
        for i, out_channels in enumerate(out_channels_list):
            setattr(
                self,
                f"tail{i}",
                HeadBlock(
                    in_channels,
                    out_channels * c_mult,
                    depth=depth,
                    bias=bias,
                    relu_in=relu_in,
                    skip_in=skip_in,
                ),
            )

        self.mode = mode
        if self.mode == "":
            self.nl = torch.nn.ReLU(inplace=False)
            if self.scale != 1:
                for i, out_channels in enumerate(out_channels_list):
                    setattr(
                        self,
                        f"up{i}",
                        upsample_convtranspose(
                            out_channels * c_mult,
                            out_channels * c_mult,
                            bias=bias,
                            mode=str(self.scale),
                        ),
                    )

    def forward(self, x, out_channels):
        i = self.out_channels_list.index(out_channels)
        x = getattr(self, f"tail{i}")(x)
        # find index
        if self.scale != 1:
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.scale, mode="bilinear", align_corners=False
                )
            else:
                x = getattr(self, f"up{i}")(x)

        return x


class HeadBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bias=True,
        depth=2,
        relu_in=False,
        skip_in=False,
    ):
        super(HeadBlock, self).__init__()

        padding = kernel_size // 2

        c = out_channels if depth < 2 else in_channels

        self.convin = torch.nn.Conv2d(
            in_channels, c, kernel_size, padding=padding, bias=bias
        )
        self.zero_conv_skip = torch.nn.Conv2d(in_channels, c, 1, bias=False)
        self.depth = depth
        self.nl_1 = torch.nn.ReLU(inplace=False)
        self.nl_2 = torch.nn.ReLU(inplace=False)
        self.relu_in = relu_in
        self.skip_in = skip_in

        for i in range(depth - 1):
            if i < depth - 2:
                c_in, c = in_channels, in_channels
            else:
                c_in, c = in_channels, out_channels

            setattr(
                self,
                f"conv1{i}",
                torch.nn.Conv2d(c_in, c_in, kernel_size, padding=padding, bias=bias),
            )
            setattr(
                self,
                f"conv2{i}",
                torch.nn.Conv2d(c_in, c, kernel_size, padding=padding, bias=bias),
            )
            setattr(self, f"skipconv{i}", torch.nn.Conv2d(c_in, c, 1, bias=False))

    def forward(self, x):

        if self.skip_in and self.relu_in:
            x = self.nl_1(self.convin(x)) + self.zero_conv_skip(x)
        elif self.skip_in and not self.relu_in:
            x = self.convin(x) + self.zero_conv_skip(x)
        else:
            x = self.convin(x)

        for i in range(self.depth - 1):
            aux = getattr(self, f"conv1{i}")(x)
            aux = self.nl_2(aux)
            aux_0 = getattr(self, f"conv2{i}")(aux)
            aux_1 = getattr(self, f"skipconv{i}")(x)
            x = aux_0 + aux_1

        return x


# --------------------------------------------------------------------------------------
class AffineConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mode="affine",
        bias=False,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="circular",
        blind=True,
    ):
        if mode == "affine":  # f(a*x + 1) = a*f(x) + 1
            bias = False
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
        )
        self.blind = blind
        self.mode = mode

    def affine(self, w):
        """returns new kernels that encode affine combinations"""
        return (
            w.view(self.out_channels, -1).roll(1, 1).view(w.size())
            - w
            + 1 / w[0, ...].numel()
        )

    def forward(self, x):
        if self.mode != "affine":
            return super().forward(x)
        else:
            kernel = (
                self.affine(self.weight)
                if self.blind
                else torch.cat(
                    (self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]),
                    dim=1,
                )
            )
            padding = tuple(
                elt for elt in reversed(self.padding) for _ in range(2)
            )  # used to translate padding arg used by Conv module to the ones used by F.pad
            padding_mode = (
                self.padding_mode if self.padding_mode != "zeros" else "constant"
            )  # used to translate padding_mode arg used by Conv module to the ones used by F.pad
            return F.conv2d(
                F.pad(x, padding, mode=padding_mode),
                kernel,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            )


"""
Functional blocks below

Parts of code borrowed from
https://github.com/cszn/DPIR/tree/master/models
https://github.com/xinntao/BasicSR
"""
from collections import OrderedDict
import torch
import torch.nn as nn


"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
):
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    padding=0,
    bias=True,
    mode="2R",
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
        "8",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
    )
    return up1


def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    padding=0,
    bias=True,
    mode="2R",
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
        "8",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
    )
    return down1


class Upsampling(Downsampling):
    def A(self, x, **kwargs):
        return super().A_adjoint(x, **kwargs)

    def A_adjoint(self, y, **kwargs):
        return super().A(y, **kwargs)

    def prox_l2(self, z, y, gamma, **kwargs):
        return super().prox_l2(z, y, gamma, **kwargs)


class MultiScalePhysics(Physics):
    def __init__(
        self,
        physics,
        img_shape,
        filter="sinc",
        scales=[2, 4, 8],
        device="cpu",
        **kwargs,
    ):
        super().__init__(noise_model=physics.noise_model, **kwargs)
        self.base = physics
        self.scales = scales
        self.img_shape = img_shape
        self.Upsamplings = [
            Upsampling(img_size=img_shape, filter=filter, factor=factor, device=device)
            for factor in scales
        ]
        self.scale = 0

    def set_scale(self, scale):
        if scale is not None:
            self.scale = scale

    def A(self, x, scale=None, **kwargs):
        self.set_scale(scale)
        if self.scale == 0:
            return self.base.A(x, **kwargs)
        else:
            return self.base.A(self.Upsamplings[self.scale - 1].A(x), **kwargs)

    def downsample(self, x, scale=None):
        self.set_scale(scale)
        if self.scale == 0:
            return x
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(x)

    def upsample(self, x, scale=None):
        self.set_scale(scale)
        if self.scale == 0:
            return x
        else:
            return self.Upsamplings[self.scale - 1].A(x)

    def update_parameters(self, **kwargs):
        self.base.update_parameters(**kwargs)


class MultiScaleLinearPhysics(MultiScalePhysics, LinearPhysics):
    def __init__(self, physics, img_shape, filter="sinc", scales=[2, 4, 8], **kwargs):
        super().__init__(
            physics=physics, img_shape=img_shape, filter=filter, scales=scales, **kwargs
        )

    def A_adjoint(self, y, scale=None, **kwargs):
        self.set_scale(scale)
        y = self.base.A_adjoint(y, **kwargs)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)


class Pad(LinearPhysics):
    def __init__(self, physics, pad):
        super().__init__(noise_model=physics.noise_model)
        self.base = physics
        self.pad = pad

    def A(self, x):
        return self.base.A(x[..., self.pad[0] :, self.pad[1] :])

    def A_adjoint(self, y):
        y = self.base.A_adjoint(y)
        y = torch.nn.functional.pad(y, (self.pad[1], 0, self.pad[0], 0))
        return y

    def remove_pad(self, x):
        return x[..., self.pad[0] :, self.pad[1] :]

    def update_parameters(self, **kwargs):
        self.base.update_parameters(**kwargs)
