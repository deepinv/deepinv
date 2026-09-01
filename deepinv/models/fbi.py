import torch
from torch import nn
from torch.nn import functional as F

from deepinv.models.utils import conv_nd, conv_transpose_nd, maxpool_nd, test_pad
from deepinv.models import Denoiser
from deepinv.models.utils import conv_nd

Conv2d = conv_nd(2)
MaxPool2d = maxpool_nd(2)
ConvTranspose2d = conv_transpose_nd(2)

MASKS = (
    ((1, 1, 1), (1, 0, 1), (1, 1, 1)),
    (
        (0, 1, 0, 1, 0),
        (1, 0, 0, 0, 1),
        (0, 0, 1, 0, 0),
        (1, 0, 0, 0, 1),
        (0, 1, 0, 1, 0),
    ),
    ((1, 0, 1), (0, 1, 0), (1, 0, 1)),
)


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        skip = torch.relu(self.conv2(x))
        return self.pool(skip), skip


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="add"):
        super().__init__()
        self.merge_mode = merge_mode
        self.upconv = ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        merged_channels = 2 * out_channels if merge_mode == "concat" else out_channels
        self.conv1 = Conv2d(merged_channels, out_channels, 3, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, skip, x):
        x = self.upconv(x)
        x = torch.cat((x, skip), 1) if self.merge_mode == "concat" else x + skip
        return torch.relu(self.conv2(torch.relu(self.conv1(x))))


class _MaskedConv(nn.Module):
    def __init__(self, cin, cout, mask, dilation=1):
        super().__init__()
        padding = len(mask) // 2 * dilation
        self.conv1 = Conv2d(cin, cout, len(mask), padding=padding, dilation=dilation)
        self.register_buffer("mask", torch.tensor(mask), persistent=False)

    def forward(self, x):
        return F.conv2d(
            x,
            self.conv1.weight * self.mask,
            self.conv1.bias,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation,
        )


class _Residual(nn.Module):
    def __init__(self, channels, mul=1):
        super().__init__()
        self.activation1 = nn.PReLU(channels * mul, 0)
        self.activation2 = nn.PReLU(channels, 0)
        self.conv1_1by1 = Conv2d(channels, channels * mul, 1)
        self.conv2_1by1 = Conv2d(channels * mul, channels, 1)

    def forward(self, x):
        r = self.conv2_1by1(self.activation1(self.conv1_1by1(x)))
        return self.activation2((x + r) / 2)


class _First(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.new1 = _MaskedConv(cin, cout, MASKS[0])
        self.residual_module = _Residual(cout)
        self.activation_new1 = nn.PReLU(1, 0)

    def forward(self, x):
        raw = self.activation_new1(self.new1(x))
        return self.residual_module(raw), raw


class _Next(nn.Module):
    def __init__(self, channels, second=False):
        super().__init__()
        name, mask, dilation = (
            ("new2", MASKS[1], 1) if second else ("new3", MASKS[2], 3)
        )
        setattr(self, name, _MaskedConv(channels, channels, mask, dilation))
        self._name = name
        self.activation_new1 = nn.PReLU(channels, 0)
        self.residual_module = _Residual(channels)
        self.activation_new2 = nn.PReLU(channels, 0)

    def forward(self, x, raw):
        raw = self.activation_new1(getattr(self, self._name)(raw))
        return self.residual_module(self.activation_new2((raw + x) / 2)), raw


class PGENet(nn.Module):
    r"""
    PGE-Net for Poisson--Gaussian noise parameter estimation.

    This compact U-Net returns spatial Gaussian standard-deviation and Poisson
    gain maps in ``(sigma, gain)`` channel order. By default, the output is
    squared to ensure nonnegative estimates. The model can be wrapped with
    :class:`deepinv.models.PoissonGaussianEstimator` to obtain noise parameters.

    :param int out_channels: Number of output channels. Default: 2.
    :param int in_channels: Number of input channels. Default: 1.
    :param int depth: Number of U-Net scales. Default: 3.
    :param int start_filts: Number of features at the first scale. Default: 64.
    :param str merge_mode: Skip-connection mode, either ``"add"`` or
        ``"concat"``. Default: ``"add"``.
    :param bool square_output: If ``True``, square the output maps. Default:
        ``True``.
    """

    def __init__(
        self,
        out_channels=2,
        in_channels=1,
        depth=3,
        start_filts=64,
        merge_mode="add",
        square_output=True,
    ):
        super().__init__()
        if depth < 2 or merge_mode not in ("add", "concat"):
            raise ValueError("Use depth >= 2 and merge_mode 'add' or 'concat'.")

        channels = [start_filts * 2**i for i in range(depth)]
        self.square_output = square_output
        self.noiseSTD = nn.Parameter(torch.log(torch.tensor(0.5)))
        self.down_convs = nn.ModuleList(
            _DownBlock(
                in_channels if i == 0 else channels[i - 1],
                channels[i],
                i < depth - 1,
            )
            for i in range(depth)
        )
        self.up_convs = nn.ModuleList(
            _UpBlock(channels[i], channels[i - 1], merge_mode)
            for i in range(depth - 1, 0, -1)
        )
        self.conv_final = Conv2d(channels[0], out_channels, 1)

    def _forward(self, x):
        skips = []
        for block in self.down_convs:
            x, skip = block(x)
            skips.append(skip)
        for i, block in enumerate(self.up_convs):
            x = block(skips[-i - 2], x)
        x = self.conv_final(x)
        x = x.square() if self.square_output else x
        # The PGE-Net expects the output channels to be in the order (sigma, gain).
        x = x[:, (1, 0)]
        return x

    def forward(self, x, **kwargs):
        r"""Estimate spatial Poisson--Gaussian noise parameters.

        :param torch.Tensor x: Input image of shape ``(B, C, H, W)``.
        :return: (:class:`torch.Tensor`) Parameter maps in ``(sigma, gain)``
            channel order.
        """
        factor = 2 ** len(self.up_convs)
        divisible = all(size % factor == 0 for size in x.shape[-2:])
        return (
            self._forward(x) if divisible else test_pad(self._forward, x, modulo=factor)
        )


class FBINet(Denoiser):
    r"""
    FBI-Net blind-spot denoiser.

    This denoiser uses masked convolutions and predicts a pixel-wise slope and
    intercept for each input channel. With ``affine=True``, the input is
    normalized per channel, denoised using the affine parameters, and rescaled
    to its original range.

    :param int in_channels: Number of input channels. Default: 1.
    :param int out_channels: Number of output channels. Default: ``None``, which
        sets the output channels to ``in_channels * 2`` if ``affine=True`, or ``in_channels`` if ``affine=False``.
    :param int layers: Number of masked-convolution stages. Default: 17.
    :param int filters: Number of features in each stage. Default: 64.
    :param float sigmoid_value: Maximum scale of the predicted affine slope.
        Default: 0.1.
    :param bool affine: Apply affine denoising and input rescaling. Default:
        ``True``.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=None,
        layers=17,
        filters=64,
        sigmoid_value=0.1,
        affine=True,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels * 2 if affine else in_channels

        self.num_layers = layers
        self.affine = affine
        self.in_channels = in_channels
        self.sigmoid_value = sigmoid_value
        self.new1 = _First(in_channels, filters)
        self.new2 = _Next(filters, second=True)
        for i in range(layers - 2):
            self.add_module(f"new_{i}", _Next(filters))
        self.residual_module = _Residual(filters)
        self.activation = nn.PReLU(filters, 0)
        self.output_layer = Conv2d(filters, out_channels, 1)

        if not (affine):
            print(
                "Warning: FBI denoiser is being used without affine rescaling."
                "sigmoid_value will be ignored."
            )

    def _forward(self, x):

        out, raw = self.new1(x)
        total = out
        out, raw = self.new2(out, raw)
        total = total + out

        for i in range(self.num_layers - 2):
            out, raw = getattr(self, f"new_{i}")(out, raw)
            total = total + out

        out = self.output_layer(
            self.residual_module(self.activation(total / self.num_layers))
        )

        if self.affine:
            slope, intercept = out.chunk(2, dim=1)
            out = torch.cat((self.sigmoid_value * slope.sigmoid(), intercept), 1)
        return out

    def forward(self, z, sigma=None, **kwargs):
        r"""Denoise an image.

        :param torch.Tensor z: Noisy input of shape ``(B, C, H, W)``.
        :param float, torch.Tensor sigma: Noise level, unused by this model.
        :return: (:class:`torch.Tensor`) Denoised image.
        """

        zn = z
        if self.affine:
            lo = z.amin(dim=(-2, -1), keepdim=True)
            scale = (z.amax(dim=(-2, -1), keepdim=True) - lo).clamp_min(1e-8)
            zn = (z - lo) / scale

        out = self._forward(zn)

        if self.affine:
            slope, intercept = out.chunk(2, dim=1)
            out = (slope * zn + intercept) * scale + lo

        return out
