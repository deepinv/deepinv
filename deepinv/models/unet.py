import torch
import torch.nn as nn
from .drunet import test_pad
from .base import Denoiser


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From Mohan et al.

    "Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks"
    S. Mohan, Z. Kadkhodaie, E. P. Simoncelli, C. Fernandez-Granda
    Int'l. Conf. on Learning Representations (ICLR), Apr 2020.
    """

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, use_bias=False, affine=True
    ):
        super(BFBatchNorm2d, self).__init__(num_features, eps, momentum)
        self.use_bias = use_bias
        self.affine = affine

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        if self.use_bias:
            mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)
        if self.training is not True:
            if self.use_bias:
                y = y - self.running_mean.view(-1, 1)
            y = y / (self.running_var.view(-1, 1) ** 0.5 + self.eps)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    if self.use_bias:
                        self.running_mean = (
                            1 - self.momentum
                        ) * self.running_mean + self.momentum * mu
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * sigma2
            if self.use_bias:
                y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1) ** 0.5 + self.eps)
        if self.affine:
            y = self.weight.view(-1, 1) * y
            if self.use_bias:
                y += self.bias.view(-1, 1)

        return y.view(return_shape).transpose(0, 1)


class UNet(Denoiser):
    r"""
    U-Net convolutional denoiser.

    This network is a fully convolutional denoiser based on the U-Net architecture. The number of downsample steps
    can be controlled with the ``scales`` parameter. The number of trainable parameters increases with the number of
    scales.

    .. warning::
        When using the bias-free batch norm ``BFBatchNorm2d`` via ``batch_norm="biasfree"``, NaNs may be encountered
        during training, causing the whole training procedure to fail.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param bool residual: use a skip-connection between output and output.
    :param bool circular_padding: circular padding for the convolutional layers.
    :param bool cat: use skip-connections between intermediate levels.
    :param bool bias: use learnable biases.
    :param bool, str batch_norm: if False, no batchnorm applied, if ``True``, use :class:`torch.nn.BatchNorm2d`,
        if ``batch_norm="biasfree"``, use ``BFBatchNorm2d`` from
        `"Robust And Interpretable Blind Image Denoising Via Bias-Free Convolutional Neural Networks" by Mohan et al. <https://arxiv.org/abs/1906.05478>`_.
    :param int scales: Number of downsampling steps used in the U-Net. The options are 2,3,4 and 5.
        The number of trainable parameters increases with the scale.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        residual=True,
        circular_padding=False,
        cat=True,
        bias=True,
        batch_norm=True,
        scales=4,
    ):
        super(UNet, self).__init__()
        self.name = "unet"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.compact = scales
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        biasfree = batch_norm == "biasfree"

        def conv_block(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    (
                        BFBatchNorm2d(ch_out, use_bias=bias)
                        if biasfree
                        else nn.BatchNorm2d(ch_out)
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias
                    ),
                    (
                        BFBatchNorm2d(ch_out, use_bias=bias)
                        if biasfree
                        else nn.BatchNorm2d(ch_out)
                    ),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias
                    ),
                    nn.ReLU(inplace=True),
                )

        def up_conv(ch_in, ch_out):
            if batch_norm:
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias
                    ),
                    (
                        BFBatchNorm2d(ch_out, use_bias=bias)
                        if biasfree
                        else nn.BatchNorm2d(ch_out)
                    ),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(
                        ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias
                    ),
                    nn.ReLU(inplace=True),
                )

        self.Conv1 = conv_block(ch_in=in_channels, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = (
            conv_block(ch_in=128, ch_out=256) if self.compact in [3, 4, 5] else None
        )
        self.Conv4 = (
            conv_block(ch_in=256, ch_out=512) if self.compact in [4, 5] else None
        )
        self.Conv5 = conv_block(ch_in=512, ch_out=1024) if self.compact in [5] else None

        self.Up5 = up_conv(ch_in=1024, ch_out=512) if self.compact in [5] else None
        self.Up_conv5 = (
            conv_block(ch_in=1024, ch_out=512) if self.compact in [5] else None
        )

        self.Up4 = up_conv(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        self.Up_conv4 = (
            conv_block(ch_in=512, ch_out=256) if self.compact in [4, 5] else None
        )

        self.Up3 = up_conv(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        self.Up_conv3 = (
            conv_block(ch_in=256, ch_out=128) if self.compact in [3, 4, 5] else None
        )

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            bias=bias,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if self.compact == 5:
            self._forward = self.forward_standard
        if self.compact == 4:
            self._forward = self.forward_compact4
        if self.compact == 3:
            self._forward = self.forward_compact3
        if self.compact == 2:
            self._forward = self.forward_compact2

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (self.compact - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def forward_standard(self, x):
        return self._forward_general(x, n_scales=5)

    def forward_compact4(self, x):
        return self._forward_general(x, n_scales=4)

    def forward_compact3(self, x):
        return self._forward_general(x, n_scales=3)

    def forward_compact2(self, x):
        return self._forward_general(x, n_scales=2)

    # Internal general forward algorithm for any supported number of scales
    def _forward_general(self, x, n_scales):
        assert n_scales in [2, 3, 4, 5], f"Unexpected {n_scales=}, expected 2, 3, 4 or 5"
        # The variable feats_stack is a stack populated with the intermediate
        # feature maps as they are computed.
        # NOTE: We rely heavily on feats_stack being None being equivalent to
        # self.cat having been False.
        if self.cat:
            feats_stack = []
        else:
            feats_stack = None

        # encoding path
        for scale in range(1, n_scales + 1):
            if scale == 1:
                inp = x
            else:
                # NOTE: The variable feats is always initialized at this point.
                inp = self.Maxpool(feats)
            m_conv = getattr(self, f"Conv{scale}")
            feats = m_conv(inp)
            if feats_stack is not None and scale != n_scales:
                feats_stack.append(feats)

        # decoding + concat path
        for scale in range(n_scales, 1, -1):
            m_up = getattr(self, f"Up{scale}")
            feats = m_up(feats)
            if feats_stack is not None:
                feats_skip = feats_stack.pop()
                feats = torch.cat((feats_skip, feats), dim=1)
                m_upconv = getattr(self, f"Up_conv{scale}")
                feats = m_upconv(feats)

        im_out = self.Conv_1x1(feats)

        # NOTE: Can self.residual be true while self.in_channels != self.out_channels?
        if self.residual and self.in_channels == self.out_channels:
            im_out = im_out + x

        return im_out
