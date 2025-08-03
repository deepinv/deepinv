import torch
import torch.nn as nn
from .drunet import test_pad
from .base import Denoiser
import warnings
from typing import Optional


class BFBatchNorm2d(nn.BatchNorm2d):
    r"""
    From :footcite:t:`mohan2020robust`.
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
        if ``batch_norm="biasfree"``, use ``BFBatchNorm2d`` from :footcite:t:`mohan2020robust`.
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

        def norm(ch: int) -> Optional[nn.Module]:
            if batch_norm == "biasfree":
                return BFBatchNorm2d(ch, use_bias=bias)
            elif batch_norm == True:
                return nn.BatchNorm2d(ch)
            elif batch_norm == False:
                return None
            else:
                warnings.warn(
                    f"Expected batch_norm to be True, False or 'biasfree', got {batch_norm=}. "
                )
                return nn.BatchNorm2d(ch)  # for backwards compatibility

        def conv_block(ch_in: int, ch_out: int) -> nn.Module:
            m = nn.Sequential(
                nn.Conv2d(
                    ch_in,
                    ch_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    padding_mode="circular" if circular_padding else "zeros",
                )
            )

            m_norm = norm(ch_out)
            if m_norm is not None:
                m.append(m_norm)

            m.extend(
                (
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        ch_out,
                        ch_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        padding_mode="circular" if circular_padding else "zeros",
                    ),
                )
            )

            m_norm = norm(ch_out)
            if m_norm is not None:
                m.append(m_norm)

            m.append(nn.ReLU(inplace=True))

            return m

        def up_conv(ch_in: int, ch_out: int) -> nn.Module:
            m = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    ch_in,
                    ch_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    padding_mode="circular" if circular_padding else "zeros",
                ),
            )

            m_norm = norm(ch_out)
            if m_norm is not None:
                m.append(m_norm)

            m.append(nn.ReLU(inplace=True))

            return m

        max_scale = 5
        # Build the U-Net architecture level by level
        for scale in range(1, max_scale + 1):
            if scale in [1, 2]:  # for backwards compatibility
                if scales not in range(scale, max_scale + 1):
                    warnings.warn(f"Unexpected {scales=}, expected 2, 3, 4 or 5.")
                present = True
            else:
                present = scales in range(scale, max_scale + 1)

            if scale == 1:
                ch_enc_in = in_channels
                ch_dec_out = out_channels
            else:
                ch_enc_in = ch_dec_out = 64 * (2 ** (scale - 2))

            ch_enc_out = ch_dec_in = 64 * (2 ** (scale - 1))

            # Encoder branch
            setattr(
                self,
                f"Conv{scale}",
                conv_block(ch_in=ch_enc_in, ch_out=ch_enc_out) if present else None,
            )

            # Decoder branch
            if scale == 1:
                self.Conv_1x1 = nn.Conv2d(
                    in_channels=ch_dec_in,
                    out_channels=ch_dec_out,
                    bias=bias,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            else:
                setattr(
                    self,
                    f"Up{scale}",
                    up_conv(ch_in=ch_dec_in, ch_out=ch_dec_out) if present else None,
                )
                setattr(
                    self,
                    f"Up_conv{scale}",
                    conv_block(ch_in=ch_dec_in, ch_out=ch_dec_out) if present else None,
                )

        if scales == 5:
            self._forward = self.forward_standard
        elif scales in [2, 3, 4]:
            self._forward = getattr(self, f"forward_compact{scales}")
        else:
            warnings.warn(
                f"Unexpected {scales=}, expected 2, 3, 4 or 5. Using standard forward."
            )

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
    def _forward_general(self, x, *, n_scales):
        if n_scales not in [2, 3, 4, 5]:
            raise ValueError(f"Unexpected {n_scales=}, expected 2, 3, 4 or 5.")

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

        if self.residual:
            if self.in_channels == self.out_channels:
                im_out = im_out + x
            else:
                warnings.warn(
                    "Residual connection requested but input and output channels do not match. "
                    "Skipping residual connection."
                )

        return im_out
