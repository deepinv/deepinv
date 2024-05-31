#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple


class AffineConv2d(nn.Conv2d):
    """
    A Convolutional layer that performs affine transformations on the kernels before applying convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor that will be convolved with the kernel.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): The size of the kernel used in the convolution operation. It can be a single integer or a tuple of two integers.
        stride (Optional[Union[int, Tuple[int, int]]], default is 1): Stride of the convolution operation. Default is 1.
        padding (Optional[Union[int, Tuple[int, int]]], default is 0): Padding size for the input tensor before applying the convolution operation. Default is 0.
        dilation (Optional[Union[int, Tuple[int, int]]], default is 1): Dilation rate for the convolution operation. Default is 1.
        groups (Optional[int], default is 1): Number of blocked connections from input channels to output channels in grouped convolutions. Default is 1.
        padding_mode (str, default is "reflect"): Mode of padding for the input tensor before applying the convolution operation. Default is "reflect".
        blind (Optional[bool], default is True): Flag to indicate whether to use blind affine transformations. If it's set to True, no additional operations are performed on the last filter of each output channel during forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "reflect",
        blind: Optional[bool] = True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=False,
        )

        self.blind = (
            blind  # Flag to indicate whether to use blind affine transformations
        )

    def affine(self, w) -> torch.Tensor:
        """
        Returns new kernels that encode affine combinations based on the input kernel weights `w`.

        Args:
            w (Tensor): The input kernel weight tensor of shape (out_channels, in_channels, kH, kW) where out_channels is the number of output channels,
                        and in_channels is the number of input channels.

        Returns:
            Tensor: A new kernel tensor with same dimensions as `w` after applying affine transformations.
        """
        return (
            w.view(self.out_channels, -1).roll(1, 1).view(w.size())
            - w
            + (1 / w[0, ...].numel())
        )  # Perform affine transformation on the input weights `w`

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the AffineConv2d layer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_channels, H, W) where batch_size is the number of samples in the batch.

        Returns:
            Tensor: The output tensor after applying affine transformed convolution operation on the input `x` with same spatial dimensions as input.
        """
        kernel = (
            self.affine(self.weight)
            if self.blind
            else torch.cat(
                (
                    self.affine(
                        # Apply affine transformations on the kernel weights
                        self.weight[:, :-1, :, :]
                    ),
                    self.weight[:, -1:, :, :],
                ),
                dim=1,
            )
        )
        # Translate padding arg used by Conv module to the ones used by F.pad
        padding = tuple(int(elt) for elt in reversed(self.padding) for _ in range(2))
        # Translate padding_mode arg used by Conv module to the ones used by F.pad
        padding_mode = self.padding_mode if self.padding_mode != "zeros" else "constant"

        return F.conv2d(
            F.pad(x, padding, mode=padding_mode),
            kernel,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )


class AffineConvTranspose2d(nn.Module):
    """
    Affine ConvTranspose2d with kernel=2 and stride=2, implemented using PixelShuffle.

    Args:
        in_channels (int): Number of channels in the input tensor that will be convolved with the kernel.
        out_channels (int): Number of channels produced by the convolution.

    Returns:
        Tensor: The output tensor after applying affine transformed transpose operation on the input `x` with same spatial dimensions as input.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = AffineConv2d(in_channels, 4 * out_channels, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.conv1x1(x), 2)


class SortPool(nn.Module):
    """
    Channel-wise sort pooling, where C must be an even number.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        This module performs channel-wise sort pooling on the input tensor x.
        It splits each feature map into two halves and sorts them independently.
        The difference between the minima of two halves is calculated using ReLU,
        which allows us to backpropagate through this operation and update weights in a neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the batch size,
                              C is the number of channels, and H and W are the height and width
                              of the feature maps respectively. C must be an even number.

        Returns:
            torch.Tensor: The output tensor after sort pooling. It has the same shape as the input tensor x.

        Shape:
            - Input: (N, C, H, W) where N is batch size and C is number of channels.
            - Output: Same shape as input.
        """
        # A trick with relu is used because the derivative for torch.aminmax is not yet implemented and torch.sort is slow.
        N, C, H, W = x.size()
        x1, x2 = torch.split(x.view(N, C // 2, 2, H, W), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1 - diff, x2 + diff), dim=2).view(N, C, H, W)


class ResidualConnection(nn.Module):
    """Residual connection module norm
    
    Args:
        mode (str): The type of residual connection to use. Options are 'ordinary', "scale-equiv" and 'norm-equiv'. Default is 'ordinary'.
    """

    def __init__(self, mode="ordinary") -> None:
        super().__init__()

        self.mode = mode
        if mode == "norm-equiv":
            self.alpha = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x, y):
        if self.mode == "norm-equiv":
            return self.alpha * x + (1 - self.alpha) * y
        return x + y


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    blind=True,
    mode="ordinary",
):
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias if mode == "ordinary" else False,
            padding_mode=padding_mode,
        )
    elif mode == "norm-equiv":
        return AffineConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode="reflect",
            blind=blind,
        )
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


def upscale2(in_channels, out_channels, bias=True, mode="ordinary"):
    """Upscaling using convtranspose with kernel 2x2 and stride 2"""
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=bias if mode == "ordinary" else False,
        )
    elif mode == "norm-equiv":
        return AffineConvTranspose2d(in_channels, out_channels)
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


def activation(mode="ordinary"):
    if mode == "ordinary" or mode == "scale-equiv":
        return nn.ReLU(inplace=True)
    elif mode == "norm-equiv":
        return SortPool()
    else:
        raise NotImplementedError(
            "Only ordinary, scale-equiv and norm-equiv modes are implemented"
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, mode="ordinary"):
        super().__init__()

        self.m_res = nn.Sequential(
            conv2d(
                in_channels, in_channels, 3, stride=1, padding=1, bias=bias, mode=mode
            ),
            activation(mode),
            conv2d(
                in_channels, out_channels, 3, stride=1, padding=1, bias=bias, mode=mode
            ),
        )

        self.sum = ResidualConnection(mode)

    def forward(self, x):
        return self.sum(x, self.m_res(x))
