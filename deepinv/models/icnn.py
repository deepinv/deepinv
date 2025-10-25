# Code borrowed from https://github.com/ZhenghanFang/learned-proximal-networks
from __future__ import annotations
import torch
from torch import nn
from .utils import fix_dim, conv_nd, avgpool_nd


class ICNN(nn.Module):
    r"""
    Convolutional Input Convex Neural Network (ICNN).

    The network is built to be convex in its input.
    The model is fully convolutional and thus can be applied to images of any size.

    Based on the implementation from :footcite:t:`tan2023data`.

    :param int in_channels: Number of input channels.
    :param int num_filters: Number of hidden units.
    :param int kernel_dim: dimension of the convolutional kernels.
    :param int num_layers: Number of layers.
    :param float strong_convexity: Strongly convex parameter.
    :param bool pos_weights: Whether to force positive weights in the forward pass.
    :param torch.device, str device: Device to put the model on.
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)

    """

    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 64,
        kernel_dim: int = 5,
        num_layers: int = 10,
        strong_convexity: float = 0.5,
        pos_weights: bool = True,
        device: torch.device | str = "cpu",
        dim: int | str = 2,
    ):
        super(ICNN, self).__init__()

        dim = fix_dim(dim)
        conv = conv_nd(dim)

        self.n_in_channels = in_channels
        self.n_layers = num_layers
        self.n_filters = num_filters
        self.kernel_size = kernel_dim
        self.padding = (self.kernel_size - 1) // 2

        # these layers should have non-negative weights
        self.wz = nn.ModuleList(
            [
                conv(
                    self.n_filters,
                    self.n_filters,
                    self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    padding_mode="circular",
                    bias=False,
                    device=device,
                )
                for i in range(self.n_layers)
            ]
        )

        # these layers can have arbitrary weights
        self.wx_quad = nn.ModuleList(
            [
                conv(
                    self.n_in_channels,
                    self.n_filters,
                    self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    padding_mode="circular",
                    bias=False,
                    device=device,
                )
                for i in range(self.n_layers + 1)
            ]
        )
        self.wx_lin = nn.ModuleList(
            [
                conv(
                    self.n_in_channels,
                    self.n_filters,
                    self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    padding_mode="circular",
                    bias=True,
                    device=device,
                )
                for i in range(self.n_layers + 1)
            ]
        )

        # one final conv layer with nonnegative weights
        self.final_conv = conv(
            self.n_filters,
            self.n_in_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
            padding_mode="circular",
            bias=False,
            device=device,
        )

        self.avgpool = avgpool_nd(dim)
        self.dim = dim

        # slope of leaky-relu
        self.negative_slope = 0.2
        self.strong_convexity = strong_convexity

        self.pos_weights = pos_weights
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Calculate potential function of the ICNN.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        if self.pos_weights:
            self.zero_clip_weights()
        z = torch.nn.functional.leaky_relu(
            self.wx_quad[0](x) ** 2 + self.wx_lin[0](x),
            negative_slope=self.negative_slope,
        )
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(
                self.wz[layer](z)
                + self.wx_quad[layer + 1](x) ** 2
                + self.wx_lin[layer + 1](x),
                negative_slope=self.negative_slope,
            )
        z = self.final_conv(z)
        z_avg = self.avgpool(z.size()[2:])(z).view(z.size()[0], -1)

        return z_avg + 0.5 * self.strong_convexity * torch.linalg.vector_norm(
            x, dim=tuple(range(1, self.dim + 2)), ord=2
        ).pow(2)

    @torch.enable_grad()
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Calculate the gradient of the potential function.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        x = x.requires_grad_(True)
        out = self.forward(x)
        return torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    # a weight initialization routine for the ICNN, with positive weights
    def initialize_weights(self, min_val=0.0, max_val=0.001):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val) * torch.rand(
                self.n_filters, self.n_filters, self.kernel_size, self.kernel_size
            ).to(self.device)
        self.final_conv.weight.data = min_val + (max_val - min_val) * torch.rand(
            1, self.n_filters, self.kernel_size, self.kernel_size
        ).to(self.device)
        return self

    # a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        self.final_conv.weight.data.clamp_(0)
        return self
