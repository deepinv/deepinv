# Code borrowed from https://github.com/ZhenghanFang/learned-proximal-networks
import numpy as np
import torch
from torch import nn


class ICNN(nn.Module):
    r"""
    Convolutional Input Convex Neural Network (ICNN).

    The network is built to be convex in its input.
    The model is fully convolutional and thus can be applied to images of any size.

    Based on the implementation from the paper
    `"Data-Driven Mirror Descent with Input-Convex Neural Networks <https://arxiv.org/abs/2206.06733>`_.

    :param int in_channels: Number of input channels.
    :param int num_filters: Number of hidden units.
    :param kernel_dim: dimension of the convolutional kernels.
    :param int num_layers: Number of layers.
    :param float strong_convexity: Strongly convex parameter.
    :param bool pos_weights: Whether to force positive weights in the forward pass.
    :param str device: Device to use for the model.
    """

    def __init__(
        self,
        in_channels=3,
        num_filters=64,
        kernel_dim=5,
        num_layers=10,
        strong_convexity=0.5,
        pos_weights=True,
        device="cpu",
    ):
        super(ICNN, self).__init__()
        self.n_in_channels = in_channels
        self.n_layers = num_layers
        self.n_filters = num_filters
        self.kernel_size = kernel_dim
        self.padding = (self.kernel_size - 1) // 2

        # these layers should have non-negative weights
        self.wz = nn.ModuleList(
            [
                nn.Conv2d(
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
                nn.Conv2d(
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
                nn.Conv2d(
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
        self.final_conv2d = nn.Conv2d(
            self.n_filters,
            self.n_in_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
            padding_mode="circular",
            bias=False,
            device=device,
        )

        # slope of leaky-relu
        self.negative_slope = 0.2
        self.strong_convexity = strong_convexity

        self.pos_weights = pos_weights
        self.device = device

    def forward(self, x):
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
        z = self.final_conv2d(z)
        z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)

        return z_avg + 0.5 * self.strong_convexity * (x**2).sum(
            dim=[1, 2, 3]
        ).reshape(-1, 1)

    def grad(self, x):
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
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand(
            1, self.n_filters, self.kernel_size, self.kernel_size
        ).to(self.device)
        return self

    # a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)
        return self
