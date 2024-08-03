# Code borrowed from https://github.com/ZhenghanFang/learned-proximal-networks
import numpy as np
import torch
from torch import nn


class ICNN(nn.Module):
    r"""
    Input Convex Neural Network.

    Mostly based on the implementation from the paper
    `What's in a Prior? Learned Proximal Networks for Inverse Problems <https://openreview.net/pdf?id=kNPcOaqC5r>`_,
    and from the implementation from the `OOT libreary <https://ott-jax.readthedocs.io/en/latest/neural/_autosummary/ott.neural.networks.icnn.ICNN.html>`_.

    :param int in_channels: Number of input channels.
    :param int dim_hidden: Number of hidden units.
    :param float beta_softplus: Beta parameter for the softplus activation function.
    :param float alpha: Strongly convex parameter.
    :param bool pos_weights: Whether to force positive weights in the forward pass.
    :param torch.nn.Module rectifier_fn: Activation function to use to force postive weight.
    :param str device: Device to use for the model.
    """

    def __init__(
        self,
        in_channels=3,
        dim_hidden=256,
        beta_softplus=100,
        alpha=0.0,
        pos_weights=False,
        rectifier_fn=torch.nn.ReLU(),
        device="cpu",
    ):
        super().__init__()

        self.hidden = dim_hidden
        self.lin = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels, dim_hidden, 3, bias=True, stride=1, padding=1
                ),  # 128
                nn.Conv2d(
                    dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1
                ),  # 64
                nn.Conv2d(
                    dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1
                ),  # 64
                nn.Conv2d(
                    dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1
                ),  # 32
                nn.Conv2d(
                    dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1
                ),  # 32
                nn.Conv2d(
                    dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1
                ),  # 16
                nn.Conv2d(dim_hidden, 64, 16, bias=False, stride=1, padding=0),  # 1
                nn.Linear(64, 1),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 64
                nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),  # 64
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 32
                nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),  # 32
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 16
                nn.Conv2d(in_channels, 64, 16, stride=1, padding=0),  # 1
            ]
        )

        self.act = nn.Softplus(beta=beta_softplus)
        self.alpha = alpha
        self.pos_weights = pos_weights
        if pos_weights:
            self.rectifier_fn = rectifier_fn

        if device is not None:
            self.to(device)

    def forward(self, x):
        bsize = x.shape[0]
        # assert x.shape[-1] == x.shape[-2]
        image_size = np.array([x.shape[-2], x.shape[-1]])
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [
            image_size,
            image_size // 2,
            image_size // 2,
            image_size // 4,
            image_size // 4,
            image_size // 8,
        ]

        if self.pos_weights:
            for core in self.lin[1:]:
                core.weight.data = self.rectifier_fn(core.weight.data)

        for core, res, (s_x, s_y) in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (s_x, s_y), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))

        x_scaled = nn.functional.interpolate(x, tuple(size[-1]), mode="bilinear")
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)
        # avg pooling
        y = torch.mean(y, dim=(2, 3))

        y = y.reshape(bsize, 64)
        y = self.lin[-1](y)  # (batch, 1)

        # strongly convex
        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)

        # return shape: (batch, 1)
        return y

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def grad(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.forward(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]

        return grad


# if __name__ == "__main__":
#     net = ICNN(pos_weights=True)
#     x = torch.randn((1, 3, 128, 128))
#     x = net.grad(x)
