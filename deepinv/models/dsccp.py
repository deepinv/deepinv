import torch
from torch import nn
from deepinv.models import Denoiser
from .utils import get_weights_url


class DScCP(Denoiser):
    r"""
    DScCP denoiser network.

    The network architecture is based on the paper from :footcite:t:`le2024unfolded`.
    and has an unrolled architecture based on the fast Chambolle-Pock algorithm using strong convexity.
    DScCP stands for Deep Strongly Convex Chambolle Pock.

    The pretrained weights are trained with the default parameters of the network, i.e. depth=20 layers, n_channels_per_layer=64 channels.
    They can be downloaded via setting ``pretrained='download'``.

    :param int depth: depth i.e. number of convolutional layers.
    :param int n_channels_per_layer: number of channels per convolutional layer.
    :param str pretrained: 'download' to download pretrained weights, or path to local weights file.
    :param torch.device, str device: 'cuda' or 'cpu'.

    """

    def __init__(
        self, depth=20, n_channels_per_layer=64, pretrained="download", device=None
    ):
        super(DScCP, self).__init__()
        self.depth = depth
        self.n_channels_per_layer = n_channels_per_layer
        self.norm_net = 0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.depth):
            self.conv.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=n_channels_per_layer,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    dtype=torch.float,
                )
            )
            self.conv.append(
                nn.ConvTranspose2d(
                    in_channels=n_channels_per_layer,
                    out_channels=3,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    dtype=torch.float,
                )
            )
            self.conv[i * 2 + 1].weight = self.conv[i * 2].weight
            nn.init.kaiming_normal_(self.conv[i * 2].weight.data, nonlinearity="relu")

        self.mu = nn.Parameter(
            torch.ones(depth, dtype=torch.float32), requires_grad=True
        )

        # apply He's initialization
        for i in range(depth):
            nn.init.kaiming_normal_(self.conv[i].weight.data, nonlinearity="relu")

        if pretrained is not None:
            if pretrained == "download":
                url = get_weights_url(
                    model_name="dsccp", file_name="ckpt_dsccp.pth.tar"
                )
                ckpt = torch.hub.load_state_dict_from_url(
                    url,
                    map_location=lambda storage, loc: storage,
                    file_name="ckpt_dsccp.pth.tar",
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt)

        self.tol = 1e-4
        self.max_iter = 50

        if device is not None:
            self.to(device)

    def forward(self, x, sigma=0.03):
        r"""
        Run the denoiser on noisy image.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level.
        """
        x_prev = x
        x_curr = x
        u = self.conv[0](x)
        gamma = 1
        sigma = self._handle_sigma(
            sigma, batch_size=x.size(0), ndim=x.ndim, device=x.device, dtype=x.dtype
        )

        for k in range(self.depth):
            xtmp = torch.randn_like(x)
            xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
            val = 1
            for i in range(self.max_iter):
                old_val = val
                xtmp = self.conv[2 * k + 1](self.conv[2 * k](xtmp))
                val = torch.linalg.norm(xtmp.flatten())
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < self.tol:
                    break
                xtmp = xtmp / val
            tau = 0.99 / val

            alphak = 1 / torch.sqrt(1 + 2 * gamma * self.mu.data[k])
            u_ = u + tau / self.mu[k] * self.conv[k * 2](
                (1 + alphak) * x_curr - alphak * x_prev
            )
            u = torch.clamp(u_, min=-(sigma**2), max=sigma**2)
            x_ = (
                (self.mu[k] / (self.mu[k] + 1)) * x
                + (1 / (1 + self.mu[k])) * x_curr
                - (self.mu[k] / (self.mu[k] + 1)) * self.conv[k * 2 + 1](u)
            )
            x_next = torch.clamp(x_, min=0, max=1)
            x_prev = x_curr
            x_curr = x_next

        return x_curr
