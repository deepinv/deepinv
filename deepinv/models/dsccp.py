import torch
from torch import nn
from torch.nn import functional

from .utils import get_weights_url


class DScCP(nn.Module):
    r"""
    DScCP denoiser network.

    The network architecture is based on the paper
    `Unfolded proximal neural networks for robust image Gaussian denoising <https://arxiv.org/abs/2308.03139>`_,
    and has an unrolled architecture based on the fast Chambolle-Pock algorithm using strong convexity.
    DScCP stands for Deep Strongly Convex Chambolle Pock.

    The pretrained weights are trained with the default parameters of the network, i.e. K=20 layers, F=64 channels.
    They can be downloaded via setting ``pretrained='download'``.

    :param int K: depth i.e. number of convolutional layers.
    :param int F: number of channels per convolutional layer.
    :param str pretrained: 'download' to download pretrained weights, or path to local weights file.
    :param torch.device, str device: 'cuda' or 'cpu'.
    """

    def __init__(self, K=20, F=64, pretrained="download", device=None):
        super(DScCP, self).__init__()
        self.K = K
        self.F = F
        self.norm_net = 0
        # convolution layers
        self.conv = nn.ModuleList()
        for i in range(self.K):
            self.conv.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=F,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    dtype=torch.float,
                )
            )
            self.conv.append(
                nn.ConvTranspose2d(
                    in_channels=F,
                    out_channels=3,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    dtype=torch.float,
                )
            )
            self.conv[i * 2 + 1].weight = self.conv[i * 2].weight
            nn.init.kaiming_normal_(self.conv[i * 2].weight.data, nonlinearity="relu")

        x = torch.ones(K)
        self.mu = nn.Parameter(
            torch.tensor(x, requires_grad=True, dtype=torch.float).cpu()
        )
        self.lip = torch.tensor(
            torch.ones(K), requires_grad=False, dtype=torch.float
        ).cpu()
        # apply He's initialization
        for i in range(K):
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
        for k in range(self.K):
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
            u = functional.hardtanh(u_, min_val=-(sigma**2), max_val=sigma**2)
            x_ = (
                (self.mu[k] / (self.mu[k] + 1)) * x
                + (1 / (1 + self.mu[k])) * x_curr
                - (self.mu[k] / (self.mu[k] + 1)) * self.conv[k * 2 + 1](u)
            )
            x_next = torch.clamp(x_, min=0, max=1)
            x_prev = x_curr
            x_curr = x_next

        return x_curr
