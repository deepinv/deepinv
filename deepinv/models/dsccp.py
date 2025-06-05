import os
import torch
from torch import nn
from torch.nn import functional


class DScCP(nn.Module):
    r"""
    DScCP denoiser network.

    The network architecture is based on the paper
    `Unfolded proximal neural networks for robust image Gaussian denoising <https://arxiv.org/pdf/2308.03139>`_,
    and has an unrolled architecture.

    # TODO
    The network takes into account the noise level of the input image, which is encoded as an additional input channel.

    A pretrained network for RGB images with K=20, and F=64
    can be downloaded via setting ``pretrained='download'``.

    :param int K: number of layers.
    :param int F: number of convolutional filters.
    :param str device: gpu or cpu.

    """

    def __init__(self, K=20, F=64, device="cpu"):
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

        outputdir = "checkpoints/unfolded_ScCP_ver2/unfolded_ScCP_ver2_F64_K20_batchsize200_param_34580_data_vary"
        checkpoint_path = os.path.join(outputdir, "checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["Net"])

    def forward(self, x, sigma=0.03):
        K = self.K
        # Initialization
        x_prev = x
        x_curr = x
        u = self.conv[0](x)
        gamma = 1
        for k in range(K):
            tol = 1e-4
            max_iter = 50
            xtmp = torch.randn_like(x)
            xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
            val = 1
            for i in range(max_iter):
                old_val = val
                xtmp = self.conv[2 * k + 1](self.conv[2 * k](xtmp))
                val = torch.linalg.norm(xtmp.flatten())
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
            tau = 0.99 / val

            alphak = 1 / torch.sqrt(1 + 2 * gamma * self.mu.data[k])
            u = functional.hardtanh(
                u
                + tau
                / self.mu[k]
                * self.conv[k * 2]((1 + alphak) * x_curr - alphak * x_prev),
                min_val=-(sigma**2),
                max_val=sigma**2,
            )
            x_next = torch.clamp(
                (self.mu[k] / (self.mu[k] + 1)) * x
                + (1 / (1 + self.mu[k])) * x_curr
                - (self.mu[k] / (self.mu[k] + 1)) * self.conv[k * 2 + 1](u),
                min=0,
                max=1,
            )
            x_prev = x_curr
            x_curr = x_next

        # K-th layer
        return x_curr
