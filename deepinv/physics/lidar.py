import torch
from deepinv.physics.forward import Physics
from deepinv.physics.noise import PoissonNoise


class SinglePhotonLidar(Physics):
    r"""
    Single photon lidar operator for depth ranging.

    See https://ieeexplore.ieee.org/abstract/document/9127841 for a review of this imaging method.

    The forward operator is given by

    .. math::
        y_{i,j,t} = \mathcal{P}(h(t-d_{i,j}) r_{i,j} + b_{i,j})

    where :math:`\mathcal{P}` is the Poisson noise model, :math:`h(t)` is a Gaussian impulse response function at
    time :math:`t`, :math:`d_{i,j}` is the depth of the scene at pixel :math:`(i,j)`,
    :math:`r_{i,j}` is the intensity of the scene at pixel :math:`(i,j)` and :math:`b_{i,j}` is the background noise
    at pixel :math:`(i,j)`.

    For a pixel grid of size (H,W) and batch size B, the signals have size (B, 3, H, W), where the first channel
    contains the depth of the scene :math:`d`, the second channel contains the intensity of the scene :math:`r` and
    the third channel contains the per pixel background noise levels :math:`b`.

    :param float sigma: Standard deviation of the Gaussian impulse response function.
    :param int bins: Number of histogram bins per pixel.
    :param str device: Device to use (gpu or cpu).
    :param torch.Generator rng: (optional) a pseudorandom random number generator for
        the Poisson noise model :class:`deepinv.physics.PoissonNoise`
    """

    def __init__(self, sigma=1.0, bins=50, device="cpu", rng: torch.Generator = None):
        super().__init__()

        self.T = bins
        self.grid = torch.meshgrid(torch.arange(bins), indexing="ij")[0].to(device)
        self.sigma = torch.nn.Parameter(
            torch.tensor(sigma, device=device), requires_grad=False
        )
        self.noise_model = PoissonNoise(rng=rng)

        h = ((self.grid - 3 * sigma) / self.sigma).pow(2)
        h = torch.exp(-h / 2.0)
        h = h[: int(6 * sigma)]
        h = h / h.sum()
        self.irf = h.unsqueeze(0).unsqueeze(0)  # set impulse response function
        self.grid = self.grid.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def A(self, x, **kwargs):
        r"""
        Applies the forward operator.

        Input is of size (B, 3, H, W) and output is of size (B, bins, H, W)

        :param torch.Tensor x: tensor containing the depth, intensity and background noise levels.
        """

        h = ((self.grid - x[:, 0, :, :]) / self.sigma).pow(2)
        h = torch.exp(-h / 2.0)
        h = h / h.sum(dim=1, keepdim=True)
        y = x[:, 1, :, :] * h + x[:, 2, :, :]
        return y

    def A_dagger(self, y, **kwargs):
        r"""
        Applies Matched filtering to find the peaks.

        Input is of size (B, bins, H, W), output of size (B, 3, H, W).

        :param torch.Tensor y: measurements
        """
        B, T, H, W = y.shape

        # reshape to (B*H*W, 1, T)
        y = y.permute(0, 2, 3, 1).reshape(B * H * W, 1, T)

        # Apply irf using convolution
        x = torch.nn.functional.conv1d(y, self.irf, padding="same")

        # Find peak value in each channel
        _, x = torch.max(x, dim=-1, keepdim=True)
        x = x.type(torch.float32)
        offset = self.irf.shape[-1] // 2
        x -= 3 * self.sigma - offset - 0.5

        mask = torch.ones_like(y)
        grid = self.grid.squeeze(-1).squeeze(-1)  # (1, T)

        mask *= (x - 4 * self.sigma) < grid
        mask *= (x + 4 * self.sigma) > grid

        b = (y * (1 - mask)).sum(dim=-1, keepdim=True)
        r = y.sum(dim=-1, keepdim=True) - b
        b /= T

        x = torch.stack([x, r, b], dim=-1)

        # reshape to (B, 3, H, W)
        x = x.reshape(B, H, W, 3).permute(0, 3, 1, 2)

        return x


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import deepinv as dinv
#
#     bins = 40
#     device = "cuda:0"
#     physics = SinglePhotonLidar(bins=bins, device=device)
#
#     x = torch.ones((1, 3, 2, 4), device=device)
#     x[:, 0, :, :] *= bins / 2
#     x[:, 1, :, :] *= 300
#     x[:, 2, :, :] *= 1
#
#     y = physics(x)
#     xhat = physics.A_dagger(y)
#
#     y0 = y[0, :, 0, 0].detach().cpu().numpy()
#     plt.plot(y0)
#     plt.show()
#
#     print(f"MSE {dinv.metric.MSE()(x, xhat)}")
