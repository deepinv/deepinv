import torch.nn as nn
import torch

from deepinv.optim.data_fidelity import L2


class NoisyDataFidelity(nn.Module):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(NoisyDataFidelity, self).__init__()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y))


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(DPSDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:

        with torch.enable_grad():
            x.requires_grad_(True)

            aux_x = x / 2 + 0.5
            x0_t = 2 * self.denoiser(aux_x, sigma / 2) - 1

            x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

            # DPS
            l2_loss = self.data_fidelity(x0_t, y, self.physics).sqrt().sum()

        norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad



class DDRMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(DDRMDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise NotImplementedError



    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:

        x_bar = physics.V_adjoint(x)

        case2 = torch.logical_and(case, (self.sigmas[t] < nsr))
        case3 = torch.logical_and(case, (self.sigmas[t] >= nsr))

        # n = np.prod(mask.shape)
        # print(f'case: {case.sum()/n*100:.2f}, case2: {case2.sum()/n*100:.2f}, case3: {case3.sum()/n*100:.2f}')

        mean = (
            x_bar
            + c * self.sigmas[t] * (x_bar_prev - x_bar) / self.sigmas[t - 1]
        )
        mean[case2] = (
            x_bar[case2]
            + c * self.sigmas[t] * (y_bar[case2] - x_bar[case2]) / nsr[case2]
        )
        mean[case3] = (1.0 - self.etab) * x_bar[case3] + self.etab * y_bar[
            case3
        ]

        std = torch.ones_like(x_bar) * self.eta * self.sigmas[t]
        std[case3] = (
            self.sigmas[t] ** 2 - (nsr[case3] * self.etab).pow(2)
        ).sqrt()

        x_bar = mean + std * torch.randn_like(x_bar)
        x_bar_prev = x_bar.clone()
        # denoise
        x = self.denoiser(physics.V(x_bar), self.sigmas[t])

        return x