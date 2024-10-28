import torch.nn as nn
import torch

from deepinv.optim.data_fidelity import L2


class NoisyDataFidelity(nn.Module):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None, data_fidelity=None):
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

    def __init__(self, physics=None, denoiser=None, data_fidelity=L2()):
        super(DPSDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser
        self.data_fidelity = data_fidelity

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise NotImplementedError

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        with torch.enable_grad():
            x.requires_grad_(True)
            l2_loss = self.forward(x, y, sigma)

        norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        aux_x = x / 2 + 0.5
        x0_t = 2 * self.denoiser(aux_x, sigma / 2) - 1
        x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

        l2_loss = self.data_fidelity(x0_t, y, self.physics).sqrt().sum()

        return l2_loss


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

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:

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

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        return self.grad(x, y, sigma)
