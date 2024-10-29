import torch.nn as nn
import torch

from deepinv.optim.data_fidelity import L2

# Sam comments:
# NoisyDatafidelity inherit from DataFidelity
# but can use other d functions (default is l2) so NOT inherit from L2()

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


    def diff(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics,  sigma) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y))

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD
        :param torch.Tensor y: TBD

        :return: (torch.Tensor) TBD
        """
        return self.grad(x, y, physics, sigma)


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(DPSDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise NotImplementedError

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
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

        l2_loss = self.data_fidelity(x0_t, y, physics, sigma).sqrt().sum()

        return l2_loss


class SNIPSDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(SNIPSDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        if hasattr(self.physics.noise_model, "sigma"):
            sigma_noise = self.physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        x_bar = self.physics.V_adjoint(x)
        y_bar = self.physics.U_adjoint(y)
        case = self.physics.mask > sigma_noise
        y_bar[case] = y_bar[case] / self.physics.mask[case]

        loss = y_bar - self.physics.mask * x_bar

        return loss

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:

        Sigma = self.physics.mask
        Sigma_T = torch.transpose(Sigma, -2, -1)

        if hasattr(self.physics.noise_model, "sigma"):
            sigma_noise = self.physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        identity = (
            torch.eye(n=Sigma.size(-2), m=Sigma.size(-1), device=x.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        identity = torch.ones_like(Sigma)

        tmp = torch.pinverse(
            torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
        )

        tmp = torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
        tmp[tmp > 0] = 1 / tmp[tmp > 0]
        tmp[tmp == 0] = 0

        grad_norm_op = -Sigma * tmp
        grad_norm = self.physics.V(grad_norm_op * self.forward(x, y, sigma))

        return grad_norm


class DDRMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(DDRMDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        if hasattr(self.physics.noise_model, "sigma"):
            sigma_noise = self.physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        x0_t = self.denoiser(x, sigma)
        x_bar = self.physics.V_adjoint(x0_t)
        y_bar = self.physics.U_adjoint(y)
        case = self.physics.mask > sigma_noise
        y_bar[case] = y_bar[case] / self.physics.mask[case]

        loss = y_bar - self.physics.mask * x_bar

        return loss

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:

        Sigma = self.physics.mask
        Sigma_T = torch.transpose(Sigma, -2, -1)

        if hasattr(self.physics.noise_model, "sigma"):
            sigma_noise = self.physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        identity = (
            torch.eye(n=Sigma.size(-2), m=Sigma.size(-1), device=x.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        identity = torch.ones_like(Sigma)

        tmp = torch.pinverse(
            torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
        )

        tmp = torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
        tmp[tmp > 0] = 1 / tmp[tmp > 0]
        tmp[tmp == 0] = 0

        grad_norm_op = -Sigma * tmp
        grad_norm = self.physics.V(grad_norm_op * self.forward(x, y, sigma))

        return grad_norm


class PGDMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(PGDMDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

    def grad(self, x, y, sigma):
        with torch.enable_grad():
            x.requires_grad_(True)
            loss = self.forward(x, y, sigma)

        norm_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad

    def forward(self, x, y, sigma):
        aux_x = x / 2 + 0.5
        x0_t = 2 * self.denoiser(aux_x, sigma / 2) - 1
        mat = self.physics.A_dagger(y) - self.physics.A_dagger(self.physics.A(x0_t))
        mat_x = (mat.detach() * x0_t).sum()

        return mat_x


class ILVRDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(ILVRDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        return self.physics.A_dagger(x)

    def diff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        out = self.physics.A(x) - y
        return out

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        y = y + sigma * torch.randn_like(y)
        return self.precond(self.diff(x, y))

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        return self.grad(x, y, sigma)


class ScoreSDE(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(ScoreSDE, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise self.physics.A_adjoint(x)

    def diff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return self.physics.A(x) - y

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        y = y + sigma * torch.randn_like(y)
        return self.precond(self.diff(x, y))

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        return self.grad(x, y, sigma)


class ScoreALD(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(ILVRDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise self.physics.A_adjoint(x)

    def diff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return self.physics.A(x) - y

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        return self.precond(self.diff(x, y))

    def forward(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        return self.grad(x, y, sigma)


class DDNMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(DDNMDataFidelity, self).__init__()

        self.physics = physics
        self.denoiser = denoiser

        self.data_fidelity = L2()

    def diff(self, x: torch.Tensor, y: torch.Tensor, sigma) -> torch.Tensor:
        aux_x = x / 2 + 0.5
        x0_t = 2 * self.denoiser(aux_x, sigma / 2) - 1
        x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

        return y - self.physics.A(x0_t)

    def grad(self, x: torch.Tensor, y: torch.Tensor, sigma, lambda_t) -> torch.Tensor:

        meas_error = self.diff(x, y, sigma)

        # pinverse_singular = 1.0 / self.physics.mask
        # pinverse_singular[self.physics.mask == 0] = 0

        # if hasattr(self.physics.noise_model, "sigma"):
        #     sigma_noise = self.physics.noise_model.sigma
        # else:
        #     sigma_noise = 0.01

        # Lambda_t = torch.ones_like(self.physics.mask)
        # case = sigma < a * sigma_noise * pinverse_singular
        # Lambda_t[case] = self.physics.mask * sigma * (1 - eta ** 2) ** 0.5 / a / sigma_noise

        guidance = -1 / sigma**2

        # grad_norm = guidance * self.physics.V(lambda_t * self.physics.V_adjoint(self.physics.A_dagger(meas_error)))

        # Define Sigma_t
        # If Sigma_t is channel-wise scaling (diagonal per channel)
        # Sigma_t = torch.tensor([lambda_t]).view(x.shape[0], x.shape[1], 1, 1)  # Shape C x 1 x 1

        grad_norm = guidance * lambda_t * self.physics.A_dagger(meas_error)

        return grad_norm
