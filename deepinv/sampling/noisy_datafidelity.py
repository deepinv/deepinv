import torch.nn as nn
import torch

from deepinv.optim.data_fidelity import L2

# Sam comments:
# NoisyDatafidelity inherit from DataFidelity
# but can use other d functions (default is l2) so NOT inherit from L2()

class NoisyDataFidelity(L2):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None, data_fidelity=None):
        super(NoisyDataFidelity, self).__init__()

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TODO
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


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, physics=None, denoiser=None):
        super(DPSDataFidelity, self).__init__()

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
            l2_loss = self.forward(x, y, physics, sigma)

        norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        aux_x = x / 2 + 0.5
        x0_t = 2 * self.denoiser(aux_x, sigma / 2) - 1
        x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

        l2_loss = self.d(y, physics.A(x0_t))

        return l2_loss


class SNIPSDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, denoiser=None):
        super(SNIPSDataFidelity, self).__init__()

        self.denoiser = denoiser

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        if hasattr(physics.noise_model, "sigma"):
            sigma_noise = physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        x_bar = self.physics.V_adjoint(x)
        y_bar = self.physics.U_adjoint(y)
        case = self.physics.mask > sigma_noise
        y_bar[case] = y_bar[case] / self.physics.mask[case]

        loss = y_bar - self.physics.mask * x_bar

        return loss

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:

        Sigma = physics.mask
        Sigma_T = torch.transpose(Sigma, -2, -1)

        if hasattr(physics.noise_model, "sigma"):
            sigma_noise = physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        identity = torch.ones_like(Sigma)

        tmp = torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
        tmp[tmp > 0] = 1 / tmp[tmp > 0]
        tmp[tmp == 0] = 0

        grad_norm_op = -Sigma * tmp
        grad_norm = physics.V(grad_norm_op * self.forward(x, y, sigma))

        return grad_norm


class DDRMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, denoiser=None):
        super(DDRMDataFidelity, self).__init__()

        self.denoiser = denoiser

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        if hasattr(physics.noise_model, "sigma"):
            sigma_noise = physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        x0_t = self.denoiser(x, sigma)
        x_bar = physics.V_adjoint(x0_t)
        y_bar = physics.U_adjoint(y)
        case = physics.mask > sigma_noise
        y_bar[case] = y_bar[case] / physics.mask[case]

        loss = y_bar - physics.mask * x_bar

        return loss

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:

        Sigma = physics.mask
        Sigma_T = torch.transpose(Sigma, -2, -1)

        if hasattr(physics.noise_model, "sigma"):
            sigma_noise = physics.noise_model.sigma
        else:
            sigma_noise = 0.01

        identity = torch.ones_like(Sigma)

        tmp = torch.abs(sigma_noise**2 * identity - sigma**2 * Sigma * Sigma_T)
        tmp[tmp > 0] = 1 / tmp[tmp > 0]
        tmp[tmp == 0] = 0

        grad_norm_op = -Sigma * tmp
        grad_norm = physics.V(grad_norm_op * self.forward(x, y, sigma))

        return grad_norm


class PGDMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, denoiser=None):
        super(PGDMDataFidelity, self).__init__()

        self.denoiser = denoiser

    def grad(self, x, y, physics, sigma):
        with torch.enable_grad():
            x.requires_grad_(True)
            loss = self.forward(x, y, sigma)

        norm_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad

    def forward(self, x, y, physics, sigma):
        # TODO: why normalization here?
        aux_x = x / 2 + 0.5
        x0_t = 2 * self.denoiser(aux_x, sigma / 2) - 1

        return physics.d(physics.A_dagger(y), physics.A_dagger(physics.A(x0_t)))


class ILVRDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, denoiser=None):
        super(ILVRDataFidelity, self).__init__()

        self.denoiser = denoiser

        self.data_fidelity = L2()

    def precond(self, x: torch.Tensor, physics) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        return physics.A_dagger(x)

    def diff(self, x: torch.Tensor, y: torch.Tensor, physics) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        out = physics.A(x) - y
        return out

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        y = y + sigma * torch.randn_like(y)
        return self.precond(self.diff(x, y, physics))

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        return self.d(physics(x), y + sigma * torch.randn_like(y))



class ScoreSDE(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, denoiser=None):
        super(ScoreSDE, self).__init__()

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

    def __init__(self, denoiser=None):
        super(ILVRDataFidelity, self).__init__()

        self.denoiser = denoiser

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        TBD

        :param torch.Tensor x: TBD

        :return: (torch.Tensor) TBD
        """
        raise self.physics.A_adjoint(x)

    def diff(self, x: torch.Tensor, y: torch.Tensor, physics) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        return self.precond(self.diff(x, y, physics))

    def forward(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        return self.d(physics(x), y + sigma * torch.randn_like(y))


class DDNMDataFidelity(NoisyDataFidelity):
    r"""
    TBD

    :param float sigma: TBD
    """

    def __init__(self, denoiser=None):
        super(DDNMDataFidelity, self).__init__()

        self.denoiser = denoiser

        self.data_fidelity = L2()

    def diff(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        x0_t = self.denoiser(x, sigma)
        # x0_t = torch.clip(x0_t, -1.0, 1.0)  # optional

        return y - physics.A(x0_t)
    
    
    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma, lambda_t=None) -> torch.Tensor:
        # TODO: DDNM needs the scaled residual
        
        residuals = self.diff(x, y, physics, sigma)
        A_dagger_residual = physics.A_dagger(residuals)
        
        # Project A_dagger_residual into the spectral space using V^T
        V_T_A_dagger_residual = physics.V_adjoint(A_dagger_residual)
        
        # Scale V_T_A_dagger_residual with Sigma_t. To do this we use Lambda_t in the spectral space
        scaled_V_T_A_dagger_residual = V_T_A_dagger_residual * lambda_t
        
        guidance = (-1 / sigma**2) 
        
        # Project back to the original space using U
        norm_grad = guidance * self.physics.U(V_T_A_dagger_residual)  # Shape: (B, C, H, W)
        
        return norm_grad

    def grad_simplified(self, x: torch.Tensor, y: torch.Tensor, physics, sigma, lambda_t=None) -> torch.Tensor:
        
        meas_error = self.diff(x, y, physics, sigma)
        
        guidance = (-1 / sigma**2) 
        grad_norm = guidance * lambda_t * self.physics.A_dagger(meas_error)
 
        return grad_norm
