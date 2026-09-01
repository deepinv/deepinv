from __future__ import annotations
import torch
from .optim_iterator import OptimIterator, fStep, gStep
from typing import TYPE_CHECKING

from deepinv.optim.linear import least_squares

if TYPE_CHECKING:
    from deepinv.optim import DataFidelity, Prior
    from deepinv.physics import Physics

class PIDALIteration(OptimIterator):
    r"""
    Iterator for the PIDAL (Poisson image deconvolution by augmented Lagrangian) algorithm.

    Class for a single iteration of the PIDAL algorithm for
    minimising :math:`f(x) + \lambda \regname(x)` when :math:`f` is a Poisson likelihood data fidelity and :math:`\regname` is a regularization term.

    .. math::
        x_{k+1} &= \underset{x}{\text{argmin}}\lbrace \frac{1}{2\gamma}\left\Vert Mx - z_{k} + u_{k}\right\Vert^2\rbrace \\
        z_{k+1}^1 &= \operatorname{prox}_{\text{KL}(. | y)}(Ax_{k+1} + u_{k}^1) \\
        z_{k+1}^2 &=  \operatorname{prox}_{\gamma \lambda \regname}(x_{k+1} + u_{k}^2) \\
        z_{k+1}^3 &= \text{max}(0, x_{k+1} + u_{k}^3) \\
        u_{k+1} &= u_{k} + Mx_{k+1} - z_{k+1}

    where :math:`\gamma>0` is a stepsize, :math:`M=\begin{bmatrix}A\\I\\I\end{bmatrix}` and :math:`z_{k} = \begin{bmatrix}z_{k}^1\\z_{k}^2\\z_{k}^3\end{bmatrix}`.

    """

    def __init__(self, **kwargs):
        super(PIDALIteration, self).__init__(**kwargs)

    def M(self, y: torch.Tensor, physics: Physics):
        Aty = physics.A_adjoint(y)
        return torch.cat([Aty, y, y], dim=2)

    def MT(self, x: torch.Tensor, physics: Physics):
        Ax = physics.A(x)
        return torch.cat([Ax, x, x], dim=3)

    def x_step(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        cur_params: dict,
        physics: Physics,
    ) -> torch.Tensor:

        return least_squares(
                    A=lambda y: self.M(y, physics),
                    AT=lambda x: self.MT(x, physics),
                    y=z - u,
                    gamma=None,
                    solver=cur_params["f_solver"],
                    kwargs=cur_params["f_solver_kwargs"],
                    max_iter=cur_params["f_max_iter"],
                    tol=cur_params["f_tol"],
        )

    def z_step(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        cur_prior: Prior,
        cur_params: dict,
        cur_data_fidelity: DataFidelity,
        y: torch.Tensor,
        physics: Physics
    ) -> torch.Tensor:

        h = u.shape[2] // 3

        u1 = u[:, :, 0:h, ...]
        u2 = u[:, :, h:2 * h, ...]
        u3 = u[:, :, 2 * h:3 * h, ...]

        z1 = cur_data_fidelity.prox_d(physics.A(x) + u1, y,)
        z2 = cur_prior.prox(x + u2, cur_params["g_param"], gamma=cur_params["lambda"] * cur_params["stepsize"])
        z3 = torch.clamp(x + u3, min=0)

        return torch.cat([z1, z2, z3], dim=2)

    def forward(
        self,
        X: dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        cur_params: dict,
        y: torch.Tensor,
        physics: Physics,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        r"""
        Single iteration of the PIDAL algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
        :return: Dictionary `{"est": (x, z, u), "cost": F}` containing the updated current iterate and the estimated current cost.
        """

        x, z, u = X["est"]

        x = self.f_step(z, u, cur_params, physics)
        z = self.g_step(x, u, cur_prior, cur_params, cur_data_fidelity, y, physics)
        u = u + self.M(x, physics) - z

        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            and self.cost_fn is not None
            and cur_data_fidelity is not None
            and cur_prior is not None
            else None
        )
        return {"est": (x, z, u), "cost": F}