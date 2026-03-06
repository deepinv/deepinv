from __future__ import annotations
from typing import TYPE_CHECKING
from .optim_iterator import OptimIterator
from torch import ones_like
from torch import Tensor

if TYPE_CHECKING:
    from deepinv.physics import LinearPhysics
    from deepinv.optim import DataFidelity, Prior


class SIRTIteration(OptimIterator):
    r"""
    Iterator for the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm.

    Class for a single iteration of the SIRT algorithm.

    The iteration is given by:

    .. math::
        \begin{equation*}
        x_{k+1} = x_k + \tau V A^{\top} W (y - A x_k)
        \end{equation*}
    where
    - :math:`\tau` is a stepsize parameter which should satisfy :math:`0 < \tau < 2`
    - :math:`W = \mathrm{diag}\left(\frac{1}{\sum_{i}a_{ij}}\right)`, a diagonal matrix where each element is the inverse of the row sums of :math:`A`,
    - :math:`V = \mathrm{diag}\left(\frac{1}{\sum_{j}a_{ij}}\right)`, a diagonal matrix where each element is the inverse of the column sums of :math:`A`.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("_cached_row_sum", None)
        self.register_buffer("_cached_col_sum", None)
        self.register_buffer("_sinogram_ptr", None)

    def _get_normalizers(
        self, x: Tensor, y: Tensor, physics: LinearPhysics, eps: float
    ) -> tuple[Tensor, Tensor]:
        """Compute the normalizing factors for the SIRT iteration.

        Args:
            :param torch.Tensor x: Current iterate :math:`x_k`.
            :param torch.Tensor y: Input data.
            :param deepinv.physics.LinearPhysics physics: Instance of the linear physics modeling the observation.
            :param float eps: Small constant to prevent division by zero.
            :return: Tuple containing the row and column normalizers.
        """
        # We cache a pointer to the sinogram to recompute normalizers only when the physics changes.
        if self._sinogram_ptr is None:
            self._sinogram_ptr = y.data_ptr()

        # We cache the normalizers to avoid recomputing them at every iteration
        # If the physics has changed, the shape of y should change, which will trigger a recomputation of the normalizers
        if (
            self._sinogram_ptr == y.data_ptr()
            and self._cached_row_sum is not None
            and self._cached_col_sum is not None
        ):
            return self._cached_row_sum, self._cached_col_sum

        ones_x = ones_like(x)
        ones_y = ones_like(y)

        row_sum = physics.A(ones_x)
        col_sum = physics.A_adjoint(ones_y)

        row_sum = row_sum.clamp(min=eps)
        col_sum = col_sum.clamp(min=eps)

        self._cached_row_sum = row_sum
        self._cached_col_sum = col_sum

        return row_sum, col_sum

    def forward(
        self,
        X: dict,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        cur_params: dict,
        y: Tensor,
        physics: LinearPhysics,
        **kwargs,
    ) -> dict:
        """Computes a single iteration of the SIRT algorithm.

        Args:
            :param dict X: Dictionary containing the current iterate and the estimated cost.
            :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
            :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
            :param dict cur_params: Dictionary containing the current parameters of the algorithm.
            :param torch.Tensor y: Input data.
            :param deepinv.physics.Physics physics: Instance of the physics modeling the observation.
            :return: Dictionary `{"est": (x,), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x = X["est"][0]
        k = 0 if "it" not in X else X["it"]

        tau = float(cur_params.get("stepsize", 1.0)) if cur_params is not None else 1.0
        eps = float(cur_params.get("eps", 1e-8)) if cur_params is not None else 1e-8

        row_sum, col_sum = self._get_normalizers(x, y, physics, eps)

        resid = y - physics.A(x)

        x_next = x + tau * physics.A_adjoint(resid / row_sum) / col_sum

        F = (
            self.cost_fn(x_next, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            and self.cost_fn is not None
            and cur_data_fidelity is not None
            else None
        )

        return {"est": (x_next,), "cost": F, "it": k + 1}
