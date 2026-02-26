from __future__ import annotations
from .optim_iterator import OptimIterator
from torch import ones_like
from torch import Tensor
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
        self._cached_row_sum = None
        self._cached_col_sum = None
        self.sinogram_shape = None

    def _get_normalizers(
        self, x: Tensor, y: Tensor, physics: LinearPhysics, eps: float
    ):
        sinogram_shape = tuple(y.shape)
        # We cache the normalizers to avoid recomputing them at every iteration
        # If the physics has changed, the shape of y should change, which will trigger a recomputation of the normalizers
        if (
            self.sinogram_shape == sinogram_shape
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
        self.sinogram_shape = sinogram_shape

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
    ):
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
