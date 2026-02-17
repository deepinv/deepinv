from __future__ import annotations
from .optim_iterator import OptimIterator
import torch


class SIRTIteration(OptimIterator):
    r"""
    SIRT iteration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_row_sum = None
        self._cached_col_sum = None
        self.sinogram_shape = None

    @torch.no_grad()
    def _get_normalizers(self, x, y, physics, eps: float):
        sinogram_shape = tuple(y.shape)
        # We cache the normalizers to avoid recomputing them at every iteration
        # If the physics has changed, the shape of y should change, which will trigger a recomputation of the normalizers
        if (
            self.sinogram_shape == sinogram_shape
            and self._cached_row_sum is not None
            and self._cached_col_sum is not None
        ):
            return self._cached_row_sum, self._cached_col_sum

        ones_x = torch.ones_like(x)
        ones_y = torch.ones_like(y)

        row_sum = physics.A(ones_x)
        col_sum = physics.A_adjoint(ones_y)

        row_sum = row_sum.clamp(min=eps)
        col_sum = col_sum.clamp(min=eps)

        self._cached_row_sum = row_sum
        self._cached_col_sum = col_sum
        self.sinogram_shape = sinogram_shape

        return row_sum, col_sum

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, **kwargs
    ):
        x = X["est"][0]

        omega = (
            float(cur_params.get("stepsize", 1.0)) if cur_params is not None else 1.0
        )
        eps = float(cur_params.get("eps", 1e-8)) if cur_params is not None else 1e-8

        row_sum, col_sum = self._get_normalizers(x, y, physics, eps)

        Ax = physics.A(x)
        resid = y - Ax

        x_next = x + omega * physics.A_adjoint(resid / row_sum) / col_sum
        return {"est": (x_next,)}
