"""Non-certified hypergradient error estimators.

Goal-oriented (dual-weighted residual) estimation
-------------------------------------------------
For a smooth lower level with IFT hypergradient ``z = -J^T q`` where
``H q ≈ grad g(xbar)``, the error in the functional admits the
goal-oriented decomposition (exact when ``h`` and ``g`` are quadratic)

    z - grad f
        = J^T H^{-1} r
          - J^T H^{-1} (grad g(xbar) - grad g(xhat))

and, with the Newton identity ``xbar - xhat = H^{-1} grad_x h(xbar)``
(exact for quadratic ``h``) and ``grad g(xbar) - grad g(xhat) = G (xbar - xhat)``
(exact for quadratic ``g``),

    z - grad f
        = J^T H^{-1} r
          - J^T H^{-1} G H^{-1} grad_x h(xbar),

where ``r = grad g(xbar) - H q`` is the linear-solve residual. Both terms
are adjoint solves against residuals already available after the main IFT
solve. This transfers the dual-weighted-residual idea from finite-element
goal-oriented error estimation (see e.g. Becker and Rannacher; also PDE
bilevel optimal control, arXiv 1907.04285) to hypergradient error. It is a
transfer of an established technique, not an invention.

Measurement (quadratic lower level, dim 200, 72 configurations; supervisor
sweep)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ratio is estimate / true error (1.0 is exact; below 1.0 under-estimates).

    certified omega              median 24.5   max 38.0   under 0/72
    DWR raw, CG budget 5         median 0.994  max 1.015  under 59/72
    DWR raw, CG budget 25        median 1.000  max 1.000  under 33/72
    DWR raw, CG budget 50        median 1.000  max 1.000  under 3/72
    DWR * 1.25, budgets 5..100   median 1.25   max 1.27   under 0/72

Raw DWR under-estimates frequently (59 of 72 at budget 5) but by at most
about 0.6 percent. The default ``safety_factor=1.25`` leaves roughly 40
times headroom over that worst observed violation and removed every
under-estimate at every budget tested. The factor is justified by that
margin, not by taste.

CG budget defaults to 5: already within 1.5 percent of exact; raising it to
100 bought nothing on the quadratic sweep.

Nonquadratic measurement
^^^^^^^^^^^^^^^^^^^^^^^^
On a smooth nonquadratic lower level (quadratic data term plus
``beta * sum log(cosh(x_i))`` and ``gamma`` strongly convex Tikhonov), a
36-configuration sweep (3 seeds x 3 scales x 2 eps x 2 delta) gave

    DWR raw sf=1.0     under 24/36  median ratio 0.985  min 0.897
    DWR sf=1.25        under  0/36  median ratio 1.231  min 1.121
    DWR sf=2.0         under  0/36  median ratio 1.970  min 1.793

So the default ``safety_factor=1.25`` remains conservative on that
nonquadratic class as well. The decomposition is still only first-order
exact for nonquadratic ``h``; a different nonlinearity can require a
larger factor. The estimator stays non-certified: use it with
``allow_uncertified=True`` and re-measure under-estimation rate when the
lower-level structure changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .cg_utils import CGResult, cg_recycle, cg_solve


@dataclass
class GoalOrientedEstimator:
    r"""Goal-oriented estimate of ``||z - grad f||`` (non-certified).

    :param float safety_factor: multiplies the raw DWR norm. Default 1.25,
        chosen so that on the quadratic dim-200 sweep the estimate never
        under-estimated (raw under-rate 59/72 at budget 5, worst shortfall
        about 0.6 percent; 1.25 is about 40 times that shortfall).
    :param int cg_budget: CG iterations per dual-weighted residual solve.
        Default 5.
    :param bool recycle_krylov: if True, Galerkin-project onto the main
        adjoint CG directions before the budgeted CG continuation.
    """

    safety_factor: float = 1.25
    cg_budget: int = 5
    recycle_krylov: bool = True

    def __post_init__(self) -> None:
        if self.safety_factor < 1.0:
            raise ValueError(
                f"safety_factor must be >= 1 (got {self.safety_factor}); "
                "values below 1 systematically under-estimate."
            )
        if self.cg_budget < 1:
            raise ValueError(f"cg_budget must be >= 1, got {self.cg_budget}")

    @property
    def certified(self) -> bool:
        return False

    def estimate(
        self,
        *,
        hess_mv: Callable[[torch.Tensor], torch.Tensor],
        mixed_jac_T_mv: Callable[[torch.Tensor], torch.Tensor],
        grad_x_h: torch.Tensor,
        residual_rhs: torch.Tensor,
        hess_g_mv: Callable[[torch.Tensor], torch.Tensor],
        main_cg: CGResult | None = None,
    ) -> float:
        r"""Return ``safety_factor * || term_A + term_B ||``.

        ``residual_rhs`` is ``grad g(xbar) - H q`` (experiment convention).
        ``hess_g_mv`` applies the Hessian of the upper-level loss in ``x``
        (for quadratic ``g(x) = ||A1 x - b1||^2`` this is ``v |-> 2 A1^T A1 v``).
        """
        directions = (
            main_cg.directions
            if (self.recycle_krylov and main_cg is not None)
            else []
        )

        # s1 ≈ H^{-1} grad_x h
        s1 = self._solve(hess_mv, grad_x_h, directions)
        # s2 ≈ H^{-1} G s1
        s2 = self._solve(hess_mv, hess_g_mv(s1), directions)
        # s3 ≈ H^{-1} r
        s3 = self._solve(hess_mv, residual_rhs, directions)

        # termA = J^T s3, termB = -J^T s2
        err_vec = mixed_jac_T_mv(s3) - mixed_jac_T_mv(s2)
        raw = float(err_vec.norm().item())
        return self.safety_factor * raw

    def _solve(
        self,
        hess_mv: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        directions: list[torch.Tensor],
    ) -> torch.Tensor:
        if directions:
            return cg_recycle(
                hess_mv, rhs, directions, max_iter=self.cg_budget
            ).x
        return cg_solve(
            hess_mv, rhs, tol=None, max_iter=self.cg_budget
        ).x
