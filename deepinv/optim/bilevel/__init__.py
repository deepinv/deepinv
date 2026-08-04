"""Bilevel optimisation: MAID and hypergradient oracles.

The outer method is MAID. Solver-specific lower levels and a posteriori
error certificates live behind :class:`HypergradientOracle`.

Two published instantiations are included:

* **Smooth** (Salehi et al., SIAM J. Math. Data Sci. 2025 / arXiv 2308.10098):
  gradient residual + IFT/CG, Theorem 2.1 bound.
* **Saddle point** (Bogensperger et al., arXiv 2412.06436):
  PDHG + piggyback adjoints, Theorem 2 bound from residual distances 6a, 6b.

Both rest on the same strong-convexity distance fact: for a ``mu``-strongly
convex ``Phi`` with minimiser ``x_star``,
``||x_star - x|| <= ||grad Phi(x)|| / mu``. See :mod:`deepinv.optim.bilevel.oracle`.

``certified`` is True only for a bound proven in one of those papers with
known constants. Non-certified bounds require an explicit opt-in.
"""

from .maid import MAID, MAIDConfig
from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
    strong_convexity_distance_bound,
)
from .quadratic_ls import QuadraticBilevelLS
from .saddle import (
    QuadraticSaddleProblem,
    SaddleHypergradientOracle,
    dual_distance_bound,
    primal_distance_bound,
    saddle_bound_constants,
    saddle_hypergradient_error_bound,
)
from .smooth import (
    SmoothHypergradientOracle,
    hypergradient_error_bound,
    inexact_gradient,
    inexact_gradient_from_oracle,
    smooth_hypergradient_error_bound,
)

__all__ = [
    "MAID",
    "MAIDConfig",
    "HypergradientOracle",
    "HypergradientState",
    "LowerLevelState",
    "strong_convexity_distance_bound",
    "QuadraticBilevelLS",
    "SmoothHypergradientOracle",
    "hypergradient_error_bound",
    "smooth_hypergradient_error_bound",
    "inexact_gradient",
    "inexact_gradient_from_oracle",
    "QuadraticSaddleProblem",
    "SaddleHypergradientOracle",
    "primal_distance_bound",
    "dual_distance_bound",
    "saddle_hypergradient_error_bound",
    "saddle_bound_constants",
]
