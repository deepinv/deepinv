"""Bilevel optimisation: MAID and hypergradient oracles.

The outer method is MAID. Solver-specific lower levels and a posteriori
error certificates live behind :class:`HypergradientOracle`.

Published instantiations:

* **Smooth certified** (Salehi et al., SIAM J. Math. Data Sci. 2025 / arXiv
  2308.10098): gradient residual + IFT/CG, Theorem 2.1 bound.
* **Saddle point** (Bogensperger et al., arXiv 2412.06436): PDHG + piggyback
  adjoints, Theorem 2 bound from residual distances 6a, 6b.

Non-certified estimator:

* **Goal-oriented (DWR)** via :class:`GoalOrientedSmoothOracle`: estimates
  the hypergradient error from dual-weighted residuals. Default safety
  factor 1.25 is justified by the quadratic sweep (raw under-rate 59/72 at
  CG budget 5, max shortfall about 0.6 percent; 1.25 removes all
  under-estimates). Exact only for quadratic lower levels; see the
  nonquadratic measurement before general use.

``certified`` is True only for a bound proven in a citable paper with
known constants. Non-certified bounds require ``allow_uncertified=True``.

Minibatch accumulation of a mean over ``m`` samples lives in
:class:`MinibatchOracle`: fixed-order sequential reduction (bitwise
invariant to chunk size), mean error-bound aggregation
``omega = (1/m) sum omega_i``, and peak working memory proportional to
chunk size. See the module docstring of
:mod:`deepinv.optim.bilevel.minibatch`.
"""

from .base_optim_lower import (
    build_solver,
    gradient_residual,
    proximal_residual,
    residual_kind_for_solver,
    solve_base_optim,
)
from .estimators import GoalOrientedEstimator
from .maid import MAID, MAIDConfig, accelerated_maid_config
from .minibatch import (
    MinibatchOracle,
    make_quadratic_dataset,
    mean_error_bound,
    wrap_smooth_dataset,
)
from .nonquadratic import NonQuadraticBilevel
from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
    strong_convexity_distance_bound,
)
from .convex_ridge import (
    ConvexRidgeConfig,
    ConvexRidgePrior,
    exp_scaling,
    n_crr_params,
    pack_init_theta,
    scaling_vector,
    unpack_theta,
)
from .crr_bilevel import (
    CRRSampleOracle,
    CRRSampleProblem,
    build_crr_minibatch_oracle,
)
from .prior_learning import TikhonovWeightOracle, TikhonovWeightProblem
from .tv_baseline import (
    assert_grad_div_adjoint,
    div,
    grid_tune_tv,
    isotropic_tv,
    nabla,
    nabla_adjoint,
    primal_objective,
    recon_tv,
    solve_isotropic_tv,
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
    GoalOrientedSmoothOracle,
    SmoothHypergradientOracle,
    hypergradient_error_bound,
    inexact_gradient,
    inexact_gradient_from_oracle,
    smooth_hypergradient_error_bound,
)

from .priors import (
    ParametricPrior,
    ConvexRidgePrior2,
    LearnedTVPrior,
    ICNNPrior,
    BatchedPriorProblem,
)
from .batched import (
    BatchedCRR,
    BatchedMinibatchOracle,
    auto_batch_size,
    auto_initial_accuracy,
    auto_initial_step,
    available_memory,
    measure_sample_bytes,
)

__all__ = [
    "MAID",
    "MAIDConfig",
    "accelerated_maid_config",
    "HypergradientOracle",
    "HypergradientState",
    "LowerLevelState",
    "strong_convexity_distance_bound",
    "QuadraticBilevelLS",
    "NonQuadraticBilevel",
    "SmoothHypergradientOracle",
    "GoalOrientedSmoothOracle",
    "GoalOrientedEstimator",
    "MinibatchOracle",
    "mean_error_bound",
    "make_quadratic_dataset",
    "wrap_smooth_dataset",
    "TikhonovWeightProblem",
    "TikhonovWeightOracle",
    "assert_grad_div_adjoint",
    "div",
    "grid_tune_tv",
    "isotropic_tv",
    "nabla",
    "nabla_adjoint",
    "primal_objective",
    "recon_tv",
    "solve_isotropic_tv",
    "ConvexRidgeConfig",
    "ConvexRidgePrior",
    "n_crr_params",
    "pack_init_theta",
    "unpack_theta",
    "scaling_vector",
    "exp_scaling",
    "CRRSampleProblem",
    "CRRSampleOracle",
    "build_crr_minibatch_oracle",
    "solve_base_optim",
    "build_solver",
    "gradient_residual",
    "proximal_residual",
    "residual_kind_for_solver",
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
    "ParametricPrior",
    "ConvexRidgePrior2",
    "LearnedTVPrior",
    "ICNNPrior",
    "BatchedPriorProblem",
    "BatchedCRR",
    "BatchedMinibatchOracle",
    "auto_batch_size",
    "auto_initial_accuracy",
    "auto_initial_step",
    "available_memory",
    "measure_sample_bytes",
]
