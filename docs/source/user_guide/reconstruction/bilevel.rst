.. _bilevel:

Bilevel learning with MAID
==========================

DeepInverse implements the Method of Adaptive Inexact Descent (MAID) for
bilevel hyperparameter learning. The outer loop is solver-agnostic. Everything
that depends on how the lower level is solved, and on which a posteriori bound
certifies the hypergradient error, lives behind a
:class:`~deepinv.optim.bilevel.HypergradientOracle`.

Problem form
------------

.. math::

    \min_\theta f(\theta)
    := \frac{1}{m}\sum_{i=1}^m g_i(\hat x_i(\theta)) + r(\theta)

    \text{s.t.}\quad
    \hat x_i(\theta)
    := \arg\min_x h_i(x, \theta).

The functions :math:`g_i` measure reconstruction quality (typically
:math:`\|x - x_i^\star\|^2`). The functions :math:`h_i` are variational
reconstruction objectives: data fidelity plus a parameterised prior. The
parameter :math:`\theta` is what MAID learns.

References: Salehi, Mukherjee, Roberts and Ehrhardt, SIAM J. Math. Data Sci.
2025 (arXiv 2308.10098); saddle-point extension in Bogensperger, Ehrhardt,
Pock, Salehi and Wong (arXiv 2412.06436).

Oracles
-------

.. list-table::
   :header-rows: 1
   :widths: 28 22 25 25

   * - Oracle
     - Lower level
     - Hypergradient
     - Error certificate
   * - :class:`~deepinv.optim.bilevel.TikhonovWeightOracle`
     - DeepInverse ``GD`` / ``PGD`` / ``FISTA``
       with residual stopping
     - IFT + CG
     - Theorem 2.1 (certified)
   * - :class:`~deepinv.optim.bilevel.SmoothHypergradientOracle`
     - Hand-rolled GD on quadratic /
       nonquadratic unit-test problems
     - IFT + CG
     - Theorem 2.1 (certified)
   * - :class:`~deepinv.optim.bilevel.SaddleHypergradientOracle`
     - Hand-rolled PDHG on a quadratic saddle
       (``PDCP`` not wired yet)
     - Piggyback adjoint
     - Theorem 2 of arXiv 2412.06436 (certified)
   * - :class:`~deepinv.optim.bilevel.GoalOrientedSmoothOracle`
     - Same as smooth unit-test path
     - IFT + CG
     - Dual-weighted residual estimate (non-certified)

DeepInverse optimiser residuals
-------------------------------

:mod:`deepinv.optim.bilevel.base_optim_lower` drives
:class:`~deepinv.optim.BaseOptim` subclasses with residual stopping, not
the default fixed-point residual ``||x_{k+1}-x_k||``.

.. list-table::
   :header-rows: 1

   * - Solver
     - Residual
     - Why
   * - ``GD``
     - Gradient residual
       ``||grad data + lambda grad prior||``
     - Smooth objective; strong convexity gives
       ``||x-xhat|| <= residual / mu``
   * - ``PGD``, ``FISTA``
     - Proximal residual
       ``||x - prox(x - gamma grad f)|| / gamma``
     - Stationarity measure for composite
       ``f + lambda g``; reduces to the gradient
       residual when the prox is a gradient step
   * - ``ADMM``, ``DRS``, ``PDCP``, ...
     - Not wired
     - No honest residual criterion is exposed yet;
       do not guess

Warm starts pass the previous reconstruction through the ``init`` argument of
:meth:`~deepinv.optim.BaseOptim.forward` (for FISTA both primal and
extrapolated points are set).

Unifying fact (Lemma 1 of arXiv 2412.06436): if :math:`\Phi` is
:math:`\mu`-strongly convex with minimiser :math:`x_\star`, then

.. math::

    \|x_\star - x\| \le \frac{1}{\mu}\,\|\nabla\Phi(x)\|.

Every residual-based distance bound used by an oracle is a variant of this.

Certified versus non-certified
------------------------------

``oracle.certified`` is ``True`` only for a bound proven in a citable paper
with known constants (in particular a known strong-convexity modulus). An
estimated ``mu`` that is too large makes every bound too small, which is the
dangerous direction: MAID can accept a non-descent direction while the line
search still appears to succeed.

Non-certified oracles require an explicit opt-in:

.. code-block:: python

    from deepinv.optim.bilevel import MAID, GoalOrientedSmoothOracle

    oracle = GoalOrientedSmoothOracle(problem)  # certified is False
    maid = MAID(oracle, allow_uncertified=True)

Convergence in the sense of Theorem 3.19 of the MAID paper is proven only
when **both** of the following hold:

1. ``oracle.certified`` is ``True``, and
2. ``MAIDConfig.check_descent_direction`` is ``True``
   (the a priori test ``omega <= (1 - eta)||z||``).

By contributor decision the default is ``check_descent_direction=False``:
every step the line search accepts still provably decreases the true upper
level (Lemma 3.5), but existence of a valid step size is no longer guaranteed
a priori. Backtracking failure is the a posteriori detector and is counted in
``history["n_backtrack_failures"]``.

Goal-oriented error estimate
----------------------------

The certified Theorem 2.1 bound is often loose (median factor about 4 to 25
on quadratic tests, up to thousands). The goal-oriented estimator forms

.. math::

    z - \nabla f
    \approx J^\top H^{-1} r
    - J^\top H^{-1} G H^{-1} \nabla_x h(\bar x),

with a default ``safety_factor=1.25`` and ``cg_budget=5``. Measured ratios
(estimate / true error):

**Quadratic** (dim 200, 72 configs, supervisor):

.. list-table::
   :header-rows: 1

   * - estimator
     - median ratio
     - max
     - under-estimates
   * - certified ``omega``
     - 24.5
     - 38.0
     - 0/72
   * - DWR raw, budget 5
     - 0.994
     - 1.015
     - 59/72
   * - DWR × 1.25
     - 1.25
     - 1.27
     - 0/72

**Nonquadratic** (log-cosh prior, data-scale sweep, raw DWR budget 5):

.. list-table::
   :header-rows: 1

   * - data scale
     - sech² spread
     - DWR median
     - DWR min
     - safety needed
   * - 1.0
     - 0.06
     - 1.0014
     - 0.9790
     - 1.02
   * - 5.0
     - 0.71
     - 1.0021
     - 0.9792
     - 1.02
   * - 20.0
     - 0.9996
     - 0.9956
     - 0.9791
     - 1.02

The default safety factor 1.25 is about twelve times the observed requirement
across these regimes. The estimator remains non-certified.

When to use MAID
----------------

MAID spends lower-level iterations on line-search trials that a fixed-accuracy
baseline does not. That overhead only pays when a tight lower-level solve is
expensive.

**Cheap lower level (fixed accuracy wins).** On a 32x32 Tikhonov inpainting
problem solved by DeepInverse ``GD`` with residual stopping, a residual of
``1e-4`` costs about one gradient step per solve. Matched outer-iteration
count (25 steps):

.. list-table::
   :header-rows: 1

   * - method
     - f final
     - BaseOptim iters
     - wall
   * - MAID
     - 47.2152
     - 246
     - 0.086 s
   * - fixed accuracy ``1e-4``
     - 47.2307
     - 26
     - 0.008 s

MAID recorded 16 backtracking failures. Those failed trials, plus successful
line-search probes, are why the BaseOptim count is roughly 9 times larger for
essentially the same upper-level value. Prefer fixed accuracy in this regime.

**Expensive lower level (MAID wins).** On the quadratic least-squares bilevel
problem of Salehi et al. (section 4.1) with lower-level condition number 30,
both methods stop at a relative gap of ``5e-3`` to the closed-form optimum:

.. list-table::
   :header-rows: 1

   * - method
     - GD iters
     - outer steps
     - wall
   * - MAID
     - 60 062
     - 6
     - 0.75 s
   * - fixed accuracy ``1e-4``
     - 129 383
     - 23
     - 1.54 s

**Crossover.** Sweeping the lower-level condition number (``n=80``, ``d=4``,
target relative gap ``5e-3``, fixed residual ``1e-4``):

.. list-table::
   :header-rows: 1

   * - condition number
     - GD MAID
     - GD fixed
     - ratio MAID/fixed
   * - 2
     - 81
     - 56
     - 1.45
   * - 5
     - 861
     - 853
     - 1.01
   * - 10
     - 4 606
     - 5 847
     - 0.79
   * - 20
     - 17 334
     - 40 691
     - 0.43
   * - 30
     - 60 062
     - 129 383
     - 0.46

The scientific claim is therefore: MAID is faster once a tight lower-level
solve costs more than roughly a few thousand gradient steps, and slower below
that. Ill-conditioned physics (motion blur, tomography), weak regularisation,
large images, and nonsmooth priors with many proximal steps are the intended
regime. The gallery example recomputes the flagship, the crossover table and
the inpainting counterpoint from scratch.

Minimal usage
-------------

.. code-block:: python

    import deepinv as dinv
    from deepinv.optim.bilevel import (
        MAID,
        MAIDConfig,
        TikhonovWeightOracle,
        TikhonovWeightProblem,
    )

    physics = dinv.physics.Denoising(
        noise_model=dinv.physics.GaussianNoise(0.1)
    )
    # y, x_star: measurement and ground truth
    problem = TikhonovWeightProblem(
        physics=physics, y=y, x_star=x_star, solver="GD"
    )
    oracle = TikhonovWeightOracle(problem)
    config = MAIDConfig(alpha0=0.5, max_iter=30, g_convex=True)
    theta, history = MAID(oracle, config).run(theta0)

    # history keys include f_exact, z_norm, n_lower_solves,
    # n_hypergradients, n_backtrack_failures
    # problem.n_gd_iters counts BaseOptim residual iterations

See also the gallery example
:ref:`sphx_glr_auto_examples_optimization_demo_maid_bilevel.py`.

API
---

See :mod:`deepinv.optim.bilevel` and the optim API page.
