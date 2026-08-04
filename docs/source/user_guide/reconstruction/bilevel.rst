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
   * - :class:`~deepinv.optim.bilevel.SmoothHypergradientOracle`
     - Gradient residual
       :math:`\|\nabla_x h\|\le \varepsilon\mu`
     - IFT + CG
     - Theorem 2.1 (certified)
   * - :class:`~deepinv.optim.bilevel.SaddleHypergradientOracle`
     - PDHG residuals (eqs 6a, 6b)
     - Piggyback adjoint
     - Theorem 2 of arXiv 2412.06436 (certified)
   * - :class:`~deepinv.optim.bilevel.GoalOrientedSmoothOracle`
     - Same as smooth
     - IFT + CG
     - Dual-weighted residual estimate (non-certified)

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

Minimal usage
-------------

.. code-block:: python

    from deepinv.optim.bilevel import (
        MAID,
        MAIDConfig,
        QuadraticBilevelLS,
        SmoothHypergradientOracle,
    )

    problem = QuadraticBilevelLS(A1, A2, A3, b1, b2)
    oracle = SmoothHypergradientOracle(problem)
    config = MAIDConfig(alpha0=1e-2, max_iter=50, g_convex=True)
    theta, history = MAID(oracle, config).run(theta0)

    # history keys include f_exact, z_norm, n_lower_solves,
    # n_hypergradients, n_backtrack_failures

See also the gallery example
:ref:`sphx_glr_auto_examples_optimization_demo_maid_bilevel.py`.

API
---

See :mod:`deepinv.optim.bilevel` and the optim API page.
