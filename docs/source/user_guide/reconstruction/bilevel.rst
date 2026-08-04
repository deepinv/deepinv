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

The outer-step column is the stronger result. Each upper-level iteration
costs a hypergradient (a full linear solve plus an adjoint). At condition
number 30 MAID uses 6 outer steps against 23 for fixed accuracy, a factor of
nearly 4 on the expensive operation. The gradient-step column (ratio 0.46)
understates that saving.

**Crossover.** Sweeping the lower-level condition number (``n=80``, ``d=4``,
target relative gap ``5e-3``, fixed residual ``1e-4``):

.. list-table::
   :header-rows: 1

   * - condition number
     - GD MAID
     - GD fixed
     - ratio GD
     - outer MAID
     - outer fixed
   * - 2
     - 81
     - 56
     - 1.45
     - 2
     - 2
   * - 5
     - 861
     - 853
     - 1.01
     - 3
     - 5
   * - 10
     - 4 606
     - 5 847
     - 0.79
     - 4
     - 9
   * - 20
     - 17 334
     - 40 691
     - 0.43
     - 5
     - 16
   * - 30
     - 60 062
     - 129 383
     - 0.46
     - 6
     - 23

The scientific claim is therefore: MAID is faster once a tight lower-level
solve costs more than roughly a few thousand gradient steps, and slower below
that. The outer-step ratio improves with condition number as well. Ill-conditioned
physics (motion blur, tomography), weak regularisation, large images, and
nonsmooth priors with many proximal steps are the intended regime. The gallery
example recomputes the flagship, the crossover table and the inpainting
counterpoint from scratch.

Accelerated MAID
----------------

Two optional switches on :class:`~deepinv.optim.bilevel.MAIDConfig`
(both off by default, so Algorithm 3.1 is unchanged):

* ``nonmonotone=True``: Zhang-Hager window reference ``C_k`` in place of
  ``U_lower`` at the current point, with a monotone Lemma 3.5 fallback so a
  sandwich-depressed ``C`` cannot block a valid step. Proof sketch in the
  :mod:`deepinv.optim.bilevel.maid` module docstring (author verification
  required; not claimed as proven).
* ``bb_init=True``: Barzilai-Borwein initial step from
  ``s = theta_k - theta_{k-1}``, ``y = z_k - z_{k-1}``, clamped, with
  fallback ``rho_bar * alpha_k``. Changes only where backtracking starts.

:func:`~deepinv.optim.bilevel.accelerated_maid_config` enables both.

The acceptance gap ``U_upper - U_lower`` is pure inexactness penalty. When
``eps`` is loose that gap is wide, backtracking fails, and MAID pays on
cheap problems. Nonmonotone acceptance and BB steps target that mechanism.

**Three-way crossover** (same protocol as above: relative gap ``5e-3``,
``n=80``, ``d=4``, fixed residual ``1e-4``):

.. list-table::
   :header-rows: 1

   * - cond
     - GD fixed
     - GD MAID
     - GD acc.
     - ratio MAID
     - ratio acc.
     - BT MAID
     - BT acc.
   * - 2
     - 56
     - 81
     - 67
     - 1.45
     - 1.20
     - 3
     - 2
   * - 5
     - 853
     - 861
     - 497
     - 1.01
     - 0.58
     - 4
     - 2
   * - 10
     - 5 847
     - 4 606
     - 1 198
     - 0.79
     - 0.20
     - 5
     - 1
   * - 20
     - 40 691
     - 17 334
     - 8 025
     - 0.43
     - 0.20
     - 5
     - 3
   * - 30
     - 129 383
     - 60 062
     - 25 037
     - 0.46
     - 0.19
     - 6
     - 4

Vanilla MAID crosses below ratio 1 near condition number 5 to 10.
Accelerated MAID is already cheaper at condition number 5 (ratio 0.58) and
cuts the ratio to about 0.20 above that. At condition number 2 it still loses
to fixed accuracy (1.20 vs 1.45 for vanilla), so the cheap end is improved
but not inverted. Backtracking failures drop at every condition number.

**Inpainting counterpoint** (same 32x32 Tikhonov setup, ``max_iter=25``):

.. list-table::
   :header-rows: 1

   * - method
     - f final
     - BaseOptim iters
     - BT failures
   * - MAID
     - 47.2152
     - 246
     - 16
   * - accelerated MAID
     - 47.2152
     - 24
     - 1
   * - fixed accuracy ``1e-4``
     - 47.2307
     - 26
     - n/a

Accelerated MAID cuts backtracking failures from 16 to 1 and BaseOptim
iterations from 246 to 24, matching fixed accuracy on this cheap problem.
The mechanism the acceleration targets is the one that was hurting.

Nesterov or heavy-ball momentum on ``theta`` is not included: it would move
the search direction away from ``-z`` and break Proposition 3.1. Future work.

Minibatch accumulation
----------------------

When ``f`` is a mean over ``m`` samples, :class:`~deepinv.optim.bilevel.MinibatchOracle`
walks the dataset in fixed-order chunks and accumulates the inexact
hypergradient without weakening the a posteriori bound.

**Reduction order (determinism).** Samples are processed in index order
``0, ..., m-1``. Accumulators use sequential floating-point addition, then a
single division by ``m``. Chunk size controls only concurrent working memory.
With a fixed seed the accumulated hypergradient is bitwise identical across
runs and bitwise invariant to chunk size.

**Error bound for the mean.** If ``||z_i - grad f_i|| <= omega_i``, then

.. math::

    \Bigl\|
      \tfrac1m\sum_i z_i - \tfrac1m\sum_i \nabla f_i
    \Bigr\|
    \le \tfrac1m\sum_i \omega_i.

So ``omega_mean = (1/m) sum_i omega_i`` is the certificate Algorithm 3.2 must
use. Using a single-sample ``omega`` or a max would either under-bound or
over-bound the mean; the mean of the bounds is the correct aggregation.

**Memory adaptivity does not change the mathematics.** Chunk sizes 1 and
``m`` produce bitwise identical MAID trajectories (same ``f``, ``theta``,
step sizes) under the fixed reduction.

Peak working memory for the accumulation is bounded by the chunk, not by
``m``. Measured on float64 sample states of length ``d = 200000``
(1.53 MB each), by counting concurrent tensor bytes during the fixed-order
pass (CPU; ``torch.cuda`` is not used on this machine):

.. list-table::
   :header-rows: 1

   * - m
     - chunk
     - peak working (MB)
     - warm-start storage (MB)
   * - 16
     - 4
     - 6.10
     - 24.4
   * - 32
     - 4
     - 6.10
     - 48.8
   * - 64
     - 4
     - 6.10
     - 97.7
   * - 128
     - 4
     - 6.10
     - 195.3

Peak working is flat in ``m``. Warm-start storage is ``O(m)`` by design
(one reconstruction per sample). Against chunk size at fixed ``m = 64``:

.. list-table::
   :header-rows: 1

   * - chunk
     - peak working (MB)
   * - 1
     - 1.53
   * - 2
     - 3.05
   * - 4
     - 6.10
   * - 8
     - 12.2
   * - 16
     - 24.4

**Goal-oriented estimator.** Each sample has its own Hessian. The three DWR
adjoint solves and any Krylov recycling are per sample; recycling does not
span a chunk. Cost is ``m`` times the per-sample DWR cost, independent of
chunk size.

**Cost honesty on the multi-sample path.** On a mean of ``m = 4`` section-4.1
problems at condition number 20, chunk size 2, 10 outer steps for MAID,
warm-started fixed accuracy ``1e-4`` to each method's final ``f`` (fresh
problem objects per method so ``n_gd_iters`` is not shared):

.. list-table::
   :header-rows: 1

   * - method
     - f final
     - GD steps
     - sample lower
     - BT
     - wall (s)
     - GD / fixed
   * - vanilla MAID
     - 43.79
     - 355 611
     - 1 168
     - 11
     - 4.33
     - 2.88
   * - fixed to vanilla ``f``
     - 43.83
     - 123 646
     - 72
     - n/a
     - 1.57
     - 1
   * - accelerated MAID
     - 43.70
     - 144 294
     - 688
     - 7
     - 1.76
     - 0.90
   * - fixed to accel ``f``
     - 43.74
     - 160 380
     - 100
     - n/a
     - 1.98
     - 1

Vanilla MAID loses on both GD and wall. The structural cause is line search:
each trial re-solves all ``m`` samples (1168 sample-lower calls against 72).
Hypergradient and certified ``omega`` remain under 1 percent of wall; the
goal-oriented estimator is not on this path.

Accelerated MAID is the measurement that matters on this path. Every avoided
backtracking failure saves ``m`` lower-level solves rather than one.
Here sample_lower drops from 1168 to 688, GD from 355611 to 144294, and the
GD ratio against fixed flips from 2.88 to 0.90. Wall follows (0.89x fixed).

Cost table over condition number (``max_iter=8``, fixed to each method's
``f``; ``rg`` is GD MAID / GD fixed):

.. list-table::
   :header-rows: 1

   * - cond
     - GD van
     - GD acc
     - GD fixed (to van)
     - sl van
     - sl acc
     - BT van
     - BT acc
     - rg van
     - rg acc
   * - 5
     - 17 112
     - 5 518
     - 5 097
     - 976
     - 588
     - 10
     - 6
     - 3.36
     - 0.60
   * - 10
     - 73 693
     - 22 514
     - 23 200
     - 1 108
     - 664
     - 11
     - 7
     - 3.18
     - 0.97
   * - 20
     - 284 232
     - 72 970
     - 101 860
     - 1 060
     - 576
     - 10
     - 6
     - 2.79
     - 0.68

An earlier draft reused the same problem objects for both methods, so the
GD counter summed MAID and fixed and falsely made fixed look more expensive
(355611 + 123646 = 479257). The tables above use independent counters.

Chunking is a memory control with bitwise trajectory invariance. With
vanilla MAID it is not a free speedup on the multi-sample line-search path.
With acceleration it is competitive on cost as well.

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
