"""Algorithm 3.1: Method of Adaptive Inexact Descent (MAID).

Salehi, Mukherjee, Roberts and Ehrhardt, SIAM J. Math. Data Sci. 2025
(arXiv 2308.10098). Reused wholesale for saddle-point lower levels in
Bogensperger, Ehrhardt, Pock, Salehi and Wong (arXiv 2412.06436).

MAID talks only to a :class:`~deepinv.optim.bilevel.oracle.HypergradientOracle`.
It does not know which lower-level solver produced ``z``.

Hyperparameters
--------------
* ``rho`` in (0, 1), ``rho_bar`` > 1: reduce / increase the step size alpha.
* ``nu`` in (0, 1), ``nu_bar`` > 1: reduce / increase accuracies eps, delta.
* ``eta`` in (0, 1): descent margin for Algorithm 3.2 (used only when
  ``check_descent_direction`` is True).
* ``lambd`` with 0 < lambd < eta when the descent test is on, else
  0 < lambd < 1: Armijo fraction in the inexact line search.
* ``max_BT``: initial backtracking budget (grows when the line search fails).
* ``check_descent_direction``: see "Descent test" below. Default False.

Line search
-----------
With g being L_g-smooth,

    U_upper(x, eps) = g(x) + ||grad g(x)|| * eps + (L_g / 2) * eps^2
    U_lower(x, eps) = g(x) - ||grad g(x)|| * eps - (L_g / 2) * eps^2

    psi(alpha) = U_upper(xbar(theta+), eps+) - U_lower(xbar(theta), eps)
                 + lambd * alpha * ||z||^2

``psi(alpha) <= 0`` implies the exact sufficient decrease
``f(theta+) - f(theta) <= -lambd * alpha * ||z||^2`` (Lemma 3.5).
That implication does not use the descent-direction test.

If ``g_convex`` is True, the tighter convex form of remark 3.6 is used:
``psi_tilde = psi - (L_g / 2) * eps^2``.

The gap ``U_upper - U_lower`` is pure inexactness penalty
(``2(||grad g|| eps + (L_g/2) eps^2)``). When ``eps`` is loose that gap is
wide, ``psi <= 0`` is hard to satisfy, backtracking fails, and ``eps``
tightens. That overhead, not the landscape, is what costs MAID below the
crossover. The accelerated options below target that mechanism.

Accelerated MAID (optional)
---------------------------
Two complementary changes, both off by default so Algorithm 3.1 is unchanged
unless requested.

**1. Nonmonotone acceptance** (``nonmonotone=True``), Zhang and Hager form.

    Q_0 = 1,   C_0 = U_lower(xbar(theta_0), eps_0)
    Q_{k+1} = eta_ref * Q_k + 1
    C_{k+1} = (eta_ref * Q_k * C_k
               + U_lower(xbar(theta_{k+1}), eps_{k+1})) / Q_{k+1}

with ``eta_ref`` in ``[0, 1)``. Accept when either

    U_upper(xbar(theta+), eps+) - C_k + lambd * alpha * ||z||^2 <= 0

or the monotone Lemma 3.5 test against the current ``U_lower`` holds.
The monotone fallback is required in practice: ``C_k`` is an average of
past ``U_lower`` values and can sit below the true objective by a leftover
sandwich penalty ``||grad g|| eps + (L_g/2) eps^2``. Pure ZH then rejects
every ``alpha`` even when the monotone test would pass (observed with
``eta_ref = 0``). When ``C_k >= U_lower(current)`` the ZH test is the
looser one and is what buys the nonmonotone relaxation. The window is
updated with ``U_lower`` at the accepted trial accuracy ``eps_k`` (not the
inflated ``eps_{k+1}``), so an extra ``nu_bar`` factor does not push ``C``
down further.

Proof sketch (not claimed as proven). Pointwise the MAID sandwich gives a
controlled relationship between ``U_lower``, ``U_upper`` and the true ``f``
(Lemma 3.5). ``C_k`` is a convex combination of past ``U_lower`` values
(Zhang-Hager weights are non-negative and sum to one after normalisation by
``Q_k``), so it is a lower certificate for the corresponding combination of
past objectives in the same sense. Accepting the ZH test therefore yields a
decrease of the true objective relative to that window average. Accepting
the monotone fallback yields the ordinary one-step decrease of Lemma 3.5.
Either way an accepted step carries a certificate; the nonmonotone path can
only make acceptance easier when the window reference sits above the current
``U_lower``. This is a sketch, not a theorem: the sandwich bookkeeping
against Lemma 3.5 should be checked before treating nonmonotone MAID as
certified.

**2. Barzilai-Borwein initial step** (``bb_init=True``).

After an accepted step, Algorithm 3.1 sets ``alpha <- rho_bar * alpha_k``.
With ``bb_init``, the starting guess for the next outer iteration is
instead a BB estimate from ``s = theta_k - theta_{k-1}`` and
``y = z_k - z_{k-1}``:

* long: ``alpha = <s,s> / <s,y>``
* short: ``alpha = <s,y> / <y,y>``

when ``<s,y> > 0``, clamped to ``[alpha_min, alpha_max]``, else fall back
to ``rho_bar * alpha_k``. This changes only where backtracking starts.
Every guarantee attached to an *accepted* step is untouched. BB steps are
nonmonotone by nature, which is why the two options are intended together.

**What is not done.** Nesterov or heavy-ball momentum on ``theta`` would
move the search direction away from ``-z``, so Proposition 3.1's descent
argument and the error-bound machinery would no longer apply. That is
future work, not part of this extension.

Descent test
------------
The paper's Algorithm 3.2 accepts ``z`` only when
``omega <= (1 - eta) * ||z||``, which guarantees that ``-z`` is a descent
direction and is the hypothesis of Lemma 3.8 (existence of a valid step
size) and therefore of Theorem 3.19 (convergence).

The default is to **skip** that test
(``check_descent_direction=False``):

* **What still holds.** Lemma 3.5 is unconditional. Every step that the
  line search accepts still provably decreases the true upper-level
  objective. The line search remains a genuine certificate of decrease.
* **What is lost.** Without the test there is no a priori guarantee that
  any ``alpha`` satisfies ``psi(alpha) <= 0``, so Lemma 3.8's existence
  result and Theorem 3.19 no longer apply as proven.
* **A posteriori substitute.** If ``-z`` is not a descent direction, no
  step size passes the line search, backtracking fails, and Algorithm 3.1
  lines 9 and 10 already tighten ``eps`` and ``delta``. The check moves
  from a priori to a posteriori. The cost is wasted backtracking
  iterations when ``z`` is poor; the saving is that ``omega`` need never
  be formed, which removes every hard-to-estimate constant from the
  default path.

Set ``check_descent_direction=True`` to restore the certified Algorithm 3.2
path. The history field ``backtrack_failures`` counts how often the
a posteriori mechanism fires; frequent failures mean the trade is not
paying.

Certification of the oracle
---------------------------
By default MAID refuses a non-certified oracle. Convergence of the outer
loop in the sense of Theorem 3.19 is proven only when both
``oracle.certified`` is True and ``check_descent_direction`` is True.
Opt in to non-certified oracles with ``allow_uncertified=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from tqdm import tqdm

from .oracle import HypergradientOracle, LowerLevelState
from .quadratic_ls import QuadraticBilevelLS
from .smooth import SmoothHypergradientOracle, inexact_gradient_from_oracle


@dataclass
class MAIDConfig:
    """Configuration for :class:`MAID`.

    :param bool check_descent_direction: If True, run Algorithm 3.2 with the
        ``omega <= (1 - eta)||z||`` test (certified path). If False
        (default), skip the test and never form ``omega``. See the module
        docstring for what is gained and lost.
    :param bool nonmonotone: If True, use the Zhang-Hager window reference
        ``C_k`` in place of ``U_lower`` at the current point. See module
        docstring for the proof sketch (not claimed as proven).
    :param float eta_ref: Zhang-Hager memory parameter in ``[0, 1)``.
        Ignored when ``nonmonotone`` is False. ``eta_ref = 0`` recovers
        the monotone test against the immediate predecessor after one step.
    :param bool bb_init: If True, initialise the next outer step size with
        a Barzilai-Borwein estimate (fallback ``rho_bar * alpha_k``).
    :param str bb_form: ``"long"`` for ``<s,s>/<s,y>``, ``"short"`` for
        ``<s,y>/<y,y>``.
    :param float alpha_min: lower clamp for BB steps.
    :param float alpha_max: upper clamp for BB steps.
    """

    # None means "derive from the problem at theta0" (the recommended
    # setting). No absolute value serves every prior: the stationarity
    # residual and the hypergradient norm are properties of the regulariser,
    # not of the algorithm. Measured at the same initialisation, eps0 = 1e-1
    # leaves a denoising problem below its noisy input while sitting in a
    # plateau for an inpainting problem, and ||z0|| ranges from 3.7 for a
    # convex ridge prior to 6.9e4 for an input-convex network. Explicit floats
    # are still honoured. See auto_initial_accuracy and auto_initial_step.
    eps0: float | None = None
    delta0: float | None = None
    alpha0: float | None = None
    #: Relative reduction of the initial residual requested when eps0 is None.
    eps0_factor: float = 0.02
    #: Relative parameter change requested of the first step when alpha0 is None.
    alpha0_rel: float = 0.01
    #: Used only when the oracle cannot supply the quantities to derive from.
    eps0_fallback: float = 1e-3
    alpha0_fallback: float = 1e-1
    rho: float = 0.5
    rho_bar: float = 1.2
    nu: float = 0.5
    nu_bar: float = 1.1
    eta: float = 0.5
    lambd: float = 0.1
    max_BT: int = 20
    max_iter: int = 100
    tol: float = 1e-6
    g_convex: bool = False
    max_outer_BT: int = 50
    eps_min: float = 1e-14
    delta_min: float = 1e-14
    check_descent_direction: bool = False
    # Accelerated options (both off => pure Algorithm 3.1).
    nonmonotone: bool = False
    eta_ref: float = 0.85
    bb_init: bool = False
    bb_form: str = "long"
    alpha_min: float = 1e-12
    alpha_max: float = 1e12
    # Reporting (DeepInverse convention, see
    # :class:`deepinv.optim.fixed_point.FixedPoint`): ``verbose`` with
    # ``print``, and a tqdm bar gated on ``show_progress_bar``. The library
    # does not use the ``logging`` module; this class follows the same pattern.
    verbose: bool = False
    show_progress_bar: bool = False
    log_every: int = 1


def accelerated_maid_config(**kwargs) -> MAIDConfig:
    """MAIDConfig with Zhang-Hager nonmonotone LS and BB step init enabled.

    Keyword arguments override any field of :class:`MAIDConfig`. Defaults
    keep Algorithm 3.1 hyper-parameters and only flip the two accelerated
    switches.
    """
    kwargs.setdefault("nonmonotone", True)
    kwargs.setdefault("bb_init", True)
    return MAIDConfig(**kwargs)


@dataclass
class MAID:
    """Method of Adaptive Inexact Descent (Algorithm 3.1)."""

    oracle: HypergradientOracle
    config: MAIDConfig = field(default_factory=MAIDConfig)
    allow_uncertified: bool = False

    def __init__(
        self,
        oracle,
        config: MAIDConfig | None = None,
        allow_uncertified: bool = False,
    ):
        # Accept either a HypergradientOracle or the smooth quadratic problem
        # used in rung 1 tests.
        if isinstance(oracle, QuadraticBilevelLS):
            oracle = SmoothHypergradientOracle(oracle)
        if not isinstance(oracle, HypergradientOracle):
            raise TypeError(
                "MAID expects a HypergradientOracle "
                f"(or QuadraticBilevelLS), got {type(oracle)!r}"
            )
        self.oracle = oracle
        self.config = config if config is not None else MAIDConfig()
        self.allow_uncertified = allow_uncertified
        self.oracle.require_certified_or_opt_in(allow_uncertified)
        self._validate_config()

    def _validate_config(self) -> None:
        c = self.config
        if not (0.0 < c.rho < 1.0):
            raise ValueError(f"rho must lie in (0, 1), got {c.rho}")
        if not (c.rho_bar > 1.0):
            raise ValueError(f"rho_bar must be > 1, got {c.rho_bar}")
        if not (0.0 < c.nu < 1.0):
            raise ValueError(f"nu must lie in (0, 1), got {c.nu}")
        if not (c.nu_bar > 1.0):
            raise ValueError(f"nu_bar must be > 1, got {c.nu_bar}")
        if c.check_descent_direction:
            if not (0.0 < c.lambd < c.eta < 1.0):
                raise ValueError(
                    f"with check_descent_direction=True require "
                    f"0 < lambd < eta < 1, got lambd={c.lambd}, eta={c.eta}"
                )
        else:
            if not (0.0 < c.lambd < 1.0):
                raise ValueError(f"lambd must lie in (0, 1), got {c.lambd}")
        if c.max_BT < 1:
            raise ValueError(f"max_BT must be >= 1, got {c.max_BT}")
        if not (0.0 <= c.eta_ref < 1.0):
            raise ValueError(f"eta_ref must lie in [0, 1), got {c.eta_ref}")
        if c.bb_form not in ("long", "short"):
            raise ValueError(f"bb_form must be 'long' or 'short', got {c.bb_form!r}")
        if not (0.0 < c.alpha_min <= c.alpha_max):
            raise ValueError(
                f"require 0 < alpha_min <= alpha_max, got "
                f"alpha_min={c.alpha_min}, alpha_max={c.alpha_max}"
            )

    def _initial_accuracy_and_step(
        self, theta0: torch.Tensor
    ) -> tuple[float, float, float]:
        """Resolve ``eps0``, ``delta0`` and ``alpha0``.

        Explicit values are honoured. Where a value is ``None`` it is derived
        from the problem at ``theta0``: the accuracy from the stationarity
        residual at :math:`A^\ast y`, the step from the hypergradient norm.
        Both references are properties of the regulariser rather than of the
        algorithm, which is why a shared constant cannot serve every prior.

        Falls back to fixed values, with a warning, when the oracle does not
        expose what the derivation needs.
        """
        import warnings

        c = self.config
        eps = None if c.eps0 is None else float(c.eps0)
        delta = None if c.delta0 is None else float(c.delta0)
        alpha = None if c.alpha0 is None else float(c.alpha0)

        need_eps = eps is None or delta is None
        need_alpha = alpha is None
        if not (need_eps or need_alpha):
            return eps, delta, alpha

        from .batched import auto_initial_accuracy, auto_initial_step

        derived_eps = None
        if need_eps:
            try:
                derived_eps = auto_initial_accuracy(
                    self.oracle, theta0, factor=c.eps0_factor
                )
            except (TypeError, AttributeError):
                derived_eps = float(c.eps0_fallback)
                warnings.warn(
                    f"{type(self.oracle).__name__} cannot supply an initial "
                    "residual, so eps0 falls back to "
                    f"{c.eps0_fallback:.1e}. Pass eps0 explicitly, or use an "
                    "oracle exposing initial_residual_rms.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if eps is None:
            eps = derived_eps
        if delta is None:
            delta = derived_eps if derived_eps is not None else eps

        if need_alpha:
            alpha = float(c.alpha0_fallback)
            solve = getattr(self.oracle, "solve_lower_level", None)
            hyper = getattr(self.oracle, "hypergradient", None)
            if solve is not None and hyper is not None:
                try:
                    lower = solve(theta0, eps=eps)
                    z0 = hyper(theta0, lower, delta=delta).z
                    alpha = auto_initial_step(self.oracle, theta0, z0, rel=c.alpha0_rel)
                except Exception:
                    warnings.warn(
                        "Could not evaluate the hypergradient at theta0, so "
                        f"alpha0 falls back to {c.alpha0_fallback:.1e}. Pass "
                        "alpha0 explicitly if this is not suitable.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if hasattr(self.oracle, "reset_counters"):
                    self.oracle.reset_counters()

        return float(eps), float(delta), float(alpha)

    def run(
        self, theta0: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, list[float] | float | int]]:
        """Run MAID from ``theta0``.

        Returns the final parameter and a history dict. Keys include
        ``f_exact``, ``z_norm``, ``omega``, ``eps``, ``delta``, ``alpha``,
        ``backtrack_failures`` (per upper-level iteration: number of
        accuracy refinements before a step was accepted),
        and scalar totals ``n_backtrack_failures``, ``n_lower_solves``,
        ``n_hypergradients``, ``n_upper_iters``. When accelerated options
        are on, also ``C_ref`` (Zhang-Hager reference) and ``bb_used``
        (1 if the BB guess was admissible that outer step, else 0).
        """
        c = self.config
        oracle = self.oracle
        theta = theta0.detach().clone()
        eps, delta, alpha = self._initial_accuracy_and_step(theta)

        if hasattr(oracle, "reset_counters"):
            oracle.reset_counters()

        history: dict[str, list[float] | float | int] = {
            "f_exact": [],
            "z_norm": [],
            "omega": [],
            "eps": [],
            "delta": [],
            "alpha": [],
            "backtrack_failures": [],
            "C_ref": [],
            "bb_used": [],
        }
        n_backtrack_failures = 0
        # Progress reporting, DeepInverse style: tqdm gated on both flags, so
        # a quiet run stays silent and a verbose one without a bar still gets
        # the printed lines below.
        pbar = tqdm(
            total=int(c.max_iter),
            disable=(not c.verbose or not c.show_progress_bar),
            desc="MAID",
            unit="outer",
        )

        warm: LowerLevelState | None = None
        # Zhang-Hager state. ``window_ready`` is True only after the first
        # accepted step; until then the reference tracks the current
        # U_lower so accuracy refinements at the same theta do not freeze
        # a stale loose certificate into C.
        Q = 1.0
        C: float | None = None
        window_ready = False
        # BB history: previous accepted (theta, z).
        theta_prev: torch.Tensor | None = None
        z_prev: torch.Tensor | None = None
        last_alpha_accepted = float(alpha)

        for _k in range(c.max_iter):
            accepted = False
            z = torch.zeros_like(theta)
            omega = float("nan")
            lower = warm
            alpha_k = alpha
            eps_k = eps
            delta_k = delta
            failures_this_iter = 0
            bb_used_flag = 0.0
            C_for_hist = float("nan")

            for j in range(c.max_BT, c.max_BT + c.max_outer_BT):
                z, eps_k, delta_k, omega, lower = inexact_gradient_from_oracle(
                    oracle,
                    theta,
                    eps=eps_k,
                    delta=delta_k,
                    eta=c.eta,
                    nu=c.nu,
                    warm_start=warm,
                    check_descent_direction=c.check_descent_direction,
                )
                oracle.update_lipschitz_estimates(lower, theta)

                # BB initial step: available once z at the new theta and a
                # stored previous (theta, z) pair exist.
                if (
                    c.bb_init
                    and theta_prev is not None
                    and z_prev is not None
                    and failures_this_iter == 0
                ):
                    bb_alpha, ok = self._bb_step(
                        theta - theta_prev, z - z_prev, last_alpha_accepted
                    )
                    if ok:
                        alpha_k = bb_alpha
                        bb_used_flag = 1.0
                    else:
                        alpha_k = c.rho_bar * last_alpha_accepted
                        bb_used_flag = 0.0

                z_norm_sq = float(torch.sum(z * z).item())
                z_norm = z_norm_sq**0.5
                stop_omega = ((not math.isnan(omega)) and omega <= c.tol) or math.isnan(
                    omega
                )
                if z_norm <= c.tol and stop_omega:
                    history["f_exact"].append(self._f_diag(theta))
                    history["z_norm"].append(z_norm)
                    history["omega"].append(omega)
                    history["eps"].append(eps_k)
                    history["delta"].append(delta_k)
                    history["alpha"].append(alpha_k)
                    history["backtrack_failures"].append(float(failures_this_iter))
                    history["C_ref"].append(float(C) if C is not None else float("nan"))
                    history["bb_used"].append(bb_used_flag)
                    pbar.close()
                    self._finalise_history(
                        history, oracle, n_backtrack_failures + failures_this_iter
                    )
                    return theta, history

                eps_next = c.nu_bar * eps_k
                g_old = float(oracle.g(lower.x).item())
                grad_g_old_norm = float(oracle.grad_g(lower.x).norm().item())
                U_lower = self._U_lower(g_old, grad_g_old_norm, eps_k)

                # Reference for the acceptance test.
                if c.nonmonotone and window_ready:
                    ref = float(C)
                else:
                    # Monotone, or first step before the window exists:
                    # always use the current U_lower (refreshed when eps
                    # tightens at the same theta).
                    ref = U_lower
                C_for_hist = ref

                alpha_try = alpha_k
                line_ok = False
                lower_trial = lower
                g_new = g_old
                grad_g_new_norm = grad_g_old_norm
                for _i in range(j):
                    theta_trial = theta - alpha_try * z
                    lower_trial = oracle.solve_lower_level(
                        theta_trial, eps=eps_k, warm_start=lower
                    )
                    g_new = float(oracle.g(lower_trial.x).item())
                    grad_g_new_norm = float(oracle.grad_g(lower_trial.x).norm().item())
                    U_upper = self._U_upper(g_new, grad_g_new_norm, eps_next)
                    # Nonmonotone test against C (or U_lower on the first step).
                    psi = U_upper - ref + c.lambd * alpha_try * z_norm_sq
                    if c.g_convex:
                        psi = psi - 0.5 * oracle.L_g * (eps_k**2)
                    # Monotone fallback: U_lower at the current point. Needed
                    # because C is an average of past U_lower values and can
                    # sit below the true objective by a leftover sandwich
                    # penalty; pure ZH then rejects every alpha even when the
                    # monotone Lemma 3.5 test would pass. Accept if either
                    # test succeeds. When C >= U_lower the ZH test is the
                    # looser one and is what buys the nonmonotone relaxation.
                    psi_mon = U_upper - U_lower + c.lambd * alpha_try * z_norm_sq
                    if c.g_convex:
                        psi_mon = psi_mon - 0.5 * oracle.L_g * (eps_k**2)
                    if psi <= 0.0 or psi_mon <= 0.0:
                        line_ok = True
                        alpha_k = alpha_try
                        break
                    alpha_try = c.rho * alpha_try

                if line_ok:
                    # Store previous pair for BB before updating theta.
                    theta_prev = theta.detach().clone()
                    z_prev = z.detach().clone()
                    last_alpha_accepted = float(alpha_k)

                    theta = theta - alpha_k * z
                    warm = lower_trial
                    eps = max(c.nu_bar * eps_k, c.eps_min)
                    delta = max(c.nu_bar * delta_k, c.delta_min)

                    # Next starting alpha: provisional rho_bar growth;
                    # overwritten by BB at the start of the next outer step
                    # when bb_init is on and the BB estimate is admissible.
                    alpha = c.rho_bar * alpha_k

                    # Zhang-Hager update of the window reference.
                    # Use the accepted trial accuracy eps_k for U_lower_new,
                    # not the inflated eps_{k+1}: a larger eps would push C
                    # below the true objective by an extra sandwich penalty
                    # and make the next nonmonotone test unsatisfiable
                    # (observed with eta_ref=0, where C = U_lower_new exactly).
                    if c.nonmonotone:
                        U_lower_new = self._U_lower(g_new, grad_g_new_norm, eps_k)
                        if not window_ready:
                            # C_0 = U_lower at the point the step left from.
                            C = ref
                            Q = 1.0
                            window_ready = True
                        Q_new = c.eta_ref * Q + 1.0
                        C = (c.eta_ref * Q * float(C) + U_lower_new) / Q_new
                        Q = Q_new

                    accepted = True
                    break

                # Backtracking failed at this budget: tighten accuracy
                # (Algorithm 3.1 lines 9 and 10). This is the a posteriori
                # detector when the descent test is off.
                failures_this_iter += 1
                n_backtrack_failures += 1
                eps_k = max(c.nu * eps_k, c.eps_min)
                delta_k = max(c.nu * delta_k, c.delta_min)
                alpha_k = alpha

            if not accepted:
                # Exhausting the accuracy refinements at the accuracy floor is
                # convergence, not failure. Near a stationary point the
                # required decrease falls as ||z||^2 while the inexactness gap
                # is bounded below by the attainable accuracy, so no step can
                # satisfy the test however far the step size is reduced. This
                # is reached quickly by priors with few parameters: a
                # three-parameter learned total variation drives ||z|| from
                # 2.6e-01 to 1.6e-02 within twenty outer iterations. Report the
                # iterate reached and why, rather than discarding it.
                at_floor = eps_k <= c.eps_min * 1.01 and delta_k <= c.delta_min * 1.01
                if at_floor:
                    if c.verbose:
                        print(
                            f"[Stopping] Accuracy floor reached at outer "
                            f"iteration {_k} with ||z|| = "
                            f"{float(z.norm().item()):.3e}: no step satisfies "
                            f"the descent test at eps_min = {c.eps_min:.1e}. "
                            "This is convergence to the attainable accuracy; "
                            "lower eps_min to continue."
                        )
                    history["f_exact"].append(self._f_diag(theta))
                    history["z_norm"].append(float(z.norm().item()))
                    history["omega"].append(omega)
                    history["eps"].append(eps_k)
                    history["delta"].append(delta_k)
                    history["alpha"].append(alpha_k)
                    history["backtrack_failures"].append(float(failures_this_iter))
                    history["C_ref"].append(C_for_hist)
                    history["bb_used"].append(bb_used_flag)
                    history["stopped_at_accuracy_floor"] = True
                    pbar.close()
                    self._finalise_history(
                        history,
                        oracle,
                        n_backtrack_failures + failures_this_iter,
                    )
                    return theta, history
                raise RuntimeError(
                    "MAID line search failed: no step accepted within "
                    f"max_outer_BT={c.max_outer_BT} accuracy refinements. "
                    f"Total backtrack failures so far: {n_backtrack_failures}."
                )

            history["f_exact"].append(self._f_diag(theta))
            history["z_norm"].append(float(z.norm().item()))
            history["omega"].append(omega)
            history["eps"].append(eps_k)
            history["delta"].append(delta_k)
            history["alpha"].append(alpha_k)
            history["backtrack_failures"].append(float(failures_this_iter))
            history["C_ref"].append(C_for_hist)
            history["bb_used"].append(bb_used_flag)

            pbar.update(1)
            pbar.set_postfix(
                f=f"{history['f_exact'][-1]:.4e}",
                z=f"{history['z_norm'][-1]:.2e}",
                alpha=f"{alpha_k:.2e}",
                eps=f"{eps_k:.1e}",
                omega=("off" if math.isnan(omega) else f"{omega:.1e}"),
            )
            if (
                c.verbose
                and not c.show_progress_bar
                and (_k % max(int(c.log_every), 1) == 0)
            ):
                # omega is NaN by contract when check_descent_direction is
                # False (Algorithm 3.1 never forms it). Say so, rather than
                # printing a bare nan that reads as a numerical failure.
                omega_str = "off" if math.isnan(omega) else f"{omega:.3e}"
                print(
                    f"MAID it={_k:4d}  f={history['f_exact'][-1]:.6e}  "
                    f"||z||={history['z_norm'][-1]:.3e}  omega={omega_str}  "
                    f"alpha={alpha_k:.3e}  eps={eps_k:.2e}  "
                    f"delta={delta_k:.2e}  BT_fail={failures_this_iter}",
                    flush=True,
                )

            if history["z_norm"][-1] <= c.tol:
                if c.verbose:
                    print(
                        f"[Stopping] ||z|| = {history['z_norm'][-1]:.3e} "
                        f"<= tol = {c.tol:.3e} at outer iteration {_k}."
                    )
                break
        else:
            if c.verbose:
                print(
                    f"[Stopping] Reached max_iter = {c.max_iter} with "
                    f"||z|| = {history['z_norm'][-1]:.3e} > tol = {c.tol:.3e}. "
                    "The objective was still descending; this is a budget "
                    "limit, not convergence."
                )
        pbar.close()

        self._finalise_history(history, oracle, n_backtrack_failures)
        return theta, history

    def _bb_step(
        self,
        s: torch.Tensor,
        y: torch.Tensor,
        fallback_alpha: float,
    ) -> tuple[float, bool]:
        """Barzilai-Borwein step estimate, or fallback when not positive.

        Returns ``(alpha, used_bb)``. ``used_bb`` is False when ``<s,y>``
        is not positive or the estimate is non-finite, in which case
        ``rho_bar * fallback_alpha`` style fallback is left to the caller
        (this method returns ``fallback_alpha`` unchanged as the value).
        """
        c = self.config
        sy = float(torch.sum(s * y).item())
        if not (sy > 0.0) or not math.isfinite(sy):
            return float(fallback_alpha), False
        if c.bb_form == "long":
            ss = float(torch.sum(s * s).item())
            if ss <= 0.0 or not math.isfinite(ss):
                return float(fallback_alpha), False
            alpha_bb = ss / sy
        else:
            yy = float(torch.sum(y * y).item())
            if yy <= 0.0 or not math.isfinite(yy):
                return float(fallback_alpha), False
            alpha_bb = sy / yy
        if not math.isfinite(alpha_bb) or alpha_bb <= 0.0:
            return float(fallback_alpha), False
        alpha_bb = min(max(alpha_bb, c.alpha_min), c.alpha_max)
        return float(alpha_bb), True

    def _finalise_history(
        self,
        history: dict,
        oracle: HypergradientOracle,
        n_backtrack_failures: int,
    ) -> None:
        history["n_backtrack_failures"] = int(n_backtrack_failures)
        history["n_upper_iters"] = len(history["f_exact"])
        history["n_lower_solves"] = int(getattr(oracle, "n_lower_solves", -1))
        history["n_hypergradients"] = int(getattr(oracle, "n_hypergradients", -1))

    def _f_diag(self, theta: torch.Tensor) -> float:
        try:
            return float(self.oracle.f_closed_form(theta).item())
        except NotImplementedError:
            lower = self.oracle.solve_lower_level(
                theta, eps=max(self.config.eps_min, 1e-8)
            )
            return float(self.oracle.g(lower.x).item())
        except RuntimeError:
            # High-accuracy closed form can fail numerically; fall back.
            lower = self.oracle.solve_lower_level(theta, eps=1e-6)
            return float(self.oracle.g(lower.x).item())

    def _U_upper(self, g_val: float, grad_g_norm: float, eps: float) -> float:
        L = self.oracle.L_g
        return g_val + grad_g_norm * eps + 0.5 * L * (eps**2)

    def _U_lower(self, g_val: float, grad_g_norm: float, eps: float) -> float:
        L = self.oracle.L_g
        return g_val - grad_g_norm * eps - 0.5 * L * (eps**2)
