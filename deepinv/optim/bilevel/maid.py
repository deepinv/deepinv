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

Descent test
------------
The paper's Algorithm 3.2 accepts ``z`` only when
``omega <= (1 - eta) * ||z||``, which guarantees that ``-z`` is a descent
direction and is the hypothesis of Lemma 3.8 (existence of a valid step
size) and therefore of Theorem 3.19 (convergence).

By contributor decision the default is to **skip** that test
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
    """

    eps0: float = 1e-1
    delta0: float = 1e-1
    alpha0: float = 1e-2
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
                raise ValueError(
                    f"lambd must lie in (0, 1), got {c.lambd}"
                )
        if c.max_BT < 1:
            raise ValueError(f"max_BT must be >= 1, got {c.max_BT}")

    def run(
        self, theta0: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, list[float] | float | int]]:
        """Run MAID from ``theta0``.

        Returns the final parameter and a history dict. Keys include
        ``f_exact``, ``z_norm``, ``omega``, ``eps``, ``delta``, ``alpha``,
        ``backtrack_failures`` (per upper-level iteration: 1 if the line
        search exhausted its budget and accuracies were tightened, else 0),
        and scalar totals ``n_backtrack_failures``, ``n_lower_solves``,
        ``n_hypergradients``, ``n_upper_iters``.
        """
        c = self.config
        oracle = self.oracle
        theta = theta0.detach().clone()
        eps = float(c.eps0)
        delta = float(c.delta0)
        alpha = float(c.alpha0)

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
        }
        n_backtrack_failures = 0

        warm: LowerLevelState | None = None

        for _k in range(c.max_iter):
            accepted = False
            z = torch.zeros_like(theta)
            omega = float("nan")
            lower = warm
            alpha_k = alpha
            eps_k = eps
            delta_k = delta
            failures_this_iter = 0

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

                z_norm_sq = float(torch.dot(z, z).item())
                z_norm = z_norm_sq**0.5
                stop_omega = (
                    (not math.isnan(omega)) and omega <= c.tol
                ) or math.isnan(omega)
                if z_norm <= c.tol and stop_omega:
                    history["f_exact"].append(self._f_diag(theta))
                    history["z_norm"].append(z_norm)
                    history["omega"].append(omega)
                    history["eps"].append(eps_k)
                    history["delta"].append(delta_k)
                    history["alpha"].append(alpha_k)
                    history["backtrack_failures"].append(float(failures_this_iter))
                    self._finalise_history(
                        history, oracle, n_backtrack_failures + failures_this_iter
                    )
                    return theta, history

                eps_next = c.nu_bar * eps_k
                g_old = float(oracle.g(lower.x).item())
                grad_g_old_norm = float(oracle.grad_g(lower.x).norm().item())
                U_lower = self._U_lower(g_old, grad_g_old_norm, eps_k)

                alpha_try = alpha_k
                line_ok = False
                lower_trial = lower
                for _i in range(j):
                    theta_trial = theta - alpha_try * z
                    lower_trial = oracle.solve_lower_level(
                        theta_trial, eps=eps_k, warm_start=lower
                    )
                    g_new = float(oracle.g(lower_trial.x).item())
                    grad_g_new_norm = float(
                        oracle.grad_g(lower_trial.x).norm().item()
                    )
                    U_upper = self._U_upper(g_new, grad_g_new_norm, eps_next)
                    psi = U_upper - U_lower + c.lambd * alpha_try * z_norm_sq
                    if c.g_convex:
                        psi = psi - 0.5 * oracle.L_g * (eps_k**2)
                    if psi <= 0.0:
                        line_ok = True
                        alpha_k = alpha_try
                        break
                    alpha_try = c.rho * alpha_try

                if line_ok:
                    theta = theta - alpha_k * z
                    warm = lower_trial
                    eps = max(c.nu_bar * eps_k, c.eps_min)
                    delta = max(c.nu_bar * delta_k, c.delta_min)
                    alpha = c.rho_bar * alpha_k
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

            if history["z_norm"][-1] <= c.tol:
                break

        self._finalise_history(history, oracle, n_backtrack_failures)
        return theta, history

    def _finalise_history(
        self,
        history: dict,
        oracle: HypergradientOracle,
        n_backtrack_failures: int,
    ) -> None:
        history["n_backtrack_failures"] = int(n_backtrack_failures)
        history["n_upper_iters"] = len(history["f_exact"])
        history["n_lower_solves"] = int(getattr(oracle, "n_lower_solves", -1))
        history["n_hypergradients"] = int(
            getattr(oracle, "n_hypergradients", -1)
        )

    def _f_diag(self, theta: torch.Tensor) -> float:
        try:
            return float(self.oracle.f_closed_form(theta).item())
        except NotImplementedError:
            lower = self.oracle.solve_lower_level(theta, eps=self.config.eps_min)
            return float(self.oracle.g(lower.x).item())

    def _U_upper(self, g_val: float, grad_g_norm: float, eps: float) -> float:
        L = self.oracle.L_g
        return g_val + grad_g_norm * eps + 0.5 * L * (eps**2)

    def _U_lower(self, g_val: float, grad_g_norm: float, eps: float) -> float:
        L = self.oracle.L_g
        return g_val - grad_g_norm * eps - 0.5 * L * (eps**2)
