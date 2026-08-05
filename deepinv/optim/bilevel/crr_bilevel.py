"""Bilevel learning of a convex ridge regulariser with MAID.

Lower level

.. math::

    h(x, \\theta)
    = \\tfrac12\\|A x - y\\|^2
      + R_\\theta(x)
      + \\tfrac{\\gamma}{2}\\|x\\|^2

with :math:`R_\\theta` the multiconv Lip-normalised CRR. Upper level
:math:`g(x) = \\tfrac12\\|x - x^\\star\\|^2`.

Strong convexity modulus is

    ``mu = mu_data + gamma``

where ``mu_data`` is a lower bound on the strong-convexity contribution of
the data term (1 for Denoising with ``A = I``, 0 for general ``A``) and
``gamma`` is the explicit ridge floor. On denoising the floor is unnecessary
and may be set to zero; ``mu >= 1`` still holds from the data term alone.

Lower-level solver
------------------
Supported solvers: ``GD``, ``FISTA`` (DeepInverse BaseOptim),
``FISTA_RESTART`` (O'Donoghue-Candes gradient adaptive restart on the
smooth CRR objective), and ``NEWTON`` (truncated Newton with CG using the
Hessian-vector product already required for the hypergradient).

Default is ``FISTA_RESTART``. Because the CRR is smooth, FISTA is
accelerated gradient descent on ``h``: the smooth objective is exposed as
a single :class:`~deepinv.optim.DataFidelity` and the prior is a null
regulariser whose prox is the identity.

Step size
---------
``step = 1 / (L_data + L_prior)``. ``L_data`` is an adaptive power
iteration on ``A^* A`` with a safety factor so the estimate is an upper
bound in practice (plain fixed-count power iteration converges from
below). ``L_prior`` is the analytic chart ``2 * exp(beta) + gamma``;
call :meth:`CRRSampleProblem.measure_prior_lipschitz` to check it against
the top eigenvalue of ``Hess R``.

The certificate is unchanged by the choice of algorithm. Strong convexity
gives

    ``||x - xstar|| <= ||grad h(x)|| / mu``

Any solver that reaches the residual tolerance earns the same distance
bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim.prior import Prior

from .base_optim_lower import (
    build_solver,
    residual_kind_for_solver,
    solve_base_optim,
)
from .cg_utils import CGResult, cg_solve
from .convex_ridge import (
    ConvexRidgeConfig,
    ConvexRidgePrior,
    get_conv_lip,
    pack_init_theta,
    ridge_energy,
    unpack_theta,
)
from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
)
from .smooth import smooth_hypergradient_error_bound


# Power-method defaults: converge in relative Rayleigh change, then inflate
# so the estimate is an upper bound in practice (power iteration approaches
# lambda_max from below).
_LDATA_RTOL = 1e-4
_LDATA_MAX_ITER = 200
_LDATA_SAFETY = 1.05
_LDATA_MIN_ITER = 4


class _NullPrior(Prior):
    """Null regulariser so FISTA/GD act on a single smooth fidelity.

    ``explicit_prior = False`` disables BaseOptim cost evaluation, which
    would otherwise require a full ``fn`` on the composite fidelity.
    """

    def __init__(self):
        super().__init__()
        self.explicit_prior = False

    def grad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.zeros_like(x)

    def prox(
        self, x: torch.Tensor, *args, gamma: float = 1.0, **kwargs
    ) -> torch.Tensor:
        return x


class CRRSmoothFidelity(DataFidelity):
    """Smooth lower-level objective as a DeepInverse data-fidelity.

    Combines the measurement term with the loaded CRR energy so that
    ``BaseOptim`` (GD / FISTA) sees one smooth map whose gradient residual
    is exactly ``||grad_x h||``.
    """

    def __init__(self, crr_prior: ConvexRidgePrior):
        super().__init__()
        self.crr_prior = crr_prior

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs
    ) -> torch.Tensor:
        return physics.A_adjoint(physics.A(x) - y) + self.crr_prior.grad(x)

    def fn(
        self, x: torch.Tensor, y: torch.Tensor, physics, *args, **kwargs
    ) -> torch.Tensor:
        r = physics.A(x) - y
        return 0.5 * (r * r).sum() + self.crr_prior.energy(x)


def power_iteration_lipschitz(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    rtol: float = _LDATA_RTOL,
    max_iter: int = _LDATA_MAX_ITER,
    min_iter: int = _LDATA_MIN_ITER,
    safety: float = _LDATA_SAFETY,
    seed: int | None = 0,
) -> tuple[float, int, float]:
    """Adaptive power iteration for the dominant eigenvalue of a PSD map.

    Iterates until the Rayleigh quotient (here ``||M x||`` with ``||x||=1``)
    changes by less than ``rtol`` relatively, for at least ``min_iter``
    steps, or until ``max_iter``. Multiplies the estimate by ``safety`` so
    the returned value is an upper bound in practice: fixed-count power
    iteration converges to ``lambda_max`` from below.

    Returns ``(safety * L_raw, n_iters, L_raw)``.
    """
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        x = torch.randn(
            x0.shape, generator=gen, dtype=x0.dtype, device="cpu"
        ).to(device=x0.device, dtype=x0.dtype)
    else:
        x = torch.randn_like(x0)
    x = x / x.flatten().norm().clamp_min(1e-30)
    prev = None
    nrm = 0.0
    n_iters = 0
    for n_iters in range(1, int(max_iter) + 1):
        y = matvec(x)
        nrm = float(y.flatten().norm().item())
        x = y / max(nrm, 1e-30)
        if (
            prev is not None
            and n_iters >= int(min_iter)
            and abs(nrm - prev) <= float(rtol) * max(abs(prev), 1e-30)
        ):
            break
        prev = nrm
    L_raw = float(nrm)
    return float(safety) * L_raw, int(n_iters), L_raw


@dataclass
class CRRSampleProblem:
    """One reconstruction sample with a flat CRR parameter vector.

    :param str solver: ``"GD"``, ``"FISTA"``, ``"FISTA_RESTART"`` (default)
        or ``"NEWTON"`` (truncated Newton + CG on the hypergradient HVP).
    """

    physics: Any
    y: torch.Tensor
    x_star: torch.Tensor
    cfg: ConvexRidgeConfig = field(default_factory=ConvexRidgeConfig)
    solver: str = "FISTA_RESTART"
    max_iter: int = 10_000
    lipschitz_data: float | None = None

    def __post_init__(self) -> None:
        self.dtype = self.x_star.dtype
        self.device = self.x_star.device
        self.solver = self.solver.upper()
        # Certificate residual is always the gradient residual of h, for any
        # algorithm that reaches it.
        self.residual_kind = "gradient"
        self.n_gd_iters = 0
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.L_g = 1.0
        self.L_H_inv = 0.0
        self.L_J = 0.0
        self._L_data_raw: float | None = None
        self._L_data_power_iters: int | None = None
        if self.lipschitz_data is None:
            L_safe, n_it, L_raw = self.estimate_data_lipschitz()
            self.lipschitz_data = L_safe
            self._L_data_raw = L_raw
            self._L_data_power_iters = n_it
        # Lower bound on lambda_min(A^* A). For Denoising (A = Id) this is 1.
        self.mu_data = self._estimate_mu_data()
        self.prior = ConvexRidgePrior(self.cfg)
        self._fidelity = CRRSmoothFidelity(self.prior)
        self._null_prior = _NullPrior()
        if self.x_star.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"image has {self.x_star.shape[1]} channels, "
                f"CRR expects in_channels={self.cfg.in_channels}"
            )
        if self.mu() <= 0.0:
            raise ValueError(
                "lower level is not strongly convex: mu_data + gamma = "
                f"{self.mu()}. Use gamma > 0, or a physics with mu_data > 0 "
                "(for example Denoising with A = I)."
            )
        if self.solver in ("GD", "FISTA"):
            # residual_kind_for_solver is consulted for BaseOptim wiring only.
            _ = residual_kind_for_solver(self.solver)
        elif self.solver not in ("FISTA_RESTART", "NEWTON"):
            raise ValueError(
                f"Unknown solver {self.solver!r}. Supported: GD, FISTA, "
                "FISTA_RESTART, NEWTON."
            )

    @property
    def n_params(self) -> int:
        return self.cfg.n_params

    def estimate_data_lipschitz(
        self,
        *,
        rtol: float = _LDATA_RTOL,
        max_iter: int = _LDATA_MAX_ITER,
        min_iter: int = _LDATA_MIN_ITER,
        safety: float = _LDATA_SAFETY,
        fixed_iters: int | None = None,
    ) -> tuple[float, int, float]:
        """Power iteration on ``A^* A``.

        If ``fixed_iters`` is set, run exactly that many iterations with
        ``safety=1`` (raw estimate, for audit comparison). Otherwise use
        adaptive relative tolerance and the configured safety factor.
        """

        def matvec(v: torch.Tensor) -> torch.Tensor:
            return self.physics.A_adjoint(self.physics.A(v))

        if fixed_iters is not None:
            return power_iteration_lipschitz(
                matvec,
                self.x_star,
                rtol=0.0,
                max_iter=int(fixed_iters),
                min_iter=int(fixed_iters),
                safety=1.0,
                seed=0,
            )
        return power_iteration_lipschitz(
            matvec,
            self.x_star,
            rtol=rtol,
            max_iter=max_iter,
            min_iter=min_iter,
            safety=safety,
            seed=0,
        )

    def _estimate_mu_data(self) -> float:
        """Lower bound on the strong-convexity contribution of the data term.

        For Denoising with ``A = Id`` this is 1. For general ``A`` we use 0
        (safe: residual_tol is tighter than necessary when gamma alone sets
        ``mu``).
        """
        x = torch.randn_like(self.x_star)
        Ax = self.physics.A(x)
        if Ax.shape == x.shape and torch.allclose(Ax, x, atol=1e-12, rtol=0.0):
            return 1.0
        return 0.0

    def mu(self, theta: torch.Tensor | None = None) -> float:
        """Strong-convexity modulus: ``mu_data + gamma``."""
        return float(self.mu_data) + float(self.cfg.gamma)

    def load_theta(self, theta: torch.Tensor) -> None:
        self.prior.load_theta(theta)

    def _data_grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.physics.A_adjoint(self.physics.A(x) - self.y)

    def grad_x_h(
        self, x: torch.Tensor, theta: torch.Tensor, *, reload: bool = True
    ) -> torch.Tensor:
        if reload:
            self.load_theta(theta)
        return self._data_grad(x) + self.prior.grad(x)

    def energy_h(
        self, x: torch.Tensor, theta: torch.Tensor, *, reload: bool = True
    ) -> torch.Tensor:
        """Scalar lower-level energy ``h(x, theta)``."""
        if reload:
            self.load_theta(theta)
        r = self.physics.A(x) - self.y
        return 0.5 * (r * r).sum() + self.prior.energy(x)

    def _stepsize(self) -> float:
        """Stepsize for smooth GD/FISTA on the full lower-level objective."""
        L = self.lipschitz_data + self.prior.lipschitz_bound()
        return 1.0 / max(L, 1e-8)

    def analytic_prior_lipschitz(self) -> float:
        """Analytic chart ``2 * exp(beta) + gamma`` (requires loaded theta)."""
        return self.prior.lipschitz_bound()

    def measure_prior_lipschitz(
        self,
        theta: torch.Tensor,
        x: torch.Tensor | None = None,
        *,
        rtol: float = 1e-4,
        max_iter: int = 100,
        safety: float = 1.0,
    ) -> tuple[float, int, float]:
        """Power iteration on ``Hess_x R_theta`` at ``x`` (default zeros).

        Uses the hypergradient HVP minus ``A^* A``. At ``x = 0`` the
        smooth_l1 potentials sit in the quadratic region, which maximises
        curvature of the ridge, so the measured value is a practical probe
        of the analytic chart.

        Returns ``(safety * L_raw, n_iters, L_raw)``.
        """
        self.load_theta(theta)
        if x is None:
            x = torch.zeros_like(self.x_star)
        else:
            x = x.detach()

        def matvec(v: torch.Tensor) -> torch.Tensor:
            Hv = self.hess_x_matvec(x, theta, v)
            return Hv - self.physics.A_adjoint(self.physics.A(v))

        return power_iteration_lipschitz(
            matvec,
            self.x_star,
            rtol=rtol,
            max_iter=max_iter,
            min_iter=4,
            safety=safety,
            seed=1,
        )

    def _h_diffable(
        self,
        x: torch.Tensor,
        weights: list[torch.Tensor],
        scaling: torch.Tensor,
        beta: torch.Tensor,
        lip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.physics.A(x) - self.y
        data = 0.5 * (r * r).sum()
        return data + ridge_energy(
            x, weights, scaling, beta, self.cfg, lip=lip
        )

    def hess_x_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        weights, scaling, beta = unpack_theta(theta.detach(), self.cfg)
        w_det = [w.detach() for w in weights]
        lip = get_conv_lip(w_det, self.cfg, detach=True)
        x_ = x.detach().requires_grad_(True)
        h = self._h_diffable(
            x_, w_det, scaling.detach(), beta.detach(), lip=lip
        )
        (g,) = torch.autograd.grad(h, x_, create_graph=True)
        s = (g * v.detach()).sum()
        (Hv,) = torch.autograd.grad(s, x_, retain_graph=False)
        return Hv.detach()

    def mixed_jac_T_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        th = theta.detach().requires_grad_(True)
        weights, scaling, beta = unpack_theta(th, self.cfg)
        lip = get_conv_lip([w.detach() for w in weights], self.cfg, detach=True)
        x_ = x.detach().requires_grad_(True)
        h = self._h_diffable(x_, weights, scaling, beta, lip=lip)
        (g,) = torch.autograd.grad(h, x_, create_graph=True)
        s = (g * v.detach()).sum()
        (jtv,) = torch.autograd.grad(s, th, retain_graph=False)
        return jtv.detach()

    def estimate_J_norm(
        self, x: torch.Tensor, theta: torch.Tensor, n_power: int = 2
    ) -> float:
        nrm = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for _ in range(max(n_power, 1)):
            th = theta.detach().requires_grad_(True)
            weights, scaling, beta = unpack_theta(th, self.cfg)
            lip = get_conv_lip(
                [w.detach() for w in weights], self.cfg, detach=True
            )
            x_ = x.detach().requires_grad_(True)
            h = self._h_diffable(x_, weights, scaling, beta, lip=lip)
            (g,) = torch.autograd.grad(h, x_, create_graph=True)
            e = torch.randn_like(x)
            e = e / e.flatten().norm().clamp_min(1e-30)
            s = (g * e).sum()
            (jte,) = torch.autograd.grad(s, th, retain_graph=False)
            nrm = torch.maximum(nrm, jte.norm())
        return float(nrm.item())

    def g(self, x: torch.Tensor) -> torch.Tensor:
        diff = x - self.x_star
        return 0.5 * (diff * diff).sum()

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.x_star

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        x, _ = self.solve_lower(theta, eps=5e-3)
        return self.g(x)

    def solve_lower(
        self,
        theta: torch.Tensor,
        eps: float,
        x_init: torch.Tensor | None = None,
        max_iter: int | None = None,
        solver: str | None = None,
        *,
        record_every: int | None = None,
    ) -> tuple[torch.Tensor, float] | tuple[torch.Tensor, float, dict]:
        """Residual-stopped solve of ``h(., theta)``.

        Stops on the gradient residual ``||grad h|| <= max(eps * mu, 1e-8)``.
        The same residual defines the strong-convexity distance bound for
        every solver name.

        If ``record_every`` is set, also returns a history dict with residual
        (and objective for GD) snapshots.
        """
        self.load_theta(theta)
        mu = self.mu(theta)
        stepsize = self._stepsize()
        name = (solver if solver is not None else self.solver).upper()
        max_it = int(max_iter if max_iter is not None else self.max_iter)
        residual_tol = max(float(eps) * float(mu), 1e-8)
        x0 = (
            x_init.detach().clone()
            if x_init is not None
            else self.physics.A_adjoint(self.y).detach().clone()
        )

        if name in ("GD", "FISTA"):
            x, residual, n_iters, hist = self._solve_baseoptim(
                name,
                theta,
                x0,
                stepsize,
                residual_tol,
                max_it,
                record_every=record_every,
            )
        elif name == "FISTA_RESTART":
            x, residual, n_iters, hist = self._solve_fista_restart(
                theta,
                x0,
                stepsize,
                residual_tol,
                max_it,
                record_every=record_every,
            )
        elif name == "NEWTON":
            x, residual, n_iters, hist = self._solve_truncated_newton(
                theta,
                x0,
                residual_tol,
                max_it,
                record_every=record_every,
            )
        else:
            raise ValueError(
                f"Unknown solver {name!r}. Supported: GD, FISTA, "
                "FISTA_RESTART, NEWTON."
            )

        self.n_lower_solves += 1
        self.n_gd_iters += n_iters
        if residual > residual_tol * 1.01:
            raise RuntimeError(
                f"CRR {name} failed residual {residual_tol} "
                f"(got {residual}) in {n_iters} iters "
                f"(step={stepsize:.4g})"
            )
        if record_every is not None:
            return x, residual, hist
        return x, residual

    def _solve_baseoptim(
        self,
        name: str,
        theta: torch.Tensor,
        x0: torch.Tensor,
        stepsize: float,
        residual_tol: float,
        max_it: int,
        *,
        record_every: int | None,
    ) -> tuple[torch.Tensor, float, int, dict]:
        """GD / vanilla FISTA via DeepInverse BaseOptim."""
        model = build_solver(
            name,
            data_fidelity=self._fidelity,
            prior=self._null_prior,
            lambda_reg=1.0,
            stepsize=stepsize,
            max_iter=max_it,
            has_cost=False,
        )
        # Manual residual loop so we can record history (solve_base_optim
        # does not expose intermediate residuals).
        if x0 is not None and name == "FISTA":
            init = (x0, x0.clone())
        else:
            init = x0
        X = model.init_iterate_fn(self.y, self.physics, init=init)
        history = self._empty_history()
        residual = float(
            self.grad_x_h(model.get_output(X), theta, reload=False)
            .flatten()
            .norm()
            .item()
        )
        obj = None
        if name == "GD":
            obj = float(self.energy_h(model.get_output(X), theta, reload=False).item())
        self._record(history, 0, residual, obj)
        if residual <= residual_tol:
            return model.get_output(X), residual, 0, history

        n_iters = 0
        prev_obj = obj
        monotone = True
        for it in range(max_it):
            cur_params = model.update_params_fn(it)
            cur_prior = model.update_prior_fn(it)
            cur_data_fidelity = model.update_data_fidelity_fn(it)
            X = model.fixed_point.iterator(
                X,
                cur_data_fidelity,
                cur_prior,
                cur_params,
                self.y,
                self.physics,
            )
            n_iters += 1
            x = model.get_output(X)
            residual = float(
                self.grad_x_h(x, theta, reload=False).flatten().norm().item()
            )
            if name == "GD":
                obj = float(self.energy_h(x, theta, reload=False).item())
                if prev_obj is not None and obj > prev_obj + 1e-12:
                    monotone = False
                prev_obj = obj
            if record_every is not None and (
                n_iters % int(record_every) == 0 or residual <= residual_tol
            ):
                self._record(history, n_iters, residual, obj)
            if residual <= residual_tol:
                history["monotone_objective"] = monotone if name == "GD" else None
                history["n_iters"] = n_iters
                return x, residual, n_iters, history

        history["monotone_objective"] = monotone if name == "GD" else None
        history["n_iters"] = n_iters
        return model.get_output(X), residual, n_iters, history

    def _solve_fista_restart(
        self,
        theta: torch.Tensor,
        x0: torch.Tensor,
        stepsize: float,
        residual_tol: float,
        max_it: int,
        *,
        record_every: int | None,
        a: float = 3.0,
    ) -> tuple[torch.Tensor, float, int, dict]:
        """FISTA with O'Donoghue-Candes gradient adaptive restart.

        Restart when ``(y - x_new) . (x_new - x) > 0`` (momentum points
        against the last step). The counter for the Chambolle-Dossal
        sequence ``alpha = (k + a - 1)/(k + a)`` is reset on restart.
        """
        x = x0.detach().clone()
        y = x.clone()
        history = self._empty_history()
        residual = float(
            self.grad_x_h(x, theta, reload=False).flatten().norm().item()
        )
        self._record(history, 0, residual, None)
        if residual <= residual_tol:
            history["n_iters"] = 0
            history["n_restarts"] = 0
            return x, residual, 0, history

        k = 0
        n_restarts = 0
        n_iters = 0
        for _ in range(max_it):
            g = self.grad_x_h(y, theta, reload=False)
            x_new = y - stepsize * g
            # Gradient adaptive restart (O'Donoghue & Candes 2015).
            restart = bool(
                ((y - x_new).flatten() * (x_new - x).flatten()).sum().item() > 0.0
            )
            if restart:
                k = 0
                n_restarts += 1
                y = x_new
            else:
                alpha = (k + a - 1.0) / (k + a)
                y = x_new + alpha * (x_new - x)
                k = k + 1
            x = x_new
            n_iters += 1
            residual = float(
                self.grad_x_h(x, theta, reload=False).flatten().norm().item()
            )
            if record_every is not None and (
                n_iters % int(record_every) == 0 or residual <= residual_tol
            ):
                self._record(history, n_iters, residual, None)
            if residual <= residual_tol:
                history["n_iters"] = n_iters
                history["n_restarts"] = n_restarts
                return x, residual, n_iters, history

        history["n_iters"] = n_iters
        history["n_restarts"] = n_restarts
        return x, residual, n_iters, history

    def _solve_truncated_newton(
        self,
        theta: torch.Tensor,
        x0: torch.Tensor,
        residual_tol: float,
        max_it: int,
        *,
        record_every: int | None,
        cg_iters: int = 20,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
        max_bt: int = 20,
    ) -> tuple[torch.Tensor, float, int, dict]:
        """Truncated Newton: CG on Hess h with Armijo line search."""
        x = x0.detach().clone()
        history = self._empty_history()
        residual = float(
            self.grad_x_h(x, theta, reload=False).flatten().norm().item()
        )
        self._record(
            history,
            0,
            residual,
            float(self.energy_h(x, theta, reload=False).item()),
        )
        if residual <= residual_tol:
            history["n_iters"] = 0
            history["n_cg"] = 0
            return x, residual, 0, history

        n_iters = 0
        n_cg = 0
        for _ in range(max_it):
            g = self.grad_x_h(x, theta, reload=False)
            residual = float(g.flatten().norm().item())
            if residual <= residual_tol:
                break

            def Hmv(v: torch.Tensor) -> torch.Tensor:
                return self.hess_x_matvec(x, theta, v)

            # Truncate CG: absolute residual a fraction of ||g||, or cg_iters.
            cg_tol = max(0.1 * residual, residual_tol * 0.1)
            cg = cg_solve(Hmv, -g, tol=cg_tol, max_iter=cg_iters)
            n_cg += cg.n_iter
            p = cg.x
            # Ensure descent direction; fall back to steepest descent.
            gTp = float((g * p).sum().item())
            if gTp >= 0.0:
                p = -g
                gTp = float((g * p).sum().item())

            f0 = float(self.energy_h(x, theta, reload=False).item())
            alpha = 1.0
            accepted = False
            for _bt in range(max_bt):
                x_trial = x + alpha * p
                f_trial = float(
                    self.energy_h(x_trial, theta, reload=False).item()
                )
                if f_trial <= f0 + armijo_c * alpha * gTp:
                    x = x_trial
                    accepted = True
                    break
                alpha *= armijo_rho
            if not accepted:
                # Last-resort gradient step with the GD stepsize.
                x = x - self._stepsize() * g

            n_iters += 1
            residual = float(
                self.grad_x_h(x, theta, reload=False).flatten().norm().item()
            )
            if record_every is not None and (
                n_iters % int(record_every) == 0 or residual <= residual_tol
            ):
                self._record(
                    history,
                    n_iters,
                    residual,
                    float(self.energy_h(x, theta, reload=False).item()),
                )
            if residual <= residual_tol:
                break

        history["n_iters"] = n_iters
        history["n_cg"] = n_cg
        return x, residual, n_iters, history

    @staticmethod
    def _empty_history() -> dict:
        return {
            "iters": [],
            "residuals": [],
            "objectives": [],
            "monotone_objective": None,
            "n_iters": 0,
            "n_restarts": 0,
            "n_cg": 0,
        }

    @staticmethod
    def _record(
        history: dict, it: int, residual: float, obj: float | None
    ) -> None:
        history["iters"].append(int(it))
        history["residuals"].append(float(residual))
        if obj is not None:
            history["objectives"].append(float(obj))

    def inexact_hypergradient(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        delta: float,
        max_cg_iter: int = 80,
    ) -> tuple[torch.Tensor, CGResult]:
        rhs = self.grad_g(x)

        def Hmv(v: torch.Tensor) -> torch.Tensor:
            return self.hess_x_matvec(x, theta, v)

        cg = cg_solve(Hmv, rhs, tol=delta, max_iter=max_cg_iter)
        z = -self.mixed_jac_T_matvec(x, theta, cg.x)
        self.n_hypergradients += 1
        return z, cg


class CRRSampleOracle(HypergradientOracle):
    """Smooth IFT oracle for one :class:`CRRSampleProblem`."""

    def __init__(self, problem: CRRSampleProblem):
        self.problem = problem
        self.n_lower_solves = 0
        self.n_hypergradients = 0

    @property
    def certified(self) -> bool:
        return True

    @property
    def citation(self) -> str:
        return (
            "Salehi et al., SIAM J. Math. Data Sci. 2025, Theorem 2.1; "
            "convex ridge regulariser with mu = mu_data + gamma"
        )

    @property
    def L_g(self) -> float:
        return self.problem.L_g

    def reset_counters(self) -> None:
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.problem.n_gd_iters = 0
        self.problem.n_lower_solves = 0
        self.problem.n_hypergradients = 0

    def solve_lower_level(
        self,
        theta: torch.Tensor,
        eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        x_init = None if warm_start is None else warm_start.x
        x, residual = self.problem.solve_lower(theta, eps=eps, x_init=x_init)
        self.n_lower_solves += 1
        return LowerLevelState(
            x=x,
            eps=eps,
            extras={
                "residual": residual,
                "mu": self.problem.mu(theta),
                "residual_kind": "gradient",
                "solver": self.problem.solver,
            },
        )

    def hypergradient(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        delta: float,
    ) -> HypergradientState:
        z, cg = self.problem.inexact_hypergradient(
            lower.x, theta, delta=delta
        )
        self.n_hypergradients += 1
        return HypergradientState(
            z=z,
            delta=delta,
            extras={
                "q": cg.x,
                "residual_vec": cg.residual,
                "cg": cg,
                "x": lower.x,
                "theta": theta,
                "mu": self.problem.mu(theta),
                "L_H_inv": 0.0,
                "L_J": 0.0,
            },
        )

    def error_bound(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        hyper: HypergradientState,
        eps: float,
        delta: float,
    ) -> float:
        x = hyper.extras.get("x", lower.x)
        th = hyper.extras.get("theta", theta)
        J_norm = self.problem.estimate_J_norm(x, th)
        grad_g_norm = float(self.grad_g(x).flatten().norm().item())
        return smooth_hypergradient_error_bound(
            eps=eps,
            delta=delta,
            mu=float(hyper.extras["mu"]),
            L_g=self.L_g,
            J_norm=J_norm,
            grad_g_norm=grad_g_norm,
            L_H_inv=0.0,
            L_J=0.0,
        )

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.problem.g(x)

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return self.problem.grad_g(x)

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        return self.problem.f_closed_form(theta)

    def update_lipschitz_estimates(
        self, lower: LowerLevelState, theta: torch.Tensor
    ) -> None:
        return None


def build_crr_minibatch_oracle(
    samples: list[tuple[Any, torch.Tensor, torch.Tensor]],
    cfg: ConvexRidgeConfig | None = None,
    chunk_size: int = 1,
    max_iter: int = 10_000,
    solver: str = "FISTA_RESTART",
):
    """Minibatch oracle over CRR samples sharing one flat theta."""
    from .minibatch import MinibatchOracle

    cfg = cfg if cfg is not None else ConvexRidgeConfig()
    oracles = []
    for physics, y, x_star in samples:
        prob = CRRSampleProblem(
            physics=physics,
            y=y,
            x_star=x_star,
            cfg=cfg,
            max_iter=max_iter,
            solver=solver,
        )
        oracles.append(CRRSampleOracle(prob))
    return MinibatchOracle(oracles, chunk_size=chunk_size)


__all__ = [
    "CRRSampleProblem",
    "CRRSampleOracle",
    "CRRSmoothFidelity",
    "build_crr_minibatch_oracle",
    "pack_init_theta",
    "ConvexRidgeConfig",
    "power_iteration_lipschitz",
]
