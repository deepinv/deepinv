"""Bilevel learning of a convex ridge regulariser with MAID.

Lower level

.. math::

    h(x, \\theta)
    = \\tfrac12\\|A x - y\\|^2
      + R_\\theta(x)
      + \\tfrac{\\gamma}{2}\\|x\\|^2

with :math:`R_\\theta` the multiconv Lip-normalised CRR. Upper level
:math:`g(x) = \\tfrac12\\|x - x^\\star\\|^2`. Strong convexity modulus is the
known floor ``gamma``.

Lower-level solver
------------------
The lower level is driven by DeepInverse ``BaseOptim`` through
:mod:`deepinv.optim.bilevel.base_optim_lower`, the same path as
:class:`~deepinv.optim.bilevel.TikhonovWeightProblem`. Default solver is
``FISTA``. Because the CRR is smooth, FISTA is accelerated gradient descent
on ``h``: the smooth objective is exposed as a single
:class:`~deepinv.optim.DataFidelity` and the prior is a null regulariser
whose prox is the identity, so the proximal residual coincides with the
gradient residual used for stopping.

The certificate is unchanged by the choice of algorithm. Strong convexity
gives

    ``||x - xstar|| <= ||grad h(x)|| / mu``

with ``mu = gamma``. Any solver that reaches the residual tolerance earns
the same distance bound. That interchangeability is intentional in the
oracle design: the residual is a property of the objective, not of GD,
FISTA or (future) truncated Newton.

If FISTA remains slow on a problem, truncated Newton is the next option
because the Hessian-vector product already exists for the hypergradient.
It is not built here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class CRRSampleProblem:
    """One reconstruction sample with a flat CRR parameter vector.

    :param str solver: DeepInverse optimiser name. Default ``"FISTA"``
        (accelerated gradient on the smooth CRR lower level). ``"GD"`` is
        also supported for ablation.
    """

    physics: Any
    y: torch.Tensor
    x_star: torch.Tensor
    cfg: ConvexRidgeConfig = field(default_factory=ConvexRidgeConfig)
    solver: str = "FISTA"
    max_iter: int = 10_000
    lipschitz_data: float | None = None

    def __post_init__(self) -> None:
        self.dtype = self.x_star.dtype
        self.device = self.x_star.device
        self.solver = self.solver.upper()
        # Certificate residual is always the gradient residual of h, for any
        # BaseOptim algorithm that reaches it.
        self.residual_kind = "gradient"
        self.n_gd_iters = 0
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.L_g = 1.0
        self.L_H_inv = 0.0
        self.L_J = 0.0
        if self.lipschitz_data is None:
            self.lipschitz_data = self._estimate_data_lipschitz()
        self.prior = ConvexRidgePrior(self.cfg)
        self._fidelity = CRRSmoothFidelity(self.prior)
        self._null_prior = _NullPrior()
        if self.x_star.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"image has {self.x_star.shape[1]} channels, "
                f"CRR expects in_channels={self.cfg.in_channels}"
            )
        # residual_kind_for_solver is consulted only for documentation;
        # we always stop on gradient residual (certificate residual).
        _ = residual_kind_for_solver(self.solver)

    @property
    def n_params(self) -> int:
        return self.cfg.n_params

    def _estimate_data_lipschitz(self, n_power: int = 12) -> float:
        x = torch.randn_like(self.x_star)
        x = x / x.flatten().norm().clamp_min(1e-30)
        for _ in range(n_power):
            Ax = self.physics.A(x)
            x = self.physics.A_adjoint(Ax)
            nrm = x.flatten().norm().clamp_min(1e-30)
            x = x / nrm
        Ax = self.physics.A(x)
        AtAx = self.physics.A_adjoint(Ax)
        return float(AtAx.flatten().norm().item())

    def mu(self, theta: torch.Tensor | None = None) -> float:
        return float(self.cfg.gamma)

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

    def _stepsize(self) -> float:
        """Stepsize for smooth GD/FISTA on the full lower-level objective."""
        L = self.lipschitz_data + self.prior.lipschitz_bound()
        return 1.0 / max(L, 1e-8)

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
    ) -> tuple[torch.Tensor, float]:
        """Residual-stopped DeepInverse solve of ``h(., theta)``.

        Routes through :func:`~deepinv.optim.bilevel.base_optim_lower.solve_base_optim`.
        Stops on the gradient residual ``||grad h|| <= max(eps * mu, 1e-8)``.
        The same residual defines the strong-convexity distance bound for
        every solver name.
        """
        self.load_theta(theta)
        mu = self.mu(theta)
        stepsize = self._stepsize()
        name = (solver if solver is not None else self.solver).upper()
        max_it = int(max_iter if max_iter is not None else self.max_iter)
        model = build_solver(
            name,
            data_fidelity=self._fidelity,
            prior=self._null_prior,
            lambda_reg=1.0,
            stepsize=stepsize,
            max_iter=max_it,
            has_cost=False,
        )
        residual_tol = max(float(eps) * float(mu), 1e-8)
        result = solve_base_optim(
            model,
            self.y,
            self.physics,
            residual_tol=residual_tol,
            residual_kind="gradient",
            x_init=x_init,
            max_iter=max_it,
        )
        self.n_lower_solves += 1
        self.n_gd_iters += result.n_iters
        if not result.converged:
            raise RuntimeError(
                f"CRR {name} failed residual {residual_tol} "
                f"(got {result.residual}) in {result.n_iters} iters "
                f"(step={stepsize:.4g})"
            )
        return result.x, result.residual

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
            "convex ridge regulariser with known gamma floor"
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
    solver: str = "FISTA",
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
]
