"""Bilevel learning of a convex ridge regulariser with MAID.

Lower level

.. math::

    h(x, \\theta)
    = \\tfrac12\\|A x - y\\|^2
      + R_\\theta(x)
      + \\tfrac{\\gamma}{2}\\|x\\|^2

with :math:`R_\\theta` from :mod:`deepinv.optim.bilevel.convex_ridge` and
``lambda_k = exp(vartheta_k)``. Upper level (per sample)

.. math::

    g(x) = \\tfrac12\\|x - x^\\star\\|^2.

Hessian and mixed-Jacobian products use ``torch.autograd.grad`` with
``create_graph=True``. Strong convexity modulus is the known floor
``gamma`` (independent of the learned weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .cg_utils import CGResult, cg_solve
from .convex_ridge import (
    ConvexRidgeConfig,
    ConvexRidgePrior,
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


@dataclass
class CRRSampleProblem:
    """One reconstruction sample with a flat CRR parameter vector."""

    physics: Any
    y: torch.Tensor
    x_star: torch.Tensor
    cfg: ConvexRidgeConfig = field(default_factory=ConvexRidgeConfig)
    max_iter: int = 3_000
    lipschitz_data: float | None = None

    def __post_init__(self) -> None:
        self.dtype = self.x_star.dtype
        self.device = self.x_star.device
        self.n_gd_iters = 0
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.L_g = 1.0
        self.L_H_inv = 0.0
        self.L_J = 0.0
        if self.lipschitz_data is None:
            self.lipschitz_data = self._estimate_data_lipschitz()
        self.prior = ConvexRidgePrior(self.cfg)

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
        """Known strong-convexity modulus (the gamma floor)."""
        return float(self.cfg.gamma)

    def load_theta(self, theta: torch.Tensor) -> None:
        self.prior.load_theta(theta)

    def _data_grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.physics.A_adjoint(self.physics.A(x) - self.y)

    def grad_x_h(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        self.load_theta(theta)
        return self._data_grad(x) + self.prior.grad(x)

    def _h_diffable(
        self, x: torch.Tensor, kernels: torch.Tensor, lambdas: torch.Tensor
    ) -> torch.Tensor:
        r = self.physics.A(x) - self.y
        data = 0.5 * (r * r).sum()
        return data + ridge_energy(x, kernels, lambdas, self.cfg)

    def hess_x_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        kernels, lambdas = unpack_theta(theta.detach(), self.cfg)
        x_ = x.detach().requires_grad_(True)
        h = self._h_diffable(x_, kernels, lambdas)
        (g,) = torch.autograd.grad(h, x_, create_graph=True)
        s = (g * v.detach()).sum()
        (Hv,) = torch.autograd.grad(s, x_, retain_graph=False)
        return Hv.detach()

    def mixed_jac_T_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        th = theta.detach().requires_grad_(True)
        kernels, lambdas = unpack_theta(th, self.cfg)
        x_ = x.detach().requires_grad_(True)
        h = self._h_diffable(x_, kernels, lambdas)
        (g,) = torch.autograd.grad(h, x_, create_graph=True)
        s = (g * v.detach()).sum()
        (jtv,) = torch.autograd.grad(s, th, retain_graph=False)
        return jtv.detach()

    def estimate_J_norm(
        self, x: torch.Tensor, theta: torch.Tensor, n_power: int = 2
    ) -> float:
        """Approximate ``||J||`` by power-style probes of ``J^T e``."""
        nrm = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for _ in range(max(n_power, 1)):
            th = theta.detach().requires_grad_(True)
            kernels, lambdas = unpack_theta(th, self.cfg)
            x_ = x.detach().requires_grad_(True)
            h = self._h_diffable(x_, kernels, lambdas)
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
        x, _ = self.solve_lower(theta, eps=1e-4)
        return self.g(x)

    def solve_lower(
        self,
        theta: torch.Tensor,
        eps: float,
        x_init: torch.Tensor | None = None,
        max_iter: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Residual-stopped GD on ``h(., theta)``."""
        self.load_theta(theta)
        mu = self.mu(theta)
        L = self.lipschitz_data + self.prior.lipschitz_bound()
        step = 1.0 / max(L, 1e-8)
        max_it = int(max_iter if max_iter is not None else self.max_iter)
        if x_init is None:
            x = self.physics.A_adjoint(self.y).detach().clone()
        else:
            x = x_init.detach().clone()
        tol = max(float(eps) * mu, 1e-8)
        grad = self.grad_x_h(x, theta)
        gnorm = float(grad.flatten().norm().item())
        n_it = 0
        for _ in range(max_it):
            if gnorm <= tol:
                break
            x = x - step * grad
            grad = self.grad_x_h(x, theta)
            gnorm = float(grad.flatten().norm().item())
            n_it += 1
        else:
            if gnorm > tol:
                raise RuntimeError(
                    f"CRR GD failed residual {tol} (got {gnorm}) "
                    f"in {max_it} iters"
                )
        self.n_gd_iters += n_it
        self.n_lower_solves += 1
        return x, gnorm

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
        # mu = gamma is a known design constant.
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
    max_iter: int = 3_000,
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
        )
        oracles.append(CRRSampleOracle(prob))
    return MinibatchOracle(oracles, chunk_size=chunk_size)


__all__ = [
    "CRRSampleProblem",
    "CRRSampleOracle",
    "build_crr_minibatch_oracle",
    "pack_init_theta",
    "ConvexRidgeConfig",
]
