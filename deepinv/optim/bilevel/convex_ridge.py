"""Convex ridge regulariser (Goujon-Unser style) for bilevel learning.

.. math::

    R_\theta(x)
    = \sum_k \lambda_k \sum_j
      \rho\bigl((W_k * x)_j\bigr)

    \rho(t) = \mu \log\cosh(t / \mu)

    \lambda_k = \exp(\vartheta_k)

with free parameters ``theta = (free kernel entries, vartheta_k)``. The lower
level also carries a ridge floor ``(gamma / 2) ||x||^2`` so that the
strong-convexity modulus satisfies ``mu >= gamma`` regardless of the
weights.

Why ``lambda_k = exp(vartheta_k)`` rather than softplus
------------------------------------------------------
1. Positivity is exact for every finite parameter value. A negative weight
   would make ``h`` nonconvex and void every MAID guarantee.
2. Gradient descent on the log-weight is multiplicative on ``lambda``,
   which is the right geometry for a quantity that ranges over orders of
   magnitude (the same geometry as a log-spaced grid search).
3. Unlike softplus, ``exp`` does not saturate for large negative arguments,
   so a weight can keep shrinking toward zero at a steady multiplicative
   rate.

This matches :class:`~deepinv.optim.bilevel.TikhonovWeightProblem`, which
already uses ``lambda = exp(theta)`` for the scalar Tikhonov weight.

Scale degeneracy and unit-norm kernels
--------------------------------------
In the small-signal regime ``rho(t) ~ t^2 / (2 mu)``, so

    lambda_k * sum_j rho((W_k * x)_j)
        ~ lambda_k * ||W_k * x||^2 / (2 mu).

Scaling ``W_k`` by ``c`` and ``lambda_k`` by ``1/c^2`` leaves the energy
unchanged. Without a constraint, the pair ``(lambda_k, W_k)`` is degenerate
along a one-parameter family: the hypergradient with respect to
``log lambda_k`` points along a nearly flat direction and the optimiser
moves only the kernels.

The fix used here is to normalise each kernel to unit Frobenius norm
inside the forward pass (after zero-mean centring). Scale then lives
entirely in ``lambda_k``, the flat direction disappears, and the weight
becomes a meaningful learned quantity. The free kernel parameters remain
unconstrained; the gradient of the composition with the normalisation is
the projected gradient on the sphere. A near-zero raw kernel is guarded
by a small floor on the norm.

Somebody extending this with a different profile function needs the same
constraint whenever the profile is (locally) quadratic: weight and kernel
scale are only separable once the kernels are normalised.

Euclidean chart on free kernel coordinates
------------------------------------------
After unit-norm projection, ``unit_norm(c * free_w) = unit_norm(free_w)``
for any ``c > 0``, so the forward map depends only on direction. The
Euclidean hypergradient on free coordinates still scales as
``1 / ||free_w_k||``: tiny free coordinates inflate the kernel block of
``z`` by orders of magnitude, line search shrinks the common step size,
and ``log lambda_k`` barely moves even though its partial is nonzero.

Free kernel coordinates are therefore initialised with per-kernel
Frobenius norm ``free_kernel_scale`` (default 1). That balances
``||z_kernels||`` with ``||z_loglambda||`` under a single outer step size
without changing the forward energy. Re-charting after large steps
(``renormalise_free_kernels``) keeps the metric from drifting.

Default size: 8 kernels of 5x5 on one channel plus 8 log-weights is 208
parameters. Convex in ``x`` by construction and ``C^2``, so the smooth MAID
oracle and the gradient residual apply unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def n_crr_params(
    n_kernels: int = 8,
    kernel_size: int = 5,
    n_channels: int = 1,
) -> int:
    """Number of free parameters in the flat theta vector."""
    return n_kernels * n_channels * kernel_size * kernel_size + n_kernels


def log_cosh(t: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log cosh(t)``."""
    a = t.abs()
    return a + torch.log1p(torch.exp(-2.0 * a)) - math.log(2.0)


@dataclass
class ConvexRidgeConfig:
    """Architecture and fixed constants for the ridge regulariser.

    :param float gamma: strong-convexity floor in the lower level. Residual
        stopping uses ``mu = gamma``, which is known by design and does not
        depend on the learned weights (``exp`` can drive any ``lambda_k``
        toward zero). Default ``1e-2``.
    :param float mu_rho: scale of ``rho(t) = mu_rho log cosh(t / mu_rho)``.
    :param float kernel_norm_eps: floor on the Frobenius norm when
        normalising kernels (avoids division by zero for a collapsed filter).
    :param float free_kernel_scale: target Frobenius norm of each free kernel
        block at initialisation (and when re-charting). Sets the Euclidean
        metric weight on the kernel block; see module docstring. Default 1.
    """

    n_kernels: int = 8
    kernel_size: int = 5
    n_channels: int = 1
    mu_rho: float = 0.05
    gamma: float = 1e-2
    kernel_norm_eps: float = 1e-12
    free_kernel_scale: float = 1.0

    @property
    def n_params(self) -> int:
        return n_crr_params(
            self.n_kernels, self.kernel_size, self.n_channels
        )


def unpack_theta(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map flat theta to (unit-norm zero-mean kernels, positive lambdas).

    Each kernel is centred then divided by its Frobenius norm so that scale
    is absorbed by ``lambda_k = exp(vartheta_k)`` (see module docstring on
    the scale degeneracy).
    """
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D, got shape {tuple(theta.shape)}")
    if theta.numel() != cfg.n_params:
        raise ValueError(
            f"theta has {theta.numel()} entries, expected {cfg.n_params}"
        )
    n_w = cfg.n_kernels * cfg.n_channels * cfg.kernel_size * cfg.kernel_size
    free_w = theta[:n_w].reshape(
        cfg.n_kernels, cfg.n_channels, cfg.kernel_size, cfg.kernel_size
    )
    raw_l = theta[n_w:]
    # Zero spatial mean so constant images are in each filter's null space.
    kernels = free_w - free_w.mean(dim=(-2, -1), keepdim=True)
    # Unit Frobenius norm per kernel: scale lives in lambda_k only.
    norms = (
        kernels.flatten(1)
        .norm(dim=1)
        .clamp_min(cfg.kernel_norm_eps)
        .view(cfg.n_kernels, 1, 1, 1)
    )
    kernels = kernels / norms
    lambdas = torch.exp(raw_l)
    return kernels, lambdas


def renormalise_free_kernels(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> torch.Tensor:
    """Re-chart free kernel blocks to Frobenius norm ``free_kernel_scale``.

    Does not change the forward energy (unit-norm projection is scale
    invariant) but keeps ``||z_kernels||`` comparable to ``||z_loglambda||``
    under a shared outer step size. Log-weights are left unchanged.
    """
    n_w = cfg.n_kernels * cfg.n_channels * cfg.kernel_size * cfg.kernel_size
    free_w = theta[:n_w].reshape(
        cfg.n_kernels, cfg.n_channels, cfg.kernel_size, cfg.kernel_size
    )
    norms = (
        free_w.flatten(1)
        .norm(dim=1)
        .clamp_min(cfg.kernel_norm_eps)
        .view(cfg.n_kernels, 1, 1, 1)
    )
    free_w = free_w / norms * cfg.free_kernel_scale
    return torch.cat([free_w.reshape(-1), theta[n_w:].clone()])


def pack_init_theta(
    cfg: ConvexRidgeConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    log_lambda0: float = -1.0,
) -> torch.Tensor:
    """Random free kernel directions at ``free_kernel_scale``; log-weights fixed.

    Each free kernel block is drawn from a standard normal then rescaled to
    Frobenius norm ``cfg.free_kernel_scale`` (default 1). Log-weights start
    at ``log_lambda0`` (default -1, so lambda ~ 0.3679). Forward kernels are
    unit-norm regardless of the free scale (see module docstring).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_w = cfg.n_kernels * cfg.n_channels * cfg.kernel_size * cfg.kernel_size
    free_w = torch.randn(n_w, generator=gen, dtype=dtype)
    free_w = free_w.reshape(
        cfg.n_kernels, cfg.n_channels, cfg.kernel_size, cfg.kernel_size
    )
    norms = (
        free_w.flatten(1)
        .norm(dim=1)
        .clamp_min(cfg.kernel_norm_eps)
        .view(cfg.n_kernels, 1, 1, 1)
    )
    free_w = free_w / norms * cfg.free_kernel_scale
    raw_l = torch.full((cfg.n_kernels,), float(log_lambda0), dtype=dtype)
    return torch.cat([free_w.reshape(-1), raw_l]).to(device=device, dtype=dtype)


def ridge_energy(
    x: torch.Tensor,
    kernels: torch.Tensor,
    lambdas: torch.Tensor,
    cfg: ConvexRidgeConfig,
) -> torch.Tensor:
    """Scalar energy ``R(x) + (gamma/2)||x||^2``."""
    pad = cfg.kernel_size // 2
    feats = F.conv2d(x, kernels, padding=pad)
    t = feats / cfg.mu_rho
    rho = cfg.mu_rho * log_cosh(t)
    weighted = rho * lambdas.view(1, -1, 1, 1)
    r = weighted.sum()
    ridge = 0.5 * cfg.gamma * (x * x).sum()
    return r + ridge


def ridge_grad_x(
    x: torch.Tensor,
    kernels: torch.Tensor,
    lambdas: torch.Tensor,
    cfg: ConvexRidgeConfig,
) -> torch.Tensor:
    """Gradient of the ridge energy in ``x`` (detached)."""
    x_ = x.detach().requires_grad_(True)
    e = ridge_energy(x_, kernels.detach(), lambdas.detach(), cfg)
    (g,) = torch.autograd.grad(e, x_)
    return g.detach()


class ConvexRidgePrior:
    """Thin energy/grad wrapper driven by a flat theta vector.

    Not a full :class:`~deepinv.optim.Prior` subclass: the bilevel path uses
    hand-rolled residual-stopped GD that only needs ``grad`` and the energy,
    which avoids fighting BaseOptim's ``lambda * prior.grad`` scaling.
    """

    def __init__(self, cfg: ConvexRidgeConfig | None = None):
        self.cfg = cfg if cfg is not None else ConvexRidgeConfig()
        self._kernels: torch.Tensor | None = None
        self._lambdas: torch.Tensor | None = None

    def load_theta(self, theta: torch.Tensor) -> None:
        k, lam = unpack_theta(theta, self.cfg)
        self._kernels = k
        self._lambdas = lam

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if self._kernels is None or self._lambdas is None:
            raise RuntimeError("call load_theta before energy/grad")
        return ridge_energy(x, self._kernels, self._lambdas, self.cfg)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        if self._kernels is None or self._lambdas is None:
            raise RuntimeError("call load_theta before energy/grad")
        return ridge_grad_x(x, self._kernels, self._lambdas, self.cfg)

    def lipschitz_bound(self) -> float:
        """Conservative Lip of ``grad R`` for stepsize selection.

        With unit-norm kernels, ``||Hess R|| <= sum_k lambda_k / mu_rho + gamma``.
        """
        if self._kernels is None or self._lambdas is None:
            raise RuntimeError("call load_theta before lipschitz_bound")
        total = float(self.cfg.gamma)
        for k in range(self.cfg.n_kernels):
            # kernels are unit-norm by construction of unpack_theta
            total += float(self._lambdas[k].item()) / self.cfg.mu_rho
        return total
