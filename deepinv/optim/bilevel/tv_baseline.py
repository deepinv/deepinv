"""Isotropic total variation baseline for bilevel prior comparisons.

Solves

    x_hat(lambda) = argmin_x  1/2 ||A x - y||^2 + lambda * TV(x)

where TV is the isotropic total variation

    TV(x) = sum_{i,j,c} sqrt( (D_h x)_{cij}^2 + (D_v x)_{cij}^2 )

with forward differences and zero Neumann (no wrap) boundaries.

The dual / proximal operator is DeepInverse ``TVPrior`` /
``TVDenoiser`` (Chambolle-Pock prox of isotropic TV). For general
physics the lower level is proximal gradient with step 1 / L_data.

Solver checks (assertions, not prints)
--------------------------------------
1. Adjoint identity: with ``div := -nabla_adjoint``,

       <nabla u, p> = -<u, div p>

   to machine precision on random probes.
2. Primal objective decrease: the accepted reconstruction has strictly
   lower objective than the measurement-side initialisation
   ``x0 = A^* y`` (for denoising ``A = I`` this is the measurement).

Grid tuning of ``lambda`` on a train set minimises the same supervised
upper level used for the scalar Tikhonov baseline,

    f = (1/m) sum_i 1/2 ||x_hat_i - x_star_i||^2.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from deepinv.models.tv import TVDenoiser
from deepinv.optim.prior import TVPrior


def nabla(u: torch.Tensor) -> torch.Tensor:
    """Forward differences (B, C, H, W) -> (B, C, H, W, 2)."""
    return TVDenoiser.nabla(u)


def nabla_adjoint(p: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`nabla` (true ``grad^*``)."""
    return TVDenoiser.nabla_adjoint(p)


def div(p: torch.Tensor) -> torch.Tensor:
    """Negative adjoint of nabla, the discrete divergence used by TV duals.

    With this definition the identity ``<nabla u, p> = -<u, div p>`` holds.
    """
    return -nabla_adjoint(p)


def assert_grad_div_adjoint(
    shape: tuple[int, ...] = (1, 3, 16, 16),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    atol: float = 1e-12,
    seed: int = 0,
) -> float:
    """Assert ``<nabla u, p> + <u, div p> = 0`` to machine precision.

    Returns the absolute residual of the identity (should be ~0).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(shape, generator=gen, dtype=dtype, device=device)
    p = torch.randn(
        (*shape, 2), generator=gen, dtype=dtype, device=device
    )
    Du = nabla(u)
    d = div(p)
    residual = float((Du * p).sum().item() + (u * d).sum().item())
    if abs(residual) > atol:
        raise AssertionError(
            f"grad/div adjoint residual {residual:.3e} exceeds atol={atol}"
        )
    return residual


def isotropic_tv(x: torch.Tensor) -> torch.Tensor:
    """Scalar isotropic TV energy of ``x`` (sum over batch of TV per sample)."""
    dx = nabla(x)
    # sqrt(gx^2 + gy^2) per pixel, then sum.
    nrm = torch.linalg.vector_norm(dx, dim=-1)
    return nrm.reshape(x.shape[0], -1).sum(dim=-1).sum()


def primal_objective(
    x: torch.Tensor,
    physics: Any,
    y: torch.Tensor,
    lam: float,
) -> float:
    """1/2 ||A x - y||^2 + lambda * TV(x)."""
    r = physics.A(x) - y
    data = 0.5 * float((r * r).sum().item())
    tv = float(isotropic_tv(x).item())
    return data + float(lam) * tv


def estimate_data_lipschitz(
    physics: Any,
    x_like: torch.Tensor,
    n_iter: int = 30,
    tol: float = 1e-4,
) -> float:
    """Power iteration for the top eigenvalue of A^* A."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(
        x_like.shape, generator=gen, dtype=x_like.dtype, device=x_like.device
    )
    v = v / v.flatten().norm().clamp_min(1e-30)
    lam = 0.0
    for _ in range(n_iter):
        w = physics.A_adjoint(physics.A(v))
        nrm = float(w.flatten().norm().item())
        if nrm <= 0:
            return 1.0
        v = w / nrm
        lam = nrm
        if nrm < tol:
            break
    # Safety factor for PGD step size.
    return max(lam * 1.05, 1e-8)


def solve_isotropic_tv(
    physics: Any,
    y: torch.Tensor,
    lam: float,
    *,
    n_it: int = 500,
    crit: float = 1e-6,
    n_it_prox: int = 50,
    x_init: torch.Tensor | None = None,
    verify: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve 1/2||A x - y||^2 + lambda TV(x) by proximal gradient.

    For pure denoising (A close to identity) a single TV prox recovers the
    dual Chambolle solution. For general A, PGD with a safe step size
    1/L_data is used. When ``verify`` is True the adjoint identity and
    primal objective decrease are asserted.

    Returns
    -------
    x_hat : reconstruction
    info : dict with obj_init, obj_final, L_data, n_iters
    """
    if lam <= 0:
        raise ValueError(f"lambda must be positive, got {lam}")

    if verify:
        assert_grad_div_adjoint(
            shape=tuple(y.shape),
            dtype=y.dtype,
            device=y.device,
        )

    x0 = physics.A_adjoint(y).detach().clone() if x_init is None else x_init.clone()
    obj0 = primal_objective(x0, physics, y, lam)

    L = estimate_data_lipschitz(physics, x0)
    step = 1.0 / L
    prior = TVPrior(def_crit=crit, n_it_max=n_it_prox)

    # Pure denoising shortcut: one prox of (lambda TV) at y is exact for A = I.
    # Detect A = I via A(x0) ~ x0 and A^* y ~ y (within a tight relative tol).
    is_identity = False
    with torch.no_grad():
        ax0 = physics.A(x0)
        if ax0.shape == x0.shape:
            rel = float((ax0 - x0).flatten().norm().item()) / max(
                float(x0.flatten().norm().item()), 1e-30
            )
            is_identity = rel < 1e-10

    if is_identity:
        # TVPrior.prox(y, gamma=lam) = argmin 1/2||x-y||^2 + lam TV(x).
        # Use a high inner iteration budget for the dual.
        prior_tight = TVPrior(def_crit=crit, n_it_max=max(n_it, n_it_prox))
        xh = prior_tight.prox(y, gamma=float(lam))
        n_done = 1
    else:
        xh = x0.clone()
        n_done = 0
        for it in range(n_it):
            grad_data = physics.A_adjoint(physics.A(xh) - y)
            z = xh - step * grad_data
            x_next = prior.prox(z, gamma=float(step * lam))
            rel = float((x_next - xh).flatten().norm().item()) / max(
                float(x_next.flatten().norm().item()), 1e-30
            )
            xh = x_next
            n_done = it + 1
            if rel < crit:
                break

    obj1 = primal_objective(xh, physics, y, lam)
    if verify:
        if not (obj1 < obj0 - 1e-12):
            raise AssertionError(
                f"TV primal objective did not decrease: "
                f"obj_init={obj0:.6e}, obj_final={obj1:.6e}, "
                f"lam={lam}, n_iters={n_done}"
            )

    return xh, {
        "obj_init": obj0,
        "obj_final": obj1,
        "L_data": L,
        "n_iters": float(n_done),
        "lam": float(lam),
    }


def grid_tune_tv(
    samples: Sequence[tuple[Any, torch.Tensor, torch.Tensor]],
    *,
    n_grid: int = 15,
    lam_min: float = 5e-3,
    lam_max: float = 5e-1,
    n_it: int = 400,
    verify_once: bool = True,
) -> dict[str, float]:
    """Grid-tune isotropic TV lambda on the train set.

    Minimises mean supervised loss f = 1/2 ||x_hat - x_star||^2 over a
    geometric grid of lambda, matching the scalar Tikhonov baseline.
    """
    lams = torch.logspace(
        float(torch.log10(torch.tensor(lam_min))),
        float(torch.log10(torch.tensor(lam_max))),
        n_grid,
        dtype=torch.float64,
    )
    best: dict[str, float] | None = None
    verified = False
    for lam_t in lams:
        lam = float(lam_t.item())
        losses: list[float] = []
        psnrs: list[float] = []
        for physics, y, x_star in samples:
            do_verify = verify_once and not verified
            xh, _info = solve_isotropic_tv(
                physics, y, lam, n_it=n_it, verify=do_verify
            )
            if do_verify:
                verified = True
            mse = float(torch.mean((xh - x_star) ** 2).item())
            losses.append(0.5 * float(torch.sum((xh - x_star) ** 2).item()))
            psnrs.append(
                float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-30))).item())
            )
        mean_f = sum(losses) / len(losses)
        mean_p = sum(psnrs) / len(psnrs)
        if best is None or mean_f < best["f"]:
            best = {
                "lam": lam,
                "f": mean_f,
                "psnr_train": mean_p,
            }
    assert best is not None
    return best


def recon_tv(
    physics: Any,
    y: torch.Tensor,
    x_star: torch.Tensor,
    lam: float,
    *,
    n_it: int = 500,
    verify: bool = False,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Reconstruct with isotropic TV and return (x_hat, PSNR, info)."""
    xh, info = solve_isotropic_tv(physics, y, lam, n_it=n_it, verify=verify)
    mse = float(torch.mean((xh - x_star) ** 2).item())
    p = float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-30))).item())
    return xh, p, info
