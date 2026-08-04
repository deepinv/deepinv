"""Conjugate gradient helpers for hypergradient adjoint solves."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CGResult:
    """Outcome of a CG solve against a fixed SPD operator.

    :param x: approximate solution.
    :param residual: residual vector ``b - A x`` (same convention as the
        experiment scripts: positive residual means the right-hand side
        still to be explained).
    :param residual_norm: Euclidean norm of ``residual``.
    :param n_iter: number of CG iterations performed.
    :param directions: Krylov search directions collected for recycling
        (optional, only when ``store_directions`` is True).
    """

    x: torch.Tensor
    residual: torch.Tensor
    residual_norm: float
    n_iter: int
    directions: list[torch.Tensor] = field(default_factory=list)


def cg_solve(
    A_mv,
    b: torch.Tensor,
    *,
    tol: float | None = None,
    max_iter: int = 10_000,
    x0: torch.Tensor | None = None,
    store_directions: bool = False,
    eps: float = 1e-30,
) -> CGResult:
    r"""CG for :math:`A x = b` with absolute residual stopping.

    If ``tol`` is None, runs exactly ``max_iter`` iterations (or until the
    residual is numerically zero). Returns the residual vector so goal-oriented
    estimators can form dual-weighted residuals without re-applying ``A``.
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_mv(x)
    p = r.clone()
    res_sq = torch.dot(r, r)
    directions: list[torch.Tensor] = []
    if store_directions:
        directions.append(p.clone())

    if tol is not None and res_sq.item() <= tol * tol:
        return CGResult(
            x=x,
            residual=r,
            residual_norm=float(res_sq.sqrt().item()),
            n_iter=0,
            directions=directions,
        )

    tol_sq = None if tol is None else tol * tol
    n_iter = 0
    for n_iter in range(1, int(max_iter) + 1):
        Ap = A_mv(p)
        pAp = torch.dot(p, Ap)
        alpha = res_sq / (pAp + eps)
        x = x + alpha * p
        r = r - alpha * Ap
        res_sq_new = torch.dot(r, r)
        if tol_sq is not None and res_sq_new.item() <= tol_sq:
            res_sq = res_sq_new
            break
        beta = res_sq_new / (res_sq + eps)
        p = r + beta * p
        res_sq = res_sq_new
        if store_directions:
            directions.append(p.clone())
    else:
        n_iter = int(max_iter)

    return CGResult(
        x=x,
        residual=r,
        residual_norm=float(res_sq.sqrt().item()),
        n_iter=n_iter,
        directions=directions,
    )


def cg_recycle(
    A_mv,
    b: torch.Tensor,
    directions: list[torch.Tensor],
    *,
    max_iter: int = 5,
    eps: float = 1e-30,
) -> CGResult:
    r"""CG warm-started by Galerkin projection onto recycled directions.

    Projects ``b`` onto the span of previous search directions (when
    available), then continues standard CG for at most ``max_iter`` steps.
    All solves share the same operator ``A``, so the recycled basis from the
    main adjoint solve reduces the residual of subsequent dual-weighted
    residual solves at almost no cost.
    """
    x = torch.zeros_like(b)
    if directions:
        # Cheap sequential A-orthogonal enrichment using stored p's.
        r = b.clone()
        for p in directions:
            Ap = A_mv(p)
            denom = torch.dot(p, Ap) + eps
            alpha = torch.dot(p, r) / denom
            x = x + alpha * p
            r = r - alpha * Ap
        return cg_solve(
            A_mv, b, tol=None, max_iter=max_iter, x0=x, store_directions=False
        )
    return cg_solve(A_mv, b, tol=None, max_iter=max_iter, store_directions=False)
