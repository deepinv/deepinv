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


def _inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product (works for any tensor shape)."""
    return torch.sum(a * b)


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
    Works for multi-dimensional tensors (image batches) via Frobenius products.
    """
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_mv(x)
    p = r.clone()
    res_sq = _inner(r, r)
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
        pAp = _inner(p, Ap)
        alpha = res_sq / (pAp + eps)
        x = x + alpha * p
        r = r - alpha * Ap
        res_sq_new = _inner(r, r)
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


def _inner_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample Frobenius product, kept broadcastable against ``(B, ...)``."""
    dims = tuple(range(1, a.dim()))
    return (a * b).sum(dim=dims, keepdim=True)


@dataclass
class BatchedCGResult:
    """CG outcome for a block-diagonal system solved sample-by-sample.

    :param x: solutions, shape ``(B, ...)``.
    :param residual: residual vectors ``b - A x``.
    :param residual_norm: per-sample residual norms, shape ``(B,)``.
    :param n_iter: iterations run (the max across samples; converged samples
        are frozen and cost nothing further).
    :param n_converged: how many samples met ``tol``.
    """

    x: torch.Tensor
    residual: torch.Tensor
    residual_norm: torch.Tensor
    n_iter: int
    n_converged: int


def cg_solve_batched(
    A_mv,
    b: torch.Tensor,
    *,
    tol: torch.Tensor | float | None = None,
    max_iter: int = 10_000,
    x0: torch.Tensor | None = None,
    eps: float = 1e-30,
) -> BatchedCGResult:
    r"""CG on ``B`` independent SPD systems sharing one operator call.

    The batched lower-level Hessians are block diagonal: sample ``i`` has its
    own :math:`H_i`, and ``A_mv`` applies all of them in a single call. CG
    itself must stay **per sample**: one shared ``alpha`` would couple systems
    that are mathematically independent and destroy the Krylov property for
    every one of them. So the inner products, ``alpha`` and ``beta`` all carry
    a batch dimension.

    Samples reaching ``tol`` are frozen by zeroing their ``alpha`` rather than
    dropped from the batch: the tensor stays rectangular, which is the point of
    batching, and a frozen sample contributes no further change to ``x`` or
    ``r``. Iteration stops once every sample has converged.

    ``tol`` may be a scalar or a per-sample tensor of shape ``(B,)``.
    """
    B = b.shape[0]
    trail = (1,) * (b.dim() - 1)
    if x0 is None:
        x = torch.zeros_like(b)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A_mv(x)
    p = r.clone()
    rs = _inner_batched(r, r)

    if tol is None:
        tol_sq = None
    else:
        t = (
            tol
            if torch.is_tensor(tol)
            else torch.full((B,), float(tol), dtype=b.dtype, device=b.device)
        )
        tol_sq = (t.to(device=b.device, dtype=b.dtype) ** 2).view(B, *trail)

    active = (
        torch.ones((B, *trail), dtype=torch.bool, device=b.device)
        if tol_sq is None
        else (rs > tol_sq)
    )
    n_iter = 0
    for it in range(1, int(max_iter) + 1):
        if not bool(active.any()):
            break
        n_iter = it
        Ap = A_mv(p)
        pAp = _inner_batched(p, Ap)
        alpha = torch.where(active, rs / (pAp + eps), torch.zeros_like(rs))
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _inner_batched(r, r)
        if tol_sq is not None:
            active = active & (rs_new > tol_sq)
        beta = torch.where(active, rs_new / (rs + eps), torch.zeros_like(rs))
        p = torch.where(active, r + beta * p, p)
        rs = rs_new

    n_conv = B if tol_sq is None else int(B - active.reshape(B).sum().item())
    return BatchedCGResult(
        x=x,
        residual=r,
        residual_norm=rs.reshape(B).sqrt(),
        n_iter=int(n_iter),
        n_converged=n_conv,
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
            denom = _inner(p, Ap) + eps
            alpha = _inner(p, r) / denom
            x = x + alpha * p
            r = r - alpha * Ap
        return cg_solve(
            A_mv, b, tol=None, max_iter=max_iter, x0=x, store_directions=False
        )
    return cg_solve(A_mv, b, tol=None, max_iter=max_iter, store_directions=False)
