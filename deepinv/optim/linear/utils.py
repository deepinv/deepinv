from __future__ import annotations
from typing import Callable
import torch
from deepinv.utils.tensorlist import TensorList, zeros_like, ones_like


def dot(a: torch.Tensor | TensorList, b: torch.Tensor, dim: int) -> torch.Tensor:
    r"""
    Computes the batched dot product between two tensors or two TensorLists along specified dimensions.

    :param torch.Tensor, deepinv.utils.TensorList a: First input tensor or TensorList.
    :param torch.Tensor, deepinv.utils.TensorList b: Second input tensor or TensorList.
    :param int dim: Dimensions along which to compute the dot product.
    :return: (:class:`torch.Tensor`) The batched dot product of a and b along the specified dimensions.
    """
    if isinstance(a, TensorList):
        aux = 0
        for ai, bi in zip(a.x, b.x, strict=True):
            aux += (ai.conj() * bi).sum(
                dim=dim, keepdim=True
            )  # performs batched dot product
        dot = aux
    else:
        dot = (a.conj() * b).sum(dim=dim, keepdim=True)  # performs batched dot product
    return dot


def _batched_dot(
    a: torch.Tensor | TensorList,
    b: torch.Tensor | TensorList,
    parallel_dim: None | int | list[int] = 0,
) -> torch.Tensor:

    def one(ai, bi):
        prod = ai.conj() * bi
        dims = _reduce_dims(ai, parallel_dim)  # each block uses its own dims
        return prod.sum(dim=dims) if dims else prod

    if isinstance(a, TensorList):
        total = 0
        for ai, bi in zip(a, b, strict=True):
            total = total + one(ai, bi)  # out-of-place, common batch shape
        return total
    return one(a, b)


def _resolve_stagtol(stagtol: float | None, b: torch.Tensor | TensorList) -> float:
    """
    Default the stagnation tolerance to ``8 * eps`` of ``b``'s floating-point
    precision when the caller passes ``None``. Shared by every solver so this
    heuristic constant is defined in a single place.
    """
    if stagtol is None:
        return 8.0 * torch.finfo(b.dtype).eps
        # the 8 approximates stagtol to 1e-6 which has performed well in testing
    return stagtol


def _safe_denom(d: torch.Tensor) -> torch.Tensor:
    """
    Replace exact zeros in a denominator by ones, element-wise.

    Used by the batched LSQR/LSMR recurrences so that an already-converged or
    trivial batch entry (whose numerators are also zero) yields a ``0 / 1 = 0``
    update instead of ``0 / 0 = NaN``, without collapsing the rest of the batch.
    """
    return torch.where(d != 0, d, torch.ones_like(d))


def _as_dim_list(parallel_dim: None | int | list[int]) -> list[int]:
    """Normalize the ``parallel_dim`` argument to a list of batch dimensions."""
    if isinstance(parallel_dim, int):
        return [parallel_dim]
    if parallel_dim is None:
        return []
    return list(parallel_dim)


def _reduce_dims(t: torch.Tensor | TensorList, parallel_dim: list[int]) -> list[int]:
    """
    List of dimensions to reduce over: every dimension of ``t`` that is not a
    batch (``parallel_dim``) dimension. For a :class:`TensorList` the first
    component determines the number of dimensions.
    """
    ref = t[0] if isinstance(t, TensorList) else t
    return [i for i in range(ref.ndim) if i not in parallel_dim]


def _all_zero(x: torch.Tensor | TensorList) -> torch.Tensor | bool:
    """Whether every entry of ``x`` (tensor or :class:`TensorList`) is zero."""
    if isinstance(x, TensorList):
        return all(torch.all(xi == 0) for xi in x)
    return torch.all(x == 0)


def _batched_norm(
    u: torch.Tensor | TensorList, parallel_dim: list[int]
) -> torch.Tensor:
    """
    Euclidean norm reduced over every non-batch dimension (batch dims kept).

    For a :class:`TensorList` it returns the norm of the stacked vector,
    ``sqrt(sum_k ||u_k||^2)``.
    """
    if isinstance(u, TensorList):
        block_norms = torch.stack(
            [
                torch.linalg.vector_norm(
                    uk, dim=_reduce_dims(uk, parallel_dim), keepdim=False
                )
                for uk in u
            ],
            dim=0,
        )  # shape (K, B): one norm per block per sample
        return torch.linalg.vector_norm(block_norms, dim=0)  # (B,)
    return torch.linalg.vector_norm(u, dim=_reduce_dims(u, parallel_dim), keepdim=False)


def _sample_shape(t: torch.Tensor | TensorList, parallel_dim: list[int]) -> list:
    """
    Shape of ``t`` with every non-batch dimension collapsed to 1, so a per-sample
    scalar can be broadcast back onto ``t`` via ``.view(...)``. Returns a nested
    list (one shape per component) for a :class:`TensorList`.
    """
    if isinstance(t, TensorList):
        return [
            [s if i in parallel_dim else 1 for i, s in enumerate(tk.shape)] for tk in t
        ]
    return [s if i in parallel_dim else 1 for i, s in enumerate(t.shape)]


def _make_scalar(b_shape: list, Atb_shape: list):
    """
    Build the ``scalar(v, alpha, b_domain)`` helper used by LSQR/LSMR to multiply
    a vector ``v`` by a per-sample scalar ``alpha``, broadcasting ``alpha`` onto
    the measurement-domain (``b_domain=True``) or signal-domain shape.
    """

    def scalar(v, alpha, b_domain):
        if b_domain:
            if isinstance(v, TensorList):
                return TensorList(
                    [
                        vi * alpha.view(bi_shape)
                        for vi, bi_shape in zip(v, b_shape, strict=True)
                    ]
                )
            return v * alpha.view(b_shape)
        if isinstance(v, TensorList):
            return TensorList(
                [
                    vi * alpha.view(ai_shape)
                    for vi, ai_shape in zip(v, Atb_shape, strict=True)
                ]
            )
        return v * alpha.view(Atb_shape)

    return scalar


def _init_lsq_solution(
    x0: None | float | torch.Tensor | TensorList,
    b: torch.Tensor | TensorList,
    AT: Callable,
) -> tuple:
    """
    Set up the initial iterate shared by LSQR and LSMR from the ``x0`` argument.

    :return: ``(x, xt, x_ref)`` where ``x`` is the initial solution, ``xt`` is
        the pre-computed ``A^T b`` (``None`` when ``x0`` is an explicit tensor,
        as it is then not needed), and ``x_ref`` is a signal-domain tensor whose
        shape is used to broadcast per-sample scalars.
    """
    if x0 is None:
        xt = AT(b)
        return zeros_like(xt), xt, xt
    if isinstance(x0, float):
        xt = AT(b)
        return x0 * ones_like(xt), xt, xt
    x = x0.clone()
    return x, None, x


def _validate_eta(
    eta: None | float | torch.Tensor,
    b: torch.Tensor | TensorList,
    device: torch.device,
) -> torch.Tensor:
    """
    Validate and normalize the damping parameter ``eta`` for LSQR/LSMR: default
    ``None`` to 0, promote to a tensor, check a batched ``eta`` matches ``b``'s
    batch size, and require ``eta >= 0``.
    """
    if eta is None:
        eta = 0.0
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, device=device)
    if eta.ndim > 0:  # if batched eta
        batch_size = b[0].size(0) if isinstance(b, TensorList) else b.size(0)
        if eta.size(0) != batch_size:
            raise ValueError(
                "If eta is batched, its batch size must match the one of b."
            )
        eta = eta.squeeze()  # ensure eta has ndim as b
    if torch.any(eta < 0):
        raise ValueError(
            "Damping parameter eta must be non-negative. The least squares solver "
            "cannot be applied to problems with negative eta."
        )
    return eta


def _sym_ortho(a: torch.Tensor, b: torch.Tensor):
    a, b = torch.broadcast_tensors(a, b)

    big_b = b.abs() > a.abs()
    p = torch.where(big_b, b, a)  # larger magnitude
    q = torch.where(big_b, a, b)  # smaller magnitude

    safe_p = torch.where(p == 0, 1.0, p)  # p == 0 only when both are 0
    tau = q / safe_p
    root = torch.sqrt(1 + tau * tau)

    w = torch.sign(p) / root
    t = w * tau

    c = torch.where(big_b, t, w)
    s = torch.where(big_b, w, t)
    r = p.abs() * root
    return c, s, r
