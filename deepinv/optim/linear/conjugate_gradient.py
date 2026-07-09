from __future__ import annotations
from typing import Callable
import torch
from .utils import dot, _as_dim_list, _resolve_stagtol, _reduce_dims, _safe_b_norm_sq
from deepinv.utils.tensorlist import zeros_like


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    stagtol: float | None = None,
    eps: float | None = None,
    parallel_dim: None | int | list[int] = 0,
    init: torch.Tensor | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    r"""
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param Callable A: Linear operator as a callable function, has to be square!
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param int max_iter: maximum number of CG iterations
    :param float tol: relative tolerance for stopping the CG algorithm.
    :param float stagtol: absolute tolerance for stopping the CG algorithm if iterates stagnate, default via dtype precision.
    :param float eps: a small value added to the (squared) denominators for numerical stability.
        If ``None`` (default), it is set precision-dependently to ``finfo(b.dtype).eps ** 2``,
        which guards against division by zero at convergence/breakdown without capping the
        attainable accuracy (a fixed constant here would floor the residual at ``~sqrt(eps)``).
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param torch.Tensor init: Optional initial guess.
    :param bool verbose: Output progress information in the console.
    :return: :class:`torch.Tensor` :math:`x` of shape (B, ...) verifying :math:`Ax=b`.

    """

    stagtol = _resolve_stagtol(stagtol, b)

    # Precision-dependent stabilization constant. It is added to squared denominators
    # (curvature ``<p, Ap>`` and ``||r||^2``), so using ``eps**2`` keeps the residual
    # floor at machine precision instead of a dtype-blind ``~sqrt(eps)``.
    if eps is None:
        eps = torch.finfo(b.dtype).eps ** 2

    parallel_dim = _as_dim_list(parallel_dim)

    dim = _reduce_dims(b, parallel_dim)

    if init is None:
        x = zeros_like(b)
    else:
        x = init.clone()

    r = b - A(x)
    p = r
    res_old = dot(r, r, dim=dim).real
    b_norm_sq = _safe_b_norm_sq(b, dim)
    stagtol = stagtol**2
    tol = b_norm_sq * (tol**2)

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = res_old / (dot(p, Ap, dim=dim) + eps)
        search_update = p * alpha
        x = x + search_update
        r = r - Ap * alpha
        res_new = dot(r, r, dim=dim).real

        search_update_norm = dot(search_update, search_update, dim=dim).real
        xnorm = dot(x, x, dim=dim).real

        if torch.all(res_new <= tol):
            if verbose:
                print("CG converged at iteration", i + 1)
            break
        elif torch.all(search_update_norm <= stagtol * xnorm):
            if verbose:
                print("CG stagnated at iteration", i + 1)
            break
        p = r + p * (res_new / (res_old + eps))
        res_old = res_new

        # safeguard to avoid accumulating numerical errors in the computation of r
        # only applied every 100 iterations to limit computational overhead
        if i > 0 and i % 100 == 0:
            r = b - A(x)
            res_old = dot(r, r, dim=dim).real
    else:
        if verbose:
            print("CG did not converge")

    return x
