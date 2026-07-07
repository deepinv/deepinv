from __future__ import annotations
from typing import Callable
import torch
from .utils import dot


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    max_iter: int = 1e2,
    tol: float = 1e-6,
    stagtol: float = 1e-6,
    eps: float = 1e-8,
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
    :param float stagtol: absolute tolerance for stopping the CG algorithm if iterates stagnate.
    :param float eps: a small value for numerical stability
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param torch.Tensor init: Optional initial guess.
    :param bool verbose: Output progress information in the console.
    :return: :class:`torch.Tensor` :math:`x` of shape (B, ...) verifying :math:`Ax=b`.

    """

    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]
    if parallel_dim is None:
        parallel_dim = []

    dim = [i for i in range(b.ndim) if i not in parallel_dim]

    if init is None:
        x = torch.zeros_like(b)
    else:
        x = init.clone()

    r = b - A(x)
    p = r
    res_old = dot(r, r, dim=dim).real
    b_norm_sq = dot(b, b, dim=dim).real
    # handles case b=0
    b_norm_sq = torch.where(b_norm_sq > 0, b_norm_sq, torch.ones_like(b_norm_sq))
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
