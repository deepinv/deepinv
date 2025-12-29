from __future__ import annotations
from typing import Callable
import torch
from torch import Tensor
from deepinv.utils import TensorList, zeros_like
from .utils import dot
import warnings


def conjugate_gradient(
    A,
    b,
    max_iter: int = 1e2,
    tol: float = 1e-5,
    eps: float = 1e-8,
    parallel_dim: None | int | list[int] = 0,
    init=None,
    verbose: bool = False,
):
    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]
    if parallel_dim is None:
        parallel_dim = []

    dim = [i for i in range(b.ndim) if i not in parallel_dim]

    if init is None:
        x = torch.zeros_like(b)
    else:
        x = init

    r = b - A(x)
    p = r
    rsold = dot(r, r, dim=dim).real
    b_norm_sq = dot(b, b, dim=dim).real
    # handles case b=0
    b_norm_sq = torch.where(b_norm_sq > 0, b_norm_sq, torch.ones_like(b_norm_sq))
    tol = b_norm_sq * (tol**2)

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / (dot(p, Ap, dim=dim) + eps)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = dot(r, r, dim=dim).real
        if torch.all(rsnew < tol):
            if verbose:
                print("CG Converged at iteration", i + 1)
            break
        p = r + p * (rsnew / (rsold + eps))
        rsold = rsnew

        # safeguard to avoid accumulating numerical errors in the computation of r
        # only applied every 50 iterations to limit computational overhead
        if i > 0 and i % 50 == 0:
            r = b - A(x)
            rsold = dot(r, r, dim=dim).real
    else:
        if verbose:
            print("CG did not converge")

    return x


def conjugate_gradient_new(
    A: Callable,
    b: Tensor,
    max_iter: int = 1e2,
    tol: float = 1e-5,
    eps: float = None,
    parallel_dim: None | int | list[int] = 0,
    init: Tensor = None,
    verbose: bool = False,
) -> Tensor:
    """
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param Callable A: Linear operator as a callable function, has to be square!
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param int max_iter: maximum number of CG iterations
    :param float tol: absolute tolerance for stopping the CG algorithm.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions.
        If None, all dimensions are considered as batch dimensions.
    :param torch.Tensor init: Optional initial guess.
    :param bool verbose: Output progress information in the console.
    :return: torch.Tensor :math:`x` of shape (B, ...) verifying :math:`Ax=b`.

    """
    if eps is not None:
        warnings.warn(
            "The 'eps' parameter is deprecated in the CG solver and will be removed in future versions.",
            DeprecationWarning,
        )

    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]
    if parallel_dim is None:
        parallel_dim = []

    if isinstance(b, TensorList):
        dim = [i for i in range(b[0].ndim) if i not in parallel_dim]
    else:
        dim = [i for i in range(b.ndim) if i not in parallel_dim]

    if init is not None:
        x = init
    else:
        x = zeros_like(b)

    r = b - A(x)
    p = r
    res_old = dot(r, r, dim=dim).real
    b_norm_sq = dot(b, b, dim=dim).real
    b_norm_sq = torch.where(b_norm_sq > 0, b_norm_sq, torch.ones_like(b_norm_sq))
    tol_sq = b_norm_sq * tol**2
    active = res_old >= tol_sq

    for iteration in range(int(max_iter)):
        Ap = A(p)
        pAp = dot(p, Ap, dim=dim).real

        # only update alpha for active batches
        alpha = res_old / pAp * active

        x = x + p * alpha
        r = r - Ap * alpha
        res_new = dot(r, r, dim=dim).real

        active = res_new >= tol_sq

        if not torch.any(active):
            if verbose:
                print("CG Converged at iteration", iteration + 1)
            break

        beta = res_new / res_old * active
        p = r + p * beta
        res_old = res_new

        # safeguard to avoid accumulating numerical errors in the computation of r
        # only applied every 50 iterations to limit computational overhead
        if iteration > 0 and iteration % 50 == 0 and torch.any(active):
            r_true = b - A(x)
            r = torch.where(active.expand_as(r), r_true, r)
            res_old = dot(r, r, dim=dim).real
    else:
        if verbose:
            print("CG did not converge")

    return x
