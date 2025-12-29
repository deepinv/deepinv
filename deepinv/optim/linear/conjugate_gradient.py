from typing import Callable
import torch
from torch import Tensor
from deepinv.utils import TensorList, zeros_like
from .utils import dot
import warnings


def conjugate_gradient(
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

        # only update alpha for active dimensions
        alpha = res_old / pAp * active

        x = x + p * alpha
        r = r - Ap * alpha
        res_new = dot(r, r, dim=dim).real

        beta = res_new / res_old

        active = res_new >= tol_sq

        if not torch.any(active):
            if verbose:
                print("CG Converged at iteration", iteration + 1)
            break

        beta = beta * active
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
