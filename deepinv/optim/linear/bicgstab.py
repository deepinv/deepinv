from __future__ import annotations
from .utils import dot
from deepinv.utils.tensorlist import TensorList, zeros_like
import torch
from torch import Tensor
from typing import Callable


def bicgstab(
    A: Callable,
    b: Tensor,
    init: Tensor = None,
    max_iter: int = 1e2,
    tol: float = 1e-5,
    parallel_dim: None | int | list[int] = 0,
    verbose: bool = False,
    left_precon=lambda x: x,
    right_precon=lambda x: x,
) -> Tensor:
    """
    Biconjugate gradient stabilized algorithm.

    Solves :math:`Ax=b` with :math:`A` squared using the BiCGSTAB algorithm in :cite:t:`van1992bi`.

    For more details see: http://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    :param Callable A: Linear operator as a callable function.
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param torch.Tensor init: Optional initial guess.
    :param int max_iter: maximum number of BiCGSTAB iterations.
    :param float tol: absolute tolerance for stopping the BiCGSTAB algorithm.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param bool verbose: Output progress information in the console.
    :param Callable left_precon: left preconditioner as a callable function.
    :param Callable right_precon: right preconditioner as a callable function.
    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...)
    """

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
    r_hat = r.clone()
    rho = dot(r, r_hat, dim=dim)
    p = r
    max_iter = int(max_iter)

    tol = dot(b, b, dim=dim).real * (tol**2)
    eps = torch.finfo(b.dtype).eps  # Breakdown tolerance, to avoid division by zero
    flag = False
    for i in range(max_iter):
        y = right_precon(left_precon(p))
        v = A(y)
        # Safeguard: avoid division by small/zero
        alpha_denom = dot(r_hat, v, dim=dim)
        alpha = torch.where(
            torch.abs(alpha_denom) > eps, rho / alpha_denom, torch.zeros_like(rho)
        )

        h = x + alpha * y
        s = r - alpha * v
        z = right_precon(left_precon(s))
        t = A(z)

        # Safeguard: avoid division by small/zero
        left_s = left_precon(s)
        left_t = left_precon(t)
        omega_num = dot(left_t, left_s, dim=dim)
        omega_denom = dot(left_t, left_t, dim=dim)
        omega = torch.where(
            torch.abs(omega_denom) > eps,
            omega_num / omega_denom,
            torch.zeros_like(omega_num),
        )

        x = h + omega * z
        r = s - omega * t
        if torch.all(dot(r, r, dim=dim).real < tol):
            flag = True
            if verbose:
                print("BiCGSTAB Converged at iteration", i)
            break

        rho_new = dot(r, r_hat, dim=dim)
        # Safeguard for beta: if rho or omega small, set beta=0
        beta_mask = (torch.abs(rho) > eps) & (torch.abs(omega) > eps)
        beta = torch.where(
            beta_mask, (rho_new / rho) * (alpha / omega), torch.zeros_like(rho_new)
        )
        p = r + beta * (p - omega * v)
        rho = rho_new

    if not flag and verbose:
        print("BiCGSTAB did not converge")

    return x
