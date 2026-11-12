from __future__ import annotations
from .utils import (
    _batched_dot as dot,
    _as_dim_list,
    _resolve_stagtol,
    _sample_shape,
    _make_scalar,
)
from deepinv.utils.tensorlist import zeros_like
import torch
from typing import Callable


def bicgstab(
    A: Callable,
    b: torch.Tensor,
    init: torch.Tensor | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    parallel_dim: None | int | list[int] = 0,
    verbose: bool = False,
    left_precon=lambda x: x,
    right_precon=lambda x: x,
    stagtol: float | None = None,
) -> torch.Tensor:
    """
    Biconjugate gradient stabilized algorithm.

    Solves :math:`Ax=b` with :math:`A` squared using the BiCGSTAB algorithm in :cite:t:`van1992bi`.

    For more details see: http://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method

    .. note::
        A stagnation eps (configured via ``stagtol``) is used to prevent the algorithm from entering an infinite loop when numerical precision limits prevent further progress.
        This condition is verified via the norm of the search update: :math:`\frac{\|x_{k+1} - x_k\|_2}{\|x_k\|_2} \leq \text{stagtol}`.

    :param Callable A: Linear operator as a callable function.
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param torch.Tensor init: Optional initial guess.
    :param int max_iter: maximum number of BiCGSTAB iterations.
    :param float tol: relative tolerance for stopping the BiCGSTAB algorithm based on the size of the residual: :math:`\frac{\|b - Ax_k\|_2}{\|b\|_2} \leq \text{tol}`.
    :param float stagtol: absolute tolerance for stopping the BiCGSTAB algorithm if iterates stagnate, default via dtype precision.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param bool verbose: Output progress information in the console.
    :param Callable left_precon: left preconditioner as a callable function. **Experimental / currently untested.**
    :param Callable right_precon: right preconditioner as a callable function. **Experimental / currently untested.**
    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...)
    """

    stagtol = _resolve_stagtol(stagtol, b)

    parallel_dim = _as_dim_list(parallel_dim)
    shape = _sample_shape(b, parallel_dim)
    scalar = _make_scalar(shape, shape)

    if init is not None:
        x = init.clone()
    else:
        x = zeros_like(b)

    r = b - A(x)
    r_hat = r.clone()
    rho = dot(r, r_hat, parallel_dim)
    p = r
    max_iter = int(max_iter)

    b_norm_sq = dot(b, b, parallel_dim).real
    # handles case b=0
    b_norm_sq = torch.where(b_norm_sq > 0, b_norm_sq, torch.ones_like(b_norm_sq))
    stagtol = stagtol**2
    tol = b_norm_sq * (tol**2)

    eps = torch.finfo(b.dtype).eps  # Breakdown tolerance, to avoid division by zero
    flag = False
    for i in range(max_iter):
        y = right_precon(left_precon(p))
        v = A(y)
        # Safeguard: avoid division by small/zero
        alpha_denom = dot(r_hat, v, parallel_dim)
        alpha = torch.where(
            torch.abs(alpha_denom) > eps, rho / alpha_denom, torch.zeros_like(rho)
        )

        h = x + scalar(y, alpha, b_domain=True)
        s = r - scalar(v, alpha, b_domain=True)
        left_s = left_precon(s)
        z = right_precon(left_s)
        t = A(z)

        # Safeguard: avoid division by small/zero
        left_t = left_precon(t)
        omega_num = dot(left_t, left_s, parallel_dim)
        omega_denom = dot(left_t, left_t, parallel_dim)
        omega = torch.where(
            torch.abs(omega_denom) > eps,
            omega_num / omega_denom,
            torch.zeros_like(omega_num),
        )

        x = h + scalar(z, omega, b_domain=True)
        r = s - scalar(t, omega, b_domain=True)

        search_update = scalar(y, alpha, b_domain=True) + scalar(
            z, omega, b_domain=True
        )
        search_update_norm = dot(search_update, search_update, parallel_dim).real
        xnorm = dot(x, x, parallel_dim).real

        if torch.all(dot(r, r, parallel_dim).real <= tol):
            flag = True
            if verbose:
                print("BiCGSTAB converged at iteration", i + 1)
            break
        elif torch.all(search_update_norm <= stagtol * xnorm):
            flag = True
            if verbose:
                print("BiCGSTAB stagnated at iteration", i + 1)
            break

        rho_new = dot(r, r_hat, parallel_dim)
        # Safeguard for beta: if rho or omega small, set beta=0
        beta_mask = (torch.abs(rho) > eps) & (torch.abs(omega) > eps)
        beta = torch.where(
            beta_mask, (rho_new / rho) * (alpha / omega), torch.zeros_like(rho_new)
        )
        p = r + scalar(p - scalar(v, omega, b_domain=True), beta, b_domain=True)
        rho = rho_new

    if not flag and verbose:
        print("BiCGSTAB did not converge")

    return x
