from __future__ import annotations
import torch
from typing import Callable
from .utils import (
    _sym_ortho,
    _safe_denom,
    _as_dim_list,
    _batched_norm,
    _sample_shape,
    _make_scalar,
    _validate_eta,
    _init_lsq_solution,
    _all_zero,
    _resolve_stagtol,
)


def lsqr(
    A: Callable,
    AT: Callable,
    b: torch.Tensor,
    eta: float | torch.Tensor = 0.0,
    x0: torch.Tensor = None,
    tol: float = 1e-6,
    stagtol: float | None = None,
    conlim: float | None = None,
    max_iter: int = 100,
    parallel_dim: None | int | list[int] = 0,
    verbose: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""
    LSQR algorithm for solving linear systems.

    Code adapted from SciPy's implementation of LSQR: https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py

    The function solves the linear system :math:`\min_x \|Ax-b\|^2 + \eta \|x-x_0\|^2` in the least squares sense
    using the LSQR algorithm of :cite:t:`paige1982lsqr`.

    :param Callable A: Linear operator as a callable function.
    :param Callable AT: Adjoint operator as a callable function.
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param float, torch.Tensor eta: damping parameter :math:`eta \geq 0`. Can be batched (shape (B, ...)) or a scalar.
    :param None, torch.Tensor x0: Optional :math:`x_0`, which is also used as the initial guess.
    :param float tol: relative tolerance for stopping the LSQR algorithm.
    :param float stagtol: absolute tolerance for stopping the LSQR algorithm if iterates stagnate, default via dtype precision.
    :param float conlim: maximum value of the condition number of the system, default via dtype precision.
    :param int max_iter: maximum number of LSQR iterations.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param bool verbose: Output progress information in the console.
    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...), (:class:`torch.Tensor`) condition number of the system.
    """

    stagtol = _resolve_stagtol(stagtol, b)

    if conlim is not None:
        conlim = 12.0 * torch.finfo(b.dtype).eps
        #multiplication causes the default to be 1e8 on single precision

    parallel_dim = _as_dim_list(parallel_dim)
    device = b.device

    x, xt, x_ref = _init_lsq_solution(x0, b, AT)
    x_is_zero = _all_zero(x)

    def normf(u):
        return _batched_norm(u, parallel_dim)

    b_shape = _sample_shape(b, parallel_dim)
    Atb_shape = _sample_shape(x_ref, parallel_dim)
    scalar = _make_scalar(b_shape, Atb_shape)

    eta = _validate_eta(eta, b, device)
    # this should be safe as eta should be non-negative
    eta_sqrt = torch.sqrt(eta)

    # ctol = 1 / conlim if conlim > 0 else 0
    anorm = 0.0
    acond = torch.zeros(1, device=device)
    dampsq = eta
    ddnorm = 0.0
    # res2 = 0.0
    xnorm = 0.0
    xxnorm = 0.0
    z = 0.0
    cs2 = -1.0
    sn2 = 0.0

    u = b.clone()
    bnorm = normf(b)

    if x_is_zero:
        u = b.clone()
        beta = bnorm
    else:
        u = b.clone() - A(x)
        beta = normf(u)

    # Safe per-element normalization: elements whose residual is already zero
    # (beta == 0) keep a divisor of 1 so a single trivial system in a batch does
    # not collapse the whole batch (they stay at 0).
    safe_beta = _safe_denom(beta)
    u = scalar(u, 1 / safe_beta, b_domain=True)
    if xt is not None and x_is_zero:
        v = scalar(xt, 1 / safe_beta, b_domain=False)
    else:
        v = AT(u)
    alpha = normf(v)

    safe_alpha = _safe_denom(alpha)
    v = scalar(v, 1 / safe_alpha, b_domain=False)  # v / view(alpha, Atb_shape)

    w = v.clone()
    rhobar = alpha
    phibar = beta
    arnorm = alpha * beta

    if torch.all(arnorm == 0):
        return x, acond

    flag = False
    for itn in range(max_iter):
        u = A(v) - scalar(u, alpha, b_domain=True)
        beta = normf(u)

        # Safe per-element normalization (see init): a converged element
        # (beta == 0) uses a divisor of 1 and stays put instead of stalling the batch.
        safe_beta = _safe_denom(beta)
        u = scalar(u, 1 / safe_beta, b_domain=True)
        anorm = torch.sqrt(anorm**2 + alpha**2 + beta**2 + dampsq)
        v = AT(u) - scalar(v, beta, b_domain=False)
        alpha = normf(v)
        safe_alpha = _safe_denom(alpha)
        v = scalar(v, 1 / safe_alpha, b_domain=False)

        if torch.any(eta > 0):
            rhobar1 = torch.sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = eta_sqrt / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.0

        cs, sn, rho = _sym_ortho(rhobar1, beta)
        theta = sn * alpha
        rhobar = -cs * alpha
        phi = cs * phibar
        phibar = sn * phibar
        # tau = sn * phi

        # _safe_denom keeps a converged/trivial batch entry (rho == 0) from
        # turning these updates into 0 / 0 = NaN; its numerators are 0 too, so
        # it contributes a 0 update and stays put.
        safe_rho = _safe_denom(rho)
        t1 = phi / safe_rho
        t2 = -theta / safe_rho

        search_update = scalar(w, t1, b_domain=False)

        x = x + search_update
        # ``dk = w / rho`` is only needed for the condition-number estimate; since
        # ``rho`` is per-sample, ``||dk|| = ||w|| / |rho|``, so accumulate ``ddnorm``
        # without allocating the full-vector ``dk`` (uses ``w`` before its update).
        ddnorm = ddnorm + normf(w) ** 2 / safe_rho**2
        w = v + scalar(w, t2, b_domain=False)

        # if calc_var:
        #    var = var + dk ** 2

        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = torch.sqrt(xxnorm + zbar**2)
        gamma = torch.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        acond = anorm * torch.sqrt(ddnorm)
        rnorm = torch.sqrt(phibar**2 + psi**2)
        # arnorm = alpha * abs(tau)

        search_update_norm = normf(search_update)
        converged = rnorm <= tol * bnorm

        if torch.all(converged):
            flag = True
            if verbose:
                print("LSQR converged at iteration", itn + 1)
            break
        elif torch.all(search_update_norm <= stagtol * xnorm):
            flag = True
            if verbose:
                print("LSQR stagnated at iteration", itn + 1)
            break
        elif torch.all(converged | (acond > conlim)):
            # Stop once every sample is either converged or too ill-conditioned
            # to progress: a single ill-conditioned sample is still detected here,
            # while the converged samples keep their solution (no batch collapse).
            flag = True
            if verbose:
                print(
                    f"LSQR reached condition number limit {conlim} at iteration",
                    itn + 1,
                )
            break

    if not flag and verbose:
        print("LSQR did not converge")

    return x, acond.sqrt()
