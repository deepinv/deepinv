from __future__ import annotations
import torch
from typing import Callable
from types import SimpleNamespace
from deepinv.utils.tensorlist import zeros_like
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


def lsmr(
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
    restart: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> torch.Tensor:
    r"""
    LSMR algorithm for solving linear systems.

    Code adapted from SciPy's implementation of LSMR: https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsmr.py

    The function solves the linear system :math:`\min_x \|Ax-b\|^2 + \eta \|x-x_0\|^2` in the least squares sense
    using the LSMR algorithm of :cite:t:`fong2011lsmr`.

    :param Callable A: Linear operator as a callable function.
    :param Callable AT: Adjoint operator as a callable function.
    :param torch.Tensor b: input  of shape (B, ...)
    :param float, torch.Tensor eta: damping parameter :math:`eta \geq 0`. Can be batched (shape (B, ...)) or a scalar.
    :param None, torch.Tensor x0: Optional :math:`x_0`, which is also used as the initial guess.
    :param float tol: relative tolerance for stopping the LSMR algorithm.
    :param float stagtol: absolute tolerance for stopping the LSMR algorithm if iterates stagnate, default via dtype precision.
    :param float conlim: maximum value of the condition number of the system, default via dtype precision.
    :param int max_iter: maximum number of LSMR iterations.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param None, int restart: cycle of iterations to restart the algorithm to avoid loss of orthogonality.
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

    def normf(u):
        return _batched_norm(u, parallel_dim)

    b_shape = _sample_shape(b, parallel_dim)
    Atb_shape = _sample_shape(x_ref, parallel_dim)
    scalar = _make_scalar(b_shape, Atb_shape)

    bnorm = normf(b)

    def _reset_state(x):
        s = SimpleNamespace()

        x_is_zero = _all_zero(x)

        if x_is_zero:
            s.u = b.clone()
            s.beta = bnorm
        else:
            s.u = b.clone() - A(x)
            s.beta = normf(s.u)

        # Safe per-element normalization: elements whose residual is already
        # zero (beta == 0) keep a divisor of 1 so a single trivial system in a
        # batch does not collapse the whole batch (they stay at 0).
        safe_beta = _safe_denom(s.beta)
        s.u = scalar(s.u, 1 / safe_beta, b_domain=True)
        if xt is not None and x_is_zero:
            s.v = scalar(xt, 1 / safe_beta, b_domain=False)
        else:
            s.v = AT(s.u)
        s.alpha = normf(s.v)

        safe_alpha = _safe_denom(s.alpha)
        s.v = scalar(s.v, 1 / safe_alpha, b_domain=False)

        s.zetabar = s.alpha * s.beta
        s.alphabar = s.alpha.clone()
        s.rho = 1.0
        s.rhobar = torch.ones(1, device=device)
        s.cbar = 1.0
        s.sbar = 0.0
        s.h = s.v.clone()
        s.hbar = zeros_like(s.v)

        s.betadd = s.beta.clone()
        s.betad = torch.zeros_like(s.beta, device=device)
        s.rhod0 = torch.ones_like(s.beta, device=device)
        s.tautilde0 = 0.0
        s.thetatilde = 0.0
        s.zeta = 0.0
        s.d = 0.0

        # these initializations are faster than running the Givens rotation in the first iteration
        s.rhotilde0 = 1.0
        s.thetatilde0 = 0.0

        return s

    eta = _validate_eta(eta, b, device)
    damp = torch.sqrt(eta)

    init = _reset_state(x)

    maxrbar = torch.zeros_like(init.beta, device=device)
    minrbar = torch.full_like(init.beta, torch.inf, device=device)
    acond = 1.0

    rnorm = init.beta

    arnorm = init.alpha * init.beta
    if torch.all(arnorm == 0):
        return x, acond

    flag = False
    for itn in range(max_iter):
        if restart is not None and itn > 0:
            if itn % restart == 0:
                init = _reset_state(x)

        init.u = A(init.v) - scalar(init.u, init.alpha, b_domain=True)
        init.beta = normf(init.u)

        # Safe per-element normalization (see _reset_state): a converged element
        # (beta == 0) uses a divisor of 1 and stays put instead of stalling the batch.
        safe_beta = _safe_denom(init.beta)
        init.u = scalar(init.u, 1 / safe_beta, b_domain=True)
        init.v = AT(init.u) - scalar(init.v, init.beta, b_domain=False)
        init.alpha = normf(init.v)
        safe_alpha = _safe_denom(init.alpha)
        init.v = scalar(init.v, 1 / safe_alpha, b_domain=False)

        chat, shat, alphahat = _sym_ortho(init.alphabar, damp)

        rho0 = init.rho
        c, s, init.rho = _sym_ortho(alphahat, init.beta)
        theta = s * init.alpha
        init.alphabar = c * init.alpha

        rhobar0 = init.rhobar
        zeta0 = init.zeta
        thetabar = init.sbar * init.rho
        rhotemp = init.cbar * init.rho
        init.cbar, init.sbar, init.rhobar = _sym_ortho(init.cbar * init.rho, theta)
        init.zeta = init.cbar * init.zetabar
        init.zetabar = -init.sbar * init.zetabar

        if torch.all(init.rho == 0) or torch.all(init.rhobar == 0):
            if verbose:
                print(
                    "Error: poorly behaved rotation results in division by zero, try a non-zero eta."
                )
            break

        # _safe_denom keeps a single converged/trivial batch entry (rho == 0)
        # from turning these updates into 0 / 0 = NaN; its numerators are 0 too,
        # so it contributes a 0 update and stays put.
        t1 = (init.rho * thetabar) / _safe_denom(rhobar0 * rho0)
        t2 = init.zeta / _safe_denom(init.rhobar * init.rho)
        t3 = theta / _safe_denom(init.rho)
        init.hbar = init.h - scalar(init.hbar, t1, b_domain=False)
        search_update = scalar(init.hbar, t2, b_domain=False)
        x = x + search_update
        init.h = init.v - scalar(init.h, t3, b_domain=False)

        betaacute = chat * init.betadd
        betacheck = -shat * init.betadd
        betahat = c * betaacute
        init.betadd = -s * betaacute

        init.thetatilde0 = init.thetatilde
        ctilde0, stilde0, init.rhotilde0 = _sym_ortho(init.rhod0, thetabar)
        init.thetatilde = stilde0 * init.rhobar
        init.rhod0 = ctilde0 * init.rhobar
        init.betad = -stilde0 * init.betad + ctilde0 * betahat

        init.tautilde0 = (
            zeta0 - init.thetatilde0 * init.tautilde0
        ) / _safe_denom(init.rhotilde0)
        taud = (init.zeta - init.thetatilde * init.tautilde0) / _safe_denom(init.rhod0)

        init.d = init.d + betacheck * betacheck
        gamma = init.d + (init.betad - taud) ** 2 + init.betadd * init.betadd
        rnorm = torch.sqrt(gamma)

        maxrbar = torch.maximum(maxrbar, rhobar0)
        if itn > 0:
            minrbar = torch.minimum(minrbar, rhobar0)
        sigmamax = torch.maximum(maxrbar, rhotemp)
        sigmamin = torch.minimum(minrbar, rhotemp)

        sigmamin_nonzero = torch.where(sigmamin == 0, 1.0, sigmamin)
        acond = torch.where(sigmamin == 0, torch.inf, sigmamax / sigmamin_nonzero)

        xnorm = normf(x)
        search_update_norm = normf(search_update)
        converged = rnorm <= tol * bnorm

        if torch.all(converged):
            flag = True
            if verbose:
                print("LSMR converged at iteration", itn + 1)
            break
        elif torch.all(search_update_norm <= stagtol * xnorm):
            flag = True
            if verbose:
                print("LSMR stagnated at iteration", itn + 1)
            break
        elif torch.all(converged | (acond > conlim)):
            # Stop once every sample is either converged or too ill-conditioned
            # to progress: a single ill-conditioned sample is still detected here,
            # while the converged samples keep their solution (no batch collapse).
            flag = True
            if verbose:
                print(
                    f"LSMR reached condition number limit {conlim} at iteration",
                    itn + 1,
                )
            break

    if not flag and verbose:
        print("LSMR did not converge")

    return x, maxrbar / minrbar
