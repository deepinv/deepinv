from __future__ import annotations
import torch
from typing import Callable
from deepinv.utils.tensorlist import zeros_like
from .utils import (
    _sym_ortho,
    _as_dim_list,
    _batched_norm,
    _sample_shape,
    _scalar,
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
    conlim: float | None = None,
    max_iter: int = 100,
    parallel_dim: None | int | list[int] = 0,
    restart: int | None = None,
    verbose: bool = False,
    stagtol: float | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    LSMR algorithm for solving linear systems.

    Code adapted from SciPy's implementation of LSMR: https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsmr.py

    The function solves the linear system :math:`\min_x \|Ax-b\|^2 + \eta \|x-x_0\|^2` in the least squares sense
    using the LSMR algorithm of :cite:t:`fong2011lsmr`.

    .. note::
        A stagnation eps (configured via ``stagtol``) is used to prevent the algorithm from entering an infinite loop when numerical precision limits prevent further progress.
        This condition is verified via the norm of the search update: :math:`\frac{\|x_{k+1} - x_k\|_2}{\|x_k\|_2} \leq \text{stagtol}`.

    :param Callable A: Linear operator as a callable function.
    :param Callable AT: Adjoint operator as a callable function.
    :param torch.Tensor b: input  of shape (B, ...)
    :param float, torch.Tensor eta: damping parameter :math:`eta \geq 0`. Can be batched (shape (B, ...)) or a scalar.
    :param None, torch.Tensor x0: Optional :math:`x_0`, which is also used as the initial guess.
    :param float tol: relative tolerance for stopping the LSMR algorithm based on the size of the residual: :math:`\frac{\|b - Ax_k\|_2}{\|b\|_2} \leq \text{tol}`.
    :param float stagtol: absolute tolerance for stopping the LSMR algorithm if iterates stagnate, default via dtype precision.
    :param float conlim: maximum value of the condition number of the system, default via dtype precision.
    :param int max_iter: maximum number of LSMR iterations.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param None, int restart: cycle of iterations to restart the algorithm to avoid loss of orthogonality.
    :param bool verbose: Output progress information in the console.
    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...), (:class:`torch.Tensor`) condition number of the system.
    """

    stagtol = _resolve_stagtol(stagtol, b)

    if conlim is None:
        conlim = 12.0 / torch.finfo(b.dtype).eps
        # multiplication causes the default to be 1e8 on single precision

    parallel_dim = _as_dim_list(parallel_dim)
    device = b.device

    x, xt, x_ref = _init_lsq_solution(x0, b, AT)

    def normf(u):
        return _batched_norm(u, parallel_dim)

    b_shape = _sample_shape(b, parallel_dim)
    Atb_shape = _sample_shape(x_ref, parallel_dim)
    x_ref = None  # free up memory

    bnorm = normf(b)

    def _reset_state(x: torch.Tensor) -> tuple:
        """
        Initialise `'lsmr'` at ``x``, called at the start with ``x0`` and at east restart with the current iterate.

        :param torch.Tensor x: Current iterate.
        :return: (:class:`tuple`) Bidiagonalisation tensors and scalars, Givens rotations, update directions and residual norm recurrences.
        """
        x_is_zero = _all_zero(x)

        if x_is_zero:
            u = b.clone()
            beta = bnorm
        else:
            u = b - A(x)
            beta = normf(u)

        # Safe per-element normalization: elements whose residual is already
        # zero (beta == 0) keep a divisor of 1 so a single trivial system in a
        # batch does not collapse the whole batch (they stay at 0).
        safe_beta = torch.where(beta != 0, beta, torch.ones_like(beta))
        u = _scalar(u, 1 / safe_beta, shape=b_shape)
        if xt is not None and x_is_zero:
            v = _scalar(xt, 1 / safe_beta, shape=Atb_shape)
        else:
            v = AT(u)
        alpha = normf(v)

        safe_alpha = torch.where(alpha != 0, alpha, torch.ones_like(alpha))
        v = _scalar(v, 1 / safe_alpha, shape=Atb_shape)

        zetabar = alpha * beta
        alphabar = alpha.clone()
        rho = 1.0
        rhobar = torch.ones(1, device=device)
        cbar = 1.0
        sbar = 0.0
        h = v.clone()
        hbar = zeros_like(v)

        betadd = beta.clone()
        betad = torch.zeros_like(beta, device=device)
        rhod0 = torch.ones_like(beta, device=device)
        tautilde0 = 0.0
        thetatilde = 0.0
        zeta = 0.0
        d = 0.0

        return (
            u,
            v,
            alpha,
            beta,
            alphabar,
            zetabar,
            rho,
            rhobar,
            cbar,
            sbar,
            h,
            hbar,
            betadd,
            betad,
            rhod0,
            tautilde0,
            thetatilde,
            zeta,
            d,
        )

    eta = _validate_eta(eta, b, device)
    damp = torch.sqrt(eta)

    (
        u,
        v,
        alpha,
        beta,
        alphabar,
        zetabar,
        rho,
        rhobar,
        cbar,
        sbar,
        h,
        hbar,
        betadd,
        betad,
        rhod0,
        tautilde0,
        thetatilde,
        zeta,
        d,
    ) = _reset_state(x)
    xt = None  # free up memory (x will not be zero on reset almost certainly)

    maxrbar = torch.zeros_like(beta, device=device)
    minrbar = torch.full_like(beta, torch.inf, device=device)
    acond = torch.ones_like(beta)

    rnorm = beta

    arnorm = alpha * beta
    if torch.all(arnorm == 0):
        return x, acond

    flag = False
    fresh = True
    for itn in range(max_iter):
        if restart is not None and itn > 0 and itn % restart == 0:
            (
                u,
                v,
                alpha,
                beta,
                alphabar,
                zetabar,
                rho,
                rhobar,
                cbar,
                sbar,
                h,
                hbar,
                betadd,
                betad,
                rhod0,
                tautilde0,
                thetatilde,
                zeta,
                d,
            ) = _reset_state(x)
            fresh = True

        u = A(v) - _scalar(u, alpha, shape=b_shape)
        beta = normf(u)

        # Safe per-element normalization (see _reset_state): a converged element
        # (beta == 0) uses a divisor of 1 and stays put instead of stalling the batch.
        safe_beta = torch.where(beta != 0, beta, torch.ones_like(beta))
        u = _scalar(u, 1 / safe_beta, shape=b_shape)
        v = AT(u) - _scalar(v, beta, shape=Atb_shape)
        alpha = normf(v)
        safe_alpha = torch.where(alpha != 0, alpha, torch.ones_like(alpha))
        v = _scalar(v, 1 / safe_alpha, shape=Atb_shape)

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        rho0 = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        theta = s * alpha
        alphabar = c * alpha

        rhobar0 = rhobar
        zeta0 = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, theta)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        if torch.all(rho == 0) or torch.all(rhobar == 0):
            if verbose:
                print(
                    "LSMR stagnated: zero rotation for all samples, try a non-zero eta."
                )
            flag = True
            # the print above already states that LSMR stagnated
            # flag = True prevents redundancy of also printing that it did not converge
            break

        rhobar0rho0 = rhobar0 * rho0
        t1 = (rho * thetabar) / torch.where(
            rhobar0rho0 != 0, rhobar0rho0, torch.ones_like(rhobar0rho0)
        )
        rhobarrho = rhobar * rho
        t2 = zeta / torch.where(rhobarrho != 0, rhobarrho, torch.ones_like(rhobarrho))
        t3 = theta / torch.where(rho != 0, rho, torch.ones_like(rho))
        hbar = h - _scalar(hbar, t1, shape=Atb_shape)
        x = x + _scalar(hbar, t2, shape=Atb_shape)
        search_update_norm = t2.abs() * normf(hbar)
        h = v - _scalar(h, t3, shape=Atb_shape)

        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute

        thetatilde0 = thetatilde
        ctilde0, stilde0, rhotilde0 = _sym_ortho(rhod0, thetabar)
        thetatilde = stilde0 * rhobar
        rhod0 = ctilde0 * rhobar
        betad = -stilde0 * betad + ctilde0 * betahat

        tautilde0 = (zeta0 - thetatilde0 * tautilde0) / torch.where(
            rhotilde0 != 0, rhotilde0, torch.ones_like(rhotilde0)
        )
        taud = (zeta - thetatilde * tautilde0) / torch.where(
            rhod0 != 0, rhod0, torch.ones_like(rhod0)
        )

        d = d + betacheck * betacheck
        gamma = d + (betad - taud) ** 2 + betadd * betadd
        rnorm = torch.sqrt(gamma)

        if not fresh:
            maxrbar = torch.maximum(maxrbar, rhobar0)
            minrbar = torch.minimum(minrbar, rhobar0)
        fresh = False
        sigmamax = torch.maximum(maxrbar, rhotemp)
        sigmamin = torch.minimum(minrbar, rhotemp)

        sigmamin_nonzero = torch.where(sigmamin == 0, 1.0, sigmamin)
        acond = torch.where(sigmamin == 0, torch.inf, sigmamax / sigmamin_nonzero)

        xnorm = normf(x)
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
