import torch
from typing import Callable
from types import SimpleNamespace
from deepinv.utils.tensorlist import TensorList, zeros_like, ones_like
from .utils import _sym_ortho


def lsmr(
    A: Callable,
    AT: Callable,
    b: torch.Tensor,
    eta: float | torch.Tensor = 0.0,
    x0: torch.Tensor = None,
    tol: float = 1e-6,
    stagtol: float = 1e-6,
    conlim: float = 1e8,
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
    :param torch. b: input  of shape (B, ...)
    :param float, torch.Tensor eta: damping parameter :math:`eta \geq 0`. Can be batched (shape (B, ...)) or a scalar.
    :param None, torch.Tensor x0: Optional :math:`x_0`, which is also used as the initial guess.
    :param float tol: relative tolerance for stopping the LSMR algorithm.
    :param float stagtol: absolute tolerance for stopping the LSMR algorithm if iterates stagnate.
    :param float conlim: maximum value of the condition number of the system.
    :param int max_iter: maximum number of LSMR iterations.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param None, int restart: cycle of iterations to restart the algorithm to avoid loss of orthogonality.
    :param bool verbose: Output progress information in the console.
    :return: (:class:`torch.`) :math:`x` of shape (B, ...), (:class:`torch.`) condition number of the system.
    """

    xt = AT(b)

    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]
    if parallel_dim is None:
        parallel_dim = []

    if isinstance(b, TensorList):
        device = b[0].device
    else:
        device = b.device

    def normf(u):
        if isinstance(u, TensorList):
            total = 0.0
            dims = [[i for i in range(bi.ndim) if i not in parallel_dim] for bi in b]
            for k in range(len(u)):
                total += torch.linalg.vector_norm(
                    u[k], dim=dims[k], keepdim=False
                ) ** 2 # don't keep dim as dims might be different
            return torch.sqrt(total)
        else:
            dim = [i for i in range(u.ndim) if i not in parallel_dim]
            return torch.linalg.vector_norm(u, dim=dim, keepdim=False)

    b_shape = []
    if isinstance(b, TensorList):
        for j in range(len(b)):
            b_shape.append([])
            for i in range(len(b[j].shape)):
                b_shape[j].append(b[j].shape[i] if i in parallel_dim else 1)
    else:
        for i in range(len(b.shape)):
            b_shape.append(b.shape[i] if i in parallel_dim else 1)

    Atb_shape = []
    for i in range(len(xt.shape)):
        Atb_shape.append(xt.shape[i] if i in parallel_dim else 1)

    def scalar(v, alpha, b_domain):
        if b_domain:
            if isinstance(v, TensorList):
                return TensorList(
                    [
                        vi * alpha.view(bi_shape)
                        for vi, bi_shape in zip(v, b_shape, strict=True)
                    ]
                )
            else:
                return v * alpha.view(b_shape)
        else:
            return v * alpha.view(Atb_shape)

    def _reset_state(x):
        s = SimpleNamespace()

        s.u = b.clone() - A(x)
        s.beta = normf(s.u)

        if torch.all(s.beta > 0):
            s.u = scalar(s.u, 1 / s.beta, b_domain=True)
            s.v = AT(s.u)
            s.alpha = normf(s.v)
        else:
            s.v = torch.zeros_like(x, device=device)
            s.alpha = torch.zeros(1, device=device)

        if torch.all(s.alpha > 0):
            s.v = scalar(s.v, 1 / s.alpha, b_domain=False)

        s.zetabar = s.alpha * s.beta
        s.alphabar = s.alpha.clone()
        s.rho = 1.0
        s.rhobar = torch.ones(1, device=device)
        s.cbar = 1.0
        s.sbar = 0.0
        s.h = s.v.clone()
        s.hbar = torch.zeros_like(s.v, device=device)

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

    if eta is None:
        eta = 0.0
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, device=device)
    if eta.ndim > 0:  # if batched eta
        if eta.size(0) != b.size(0):
            raise ValueError(
                "If eta is batched, its batch size must match the one of b."
            )
        else:  # ensure eta has ndim as b
            eta = eta.squeeze()

    if torch.any(eta < 0):
        raise ValueError(
            "Damping parameter eta must be non-negative. LSMR cannot be applied to problems with negative eta."
        )

    damp = torch.sqrt(eta)

    if x0 is None:
        x = zeros_like(xt)
    else:
        if isinstance(x0, float):
            x = x0 * ones_like(xt)
        else:
            x = x0.clone()

    init = _reset_state(x)

    bnorm = normf(b)

    maxrbar = torch.zeros_like(init.beta, device=device)
    minrbar = torch.full_like(init.beta, torch.inf, device=device)
    acond = 1.0

    rnorm = init.beta

    arnorm = init.alpha * init.beta
    if torch.all(arnorm == 0):
        return x, acond

    #    if torch.all(bnorm == 0):
    #        x = zeros_like(xt)
    #        return x, acond

    flag = False
    for itn in range(max_iter):
        if restart is not None and itn > 0:
            if itn % restart == 0:
                init = _reset_state(x)

        init.u = A(init.v) - scalar(init.u, init.alpha, b_domain=True)
        init.beta = normf(init.u)

        if torch.all(init.beta > 0):
            init.u = scalar(init.u, 1 / init.beta, b_domain=True)
            init.v = AT(init.u) - scalar(init.v, init.beta, b_domain=False)
            init.alpha = normf(init.v)
            if torch.all(init.alpha > 0):
                init.v = scalar(init.v, 1 / init.alpha, b_domain=False)

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

        if torch.any(init.rho == 0) or torch.any(init.rhobar == 0):
            if verbose:
                print(
                    "Error: poorly behaved rotation results in division by zero, try a non-zero eta."
                )
            break

        t1 = (init.rho * thetabar) / (rhobar0 * rho0)
        t2 = init.zeta / (init.rhobar * init.rho)
        t3 = theta / init.rho
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

        init.tautilde0 = (zeta0 - init.thetatilde0 * init.tautilde0) / init.rhotilde0
        taud = (init.zeta - init.thetatilde * init.tautilde0) / init.rhod0
        # we already checked for rhod to not be 0 so this should be safe without checks

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

        if torch.all(rnorm <= tol * bnorm):
            flag = True
            if verbose:
                print("LSMR converged at iteration", itn + 1)
            break
        elif torch.all(search_update_norm <= stagtol * xnorm):
            flag = True
            if verbose:
                print("LSMR stagnated at iteration", itn + 1)
            break
        elif torch.any(acond > conlim):
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
