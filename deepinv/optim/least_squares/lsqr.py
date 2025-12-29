import torch
from torch import Tensor
from typing import Callable
from deepinv.utils import TensorList


def lsqr(
    A: Callable,
    AT: Callable,
    b: Tensor,
    eta: float | torch.Tensor = 0.0,
    x0: Tensor = None,
    tol: float = 1e-6,
    conlim: float = 1e8,
    max_iter: int = 100,
    parallel_dim: None | int | list[int] = 0,
    verbose: bool = False,
    **kwargs,
) -> Tensor:
    r"""
    LSQR algorithm for solving linear systems.

    Code adapted from SciPy's implementation of LSQR: https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py

    The function solves the linear system :math:`\min_x \|Ax-b\|^2 + \eta \|x-x_0\|^2` in the least squares sense
    using the LSQR algorithm from

    Paige, C. C. and M. A. Saunders, "LSQR: An Algorithm for Sparse Linear Equations And Sparse Least Squares," ACM Trans. Math. Soft., Vol.8, 1982, pp. 43-71.

    :param Callable A: Linear operator as a callable function.
    :param Callable AT: Adjoint operator as a callable function.
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param float, torch.Tensor eta: damping parameter :math:`eta \geq 0`. Can be batched (shape (B, ...)) or a scalar.
    :param None, torch.Tensor x0: Optional :math:`x_0`, which is also used as the initial guess.
    :param float tol: relative tolerance for stopping the LSQR algorithm.
    :param float conlim: maximum value of the condition number of the system.
    :param int max_iter: maximum number of LSQR iterations.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param bool verbose: Output progress information in the console.
    :retrun: (:class:`torch.Tensor`) :math:`x` of shape (B, ...), (:class:`torch.Tensor`) condition number of the system.
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
                )  # don't keep dim as dims might be different
            return total
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
                        for vi, bi_shape in zip_strict(v, b_shape)
                    ]
                )
            else:
                return v * alpha.view(b_shape)
        else:
            return v * alpha.view(Atb_shape)

    if eta is None:
        eta = 0.0
    if not isinstance(eta, Tensor):
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
            "Damping parameter eta must be non-negative. LSQR cannot be applied to problems with negative eta."
        )

    # this should be safe as eta should be non-negative
    eta_sqrt = torch.sqrt(eta)

    # ctol = 1 / conlim if conlim > 0 else 0
    anorm = 0.0
    acond = torch.zeros(1, device=device)
    dampsq = eta
    ddnorm = 0.0
    # res2 = 0.0
    # xnorm = 0.0
    xxnorm = 0.0
    z = 0.0
    cs2 = -1.0
    sn2 = 0.0

    u = b.clone()
    bnorm = normf(b)

    if x0 is None:
        x = zeros_like(xt)
        beta = bnorm
    else:
        if isinstance(x0, float):
            x = x0 * zeros_like(xt)
        else:
            x = x0.clone()

        u -= A(x)
        beta = normf(u)

    if torch.all(beta > 0):
        u = scalar(u, 1 / beta, b_domain=True)
        v = AT(u)
        alpha = normf(v)
    else:
        v = torch.zeros_like(x)
        alpha = torch.zeros(1, device=device)

    if torch.all(alpha > 0):
        v = scalar(v, 1 / alpha, b_domain=False)  # v / view(alpha, Atb_shape)

    w = v.clone()
    rhobar = alpha
    phibar = beta
    arnorm = alpha * beta

    if torch.any(arnorm == 0):
        return x, acond

    flag = False
    for itn in range(max_iter):
        u = A(v) - scalar(u, alpha, b_domain=True)
        beta = normf(u)

        if torch.all(beta > 0):
            u = scalar(u, 1 / beta, b_domain=True)
            anorm = torch.sqrt(anorm**2 + alpha**2 + beta**2 + dampsq)
            v = AT(u) - scalar(v, beta, b_domain=False)
            alpha = normf(v)
            if torch.all(alpha > 0):
                v = scalar(v, 1 / alpha, b_domain=False)

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

        t1 = phi / rho
        t2 = -theta / rho
        dk = scalar(w, 1 / rho, b_domain=False)

        x = x + scalar(w, t1, b_domain=False)
        w = v + scalar(w, t2, b_domain=False)
        ddnorm = ddnorm + normf(dk) ** 2

        # if calc_var:
        #    var = var + dk ** 2

        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        # zbar = rhs / gambar
        # xnorm = torch.sqrt(xxnorm + zbar ** 2)
        gamma = torch.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        acond = anorm * torch.sqrt(ddnorm).mean()
        rnorm = torch.sqrt(phibar**2 + psi**2)
        # arnorm = alpha * abs(tau)

        if torch.all(rnorm <= tol * bnorm):
            flag = True
            if verbose:
                print("LSQR converged at iteration", itn)
            break
        elif torch.any(acond > conlim):
            flag = True
            if verbose:
                print(f"LSQR reached condition number limit {conlim} at iteration", itn)
            break

    if not flag and verbose:
        print("LSQR did not converge")

    return x, acond.sqrt()
