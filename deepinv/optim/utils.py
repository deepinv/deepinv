from __future__ import annotations
from deepinv.utils import zeros_like
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from tqdm import tqdm
import torch.nn as nn
from deepinv.utils.tensorlist import TensorList
from deepinv.utils.compat import zip_strict
import warnings
from typing import Callable


def check_conv(X_prev, X, it, crit_conv="residual", thres_conv=1e-3, verbose=False):
    if crit_conv == "residual":
        if isinstance(X_prev, dict):
            X_prev = X_prev["est"][0]
        if isinstance(X, dict):
            X = X["est"][0]
        crit_cur = (X_prev - X).norm() / (X.norm() + 1e-06)
    elif crit_conv == "cost":
        F_prev = X_prev["cost"]
        F = X["cost"]
        crit_cur = (F_prev - F).norm() / (F.norm() + 1e-06)
    else:
        raise ValueError("convergence criteria not implemented")
    if crit_cur < thres_conv:
        if verbose:
            print(
                f"Iteration {it}, current converge crit. = {crit_cur:.2E}, objective = {thres_conv:.2E} \r"
            )
        return True
    else:
        return False


def least_squares(
    A: Callable,
    AT: Callable,
    y: Tensor,
    z: Tensor | float | None = 0.0,
    init: Tensor | None = None,
    gamma: float | Tensor | None = None,
    parallel_dim: int = 0,
    AAT: Callable | None = None,
    ATA: Callable | None = None,
    solver: str = "CG",
    max_iter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> Tensor:
    r"""
    Solves :math:`\min_x \|Ax-y\|^2 + \frac{1}{\gamma}\|x-z\|^2` using the specified solver.

    The solvers are stopped either when :math:`\|Ax-y\| \leq \text{tol} \times \|y\|` or
    when the maximum number of iterations is reached.

    The solution depends on the regularization parameter :math:`\gamma`:

    - If `gamma=None` (:math:`\gamma = \infty`), it solves the unregularized least squares problem :math:`\min_x \|Ax-y\|^2`.
        - If :math:`A` is overcomplete (rows>=columns), it computes the minimum norm solution :math:`x = A^{\top}(AA^{\top})^{-1}y`.
        - If :math:`A` is undercomplete (columns>rows), it computes the least squares solution :math:`x = (A^{\top}A)^{-1}A^{\top}y`.
    - If :math:`0 < \gamma < \infty`, it computes the least squares solution :math:`x = (A^{\top}A + \frac{1}{\gamma}I)^{-1}(A^{\top}y + \frac{1}{\gamma}z)`.

    .. warning::

        If :math:`\gamma \leq 0`, the problem can become non-convex and the solvers are not designed for that.
        A warning is raised, but solvers continue anyway (except for LSQR, which cannot be used for negative :math:`\gamma`).

    Available solvers are:

    - `'CG'`: `Conjugate Gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.
    - `'BiCGStab'`: `Biconjugate Gradient Stabilized method <https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method>`_
    - `'lsqr'`: `Least Squares QR <https://www-leland.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf>`_
    - `'minres'`: `Minimal Residual Method <https://en.wikipedia.org/wiki/Minimal_residual_method>`_

    .. note::

        Both `'CG'` and `'BiCGStab'` are used for squared linear systems, while `'lsqr'` is used for rectangular systems.

        If the chosen solver requires a squared system, we map to the problem to the normal equations:
        If the size of :math:`y` is larger than :math:`x` (overcomplete problem), it computes :math:`(A^{\top} A)^{-1} A^{\top} y`,
        otherwise (incomplete problem) it computes :math:`A^{\top} (A A^{\top})^{-1} y`.


    :param Callable A: Linear operator :math:`A` as a callable function.
    :param Callable AT: Adjoint operator :math:`A^{\top}` as a callable function.
    :param torch.Tensor y: input tensor of shape (B, ...)
    :param torch.Tensor z: input tensor of shape (B, ...) or scalar.
    :param torch.Tensor init: (Optional) initial guess for the solver. If None, it is set to a tensor of zeros.
    :param None, float, torch.Tensor gamma: (Optional) inverse regularization parameter. Can be batched (shape (B, ...)) or a scalar.
        If multi-dimensional tensor, then its shape must match that of :math:`A^{\top} y`.
        If None, it is set to :math:`\infty` (no regularization).
    :param str solver: solver to be used, options are `'CG'`, `'BiCGStab'`, `'lsqr'` and `'minres'`.
    :param Callable AAT: (Optional) Efficient implementation of :math:`A(A^{\top}(x))`. If not provided, it is computed as :math:`A(A^{\top}(x))`.
    :param Callable ATA: (Optional) Efficient implementation of :math:`A^{\top}(A(x))`. If not provided, it is computed as :math:`A^{\top}(A(x))`.
    :param int max_iter: maximum number of iterations.
    :param float tol: relative tolerance for stopping the algorithm.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param kwargs: Keyword arguments to be passed to the solver.
    :return: (class:`torch.Tensor`) :math:`x` of shape (B, ...).
    """
    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]

    if gamma is None:
        gamma = torch.tensor(0.0, device=y.device)
        gamma_provided = False
    else:
        gamma_provided = True

        if not isinstance(gamma, Tensor):
            gamma = torch.tensor(gamma, device=y.device)

        if torch.any(gamma <= 0):
            warnings.warn(
                "Regularization parameter of least squares problem (gamma) should be positive."
                "Otherwise, the problem can become non-convex and the solvers are not designed for that."
                "Continuing anyway..."
            )

    Aty = AT(y)

    if gamma.ndim > 0:  # if batched gamma
        if isinstance(Aty, TensorList):
            batch_size = Aty[0].size(0)
            ndim = Aty[0].ndim
        else:
            batch_size = Aty.size(0)
            ndim = Aty.ndim

        if gamma.size(0) != batch_size:
            raise ValueError(
                "If gamma is batched, its batch size must match the one of y."
            )
        elif gamma.ndim == 1:  # expand gamma to ATy
            gamma = gamma.view([gamma.size(0)] + [1] * (ndim - 1))
        elif gamma.ndim != ndim:
            raise ValueError(
                f"gamma should either be 0D, 1D, or match same number of dimensions as ATy, but got ndims {gamma.ndim} and {ndim}"
            )

    if solver == "lsqr":  # rectangular solver
        eta = 1 / gamma if gamma_provided else None
        x, _ = lsqr(
            A,
            AT,
            y,
            x0=z,
            eta=eta,
            max_iter=max_iter,
            tol=tol,
            parallel_dim=parallel_dim,
            **kwargs,
        )

    else:
        complete = Aty.shape == y.shape
        overcomplete = Aty.numel() < y.numel()

        if complete and (solver == "BiCGStab" or solver == "minres"):
            H = lambda x: A(x)
            b = y
        else:
            if AAT is None:
                AAT = lambda x: A(AT(x))
            if ATA is None:
                ATA = lambda x: AT(A(x))

            if gamma_provided:
                b = AT(y) + 1 / gamma * z
                H = lambda x: ATA(x) + 1 / gamma * x
                overcomplete = False
            else:
                if not overcomplete:
                    H = lambda x: AAT(x)
                    b = y
                else:
                    H = lambda x: ATA(x)
                    b = Aty

        if solver == "CG":
            x = conjugate_gradient(
                A=H,
                b=b,
                init=init,
                max_iter=max_iter,
                tol=tol,
                parallel_dim=parallel_dim,
                **kwargs,
            )
        elif solver == "BiCGStab":
            x = bicgstab(
                A=H,
                b=b,
                init=init,
                max_iter=max_iter,
                tol=tol,
                parallel_dim=parallel_dim,
                **kwargs,
            )
        elif solver == "minres":
            x = minres(
                A=H,
                b=b,
                init=init,
                max_iter=max_iter,
                tol=tol,
                parallel_dim=parallel_dim,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Solver {solver} not recognized. Choose between 'CG', 'lsqr' and 'BiCGStab'."
            )

        if not gamma_provided and not overcomplete and not complete:
            x = AT(x)
    return x


def dot(a, b, dim):
    if isinstance(a, TensorList):
        aux = 0
        for ai, bi in zip_strict(a.x, b.x):
            aux += (ai.conj() * bi).sum(
                dim=dim, keepdim=True
            )  # performs batched dot product
        dot = aux
    else:
        dot = (a.conj() * b).sum(dim=dim, keepdim=True)  # performs batched dot product
    return dot


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    max_iter: float = 1e2,
    tol: float = 1e-5,
    eps: float = 1e-8,
    parallel_dim=0,
    init=None,
    verbose=False,
):
    """
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param Callable A: Linear operator as a callable function, has to be square!
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param int max_iter: maximum number of CG iterations
    :param float tol: absolute tolerance for stopping the CG algorithm.
    :param float eps: a small value for numerical stability
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param torch.Tensor init: Optional initial guess.
    :param bool verbose: Output progress information in the console.
    :return: torch.Tensor :math:`x` of shape (B, ...) verifying :math:`Ax=b`.

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
    p = r
    rsold = dot(r, r, dim=dim).real
    flag = True
    tol = dot(b, b, dim=dim).real * (tol**2)
    for _ in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / (dot(p, Ap, dim=dim) + eps)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = dot(r, r, dim=dim).real
        if torch.all(rsnew < tol):
            if verbose:
                print("CG Converged at iteration", _)
            flag = False
            break
        p = r + p * (rsnew / (rsold + eps))
        rsold = rsnew

    if flag and verbose:
        print("CG did not converge")

    return x


def bicgstab(
    A,
    b,
    init=None,
    max_iter=1e2,
    tol=1e-5,
    parallel_dim=0,
    verbose=False,
    left_precon=lambda x: x,
    right_precon=lambda x: x,
):
    """
    Biconjugate gradient stabilized algorithm.

    Solves :math:`Ax=b` with :math:`A` squared using the BiCGSTAB algorithm:

    Van der Vorst, H. A. (1992). "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems". SIAM J. Sci. Stat. Comput. 13 (2): 631–644.

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


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Adapted from https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py

    The routine '_sym_ortho' was added for numerical stability. This is
    recommended by S.-C. Choi in "Iterative Methods for Singular Linear Equations and Least-Squares
    Problems".  It removes the unpleasant potential of
    ``1/eps`` in some important places.

    """
    a, b = torch.broadcast_tensors(a, b)
    if torch.any(b == 0):
        return torch.sign(a), 0, a.abs()
    elif torch.any(a == 0):
        return 0, torch.sign(b), b.abs()
    elif torch.any(b.abs() > a.abs()):
        tau = a / b
        s = torch.sign(b) / torch.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = torch.sign(a) / torch.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsqr(
    A,
    AT,
    b,
    eta=0.0,
    x0=None,
    tol=1e-6,
    conlim=1e8,
    max_iter=100,
    parallel_dim=0,
    verbose=False,
    **kwargs,
):
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


def minres(
    A,
    b,
    init=None,
    max_iter=1e2,
    tol=1e-5,
    eps=1e-6,
    parallel_dim=0,
    verbose=False,
    precon=lambda x: x.clone(),
):
    """
    Minimal Residual Method for solving symmetric equations.

    Solves :math:`Ax=b` with :math:`A` symmetric using the MINRES algorithm:

    Christopher C. Paige, Michael A. Saunders (1975). "Solution of sparse indefinite systems of linear equations". SIAM Journal on Numerical Analysis. 12 (4): 617–629.

    The method assumes that :math:`A` is hermite.
    For more details see: https://en.wikipedia.org/wiki/Minimal_residual_method

    Based on https://github.com/cornellius-gp/linear_operator
    Modifications and simplifications for compatibility with deepinverse

    :param Callable A: Linear operator as a callable function.
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param torch.Tensor init: Optional initial guess.
    :param int max_iter: maximum number of MINRES iterations.
    :param float tol: absolute tolerance for stopping the MINRES algorithm.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param bool verbose: Output progress information in the console.
    :param Callable precon: preconditioner is a callable function (not tested). Must be positive definite
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

    # Rescale b
    b_norm = b.norm(2, dim=dim, keepdim=True)
    b_is_zero = b_norm < 1e-10
    b_norm = b_norm.masked_fill(b_is_zero, 1)
    b = b / b_norm

    # Create space for matmul product, solution
    if init is not None:
        solution = init / b_norm
    else:
        solution = torch.zeros(b.shape, dtype=b.dtype, device=b.device)

    # Variables for Lanczos terms
    zvec_prev2 = torch.zeros(solution.shape, device=b.device)  # r_(k-1) in wiki
    zvec_prev1 = b - A(solution)  # r_k in wiki
    qvec_prev1 = precon(zvec_prev1)
    alpha_curr = torch.zeros(b.shape, dtype=b.dtype, device=b.device)
    alpha_curr = alpha_curr.norm(2, dim=dim, keepdim=True)
    beta_prev = torch.abs(dot(zvec_prev1, qvec_prev1, dim=dim).sqrt()).clamp_min(eps)

    # Divide by beta_prev
    zvec_prev1 = zvec_prev1 / beta_prev
    qvec_prev1 = qvec_prev1 / beta_prev

    # Variables for the QR rotation
    # 1) Components of the Givens rotations
    cos_prev2 = torch.ones(alpha_curr.shape, dtype=b.dtype, device=b.device)
    sin_prev2 = torch.zeros(alpha_curr.shape, dtype=b.dtype, device=b.device)
    cos_prev1 = cos_prev2
    sin_prev1 = sin_prev2

    # Variables for the solution updates
    # 1) The "search" vectors of the solution
    # Equivalent to the vectors of Q R^{-1}, where Q is the matrix of Lanczos vectors and
    # R is the QR factor of the tridiagonal Lanczos matrix.
    search_prev2 = torch.zeros_like(solution)
    search_prev1 = torch.zeros_like(solution)
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T b, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    scale_prev = beta_prev

    # Terms for checking for convergence
    solution_norm = solution.norm(2, dim=dim).unsqueeze(-1)
    search_update_norm = torch.zeros_like(solution_norm)

    # Perform iterations
    flag = True
    for i in range(int(max_iter)):
        # Perform matmul
        prod = A(qvec_prev1)

        # Get next Lanczos terms
        # --> alpha_curr, beta_curr, qvec_curr
        alpha_curr = dot(prod, qvec_prev1, dim=dim)
        prod = prod - alpha_curr * zvec_prev1 - beta_prev * zvec_prev2
        qvec_curr = precon(prod)

        beta_curr = torch.abs(dot(prod, qvec_curr, dim=dim).sqrt()).clamp_min(eps)

        prod = prod / beta_curr
        qvec_curr = qvec_curr / beta_curr

        # Perform JIT-ted update
        ###########################################
        # Start givens rotation
        # Givens rotation from 2 steps ago
        subsub_diag_term = sin_prev2 * beta_prev
        sub_diag_term = cos_prev2 * beta_prev

        # Givens rotation from 1 step ago
        diag_term = alpha_curr * cos_prev1 - sin_prev1 * sub_diag_term
        sub_diag_term = sub_diag_term * cos_prev1 + sin_prev1 * alpha_curr

        # 3) Compute next Givens terms
        radius_curr = torch.sqrt(diag_term * diag_term + beta_curr * beta_curr)
        cos_curr = diag_term / radius_curr
        sin_curr = beta_curr / radius_curr
        # 4) Apply current Givens rotation
        diag_term = diag_term * cos_curr + sin_curr * beta_curr

        # Update the solution
        # --> search_curr, scale_curr solution
        # 1) Apply the latest Givens rotation to the Lanczos-b ( ||b|| e_1 )
        # This is getting the scale terms for the "search" vectors
        scale_curr = -scale_prev * sin_curr
        # 2) Get the new search vector
        search_curr = qvec_prev1 - sub_diag_term * search_prev1
        search_curr = (search_curr - subsub_diag_term * search_prev2) / diag_term

        # 3) Update the solution
        search_update = search_curr * scale_prev * cos_curr
        solution = solution + search_update
        ###########################################

        # Check convergence criterion
        search_update_norm = search_update.norm(2, dim=dim).unsqueeze(-1)
        solution_norm = solution.norm(2, dim=dim).unsqueeze(-1)
        if (search_update_norm / solution_norm).max().item() < tol:
            if verbose:
                print("MINRES converged at iteration", i)
            flag = False
            break

        # Update terms for next iteration
        # Lanczos terms
        zvec_prev2, zvec_prev1 = zvec_prev1, prod
        qvec_prev1 = qvec_curr
        beta_prev = beta_curr
        # Givens rotations terms
        cos_prev2, cos_prev1 = cos_prev1, cos_curr
        sin_prev2, sin_prev1 = sin_prev1, sin_curr
        # Search vector terms)
        search_prev2, search_prev1 = search_prev1, search_curr
        scale_prev = scale_curr

    # For b-s that are close to zero, set them to zero
    solution = solution.masked_fill(b_is_zero, 0)
    if flag and verbose:
        print(f"MINRES did not converge in {i} iterations!")
    return solution * b_norm


def gradient_descent(grad_f, x, step_size=1.0, max_iter=1e2, tol=1e-5):
    """
    Standard gradient descent algorithm`.

    :param Callable grad_f: gradient of function to bz minimized as a callable function.
    :param torch.Tensor x: input tensor.
    :param torch.Tensor, float step_size: (constant) step size of the gradient descent algorithm.
    :param int max_iter: maximum number of iterations.
    :param float tol: absolute tolerance for stopping the algorithm.
    :return: torch.Tensor :math:`x` minimizing :math:`f(x)`.

    """
    for i in range(int(max_iter)):
        x_prev = x
        x = x - grad_f(x) * step_size
        if check_conv(x_prev, x, i, thres_conv=tol):
            break
    return x


class LeastSquaresSolver(torch.autograd.Function):
    r"""
    Custom autograd function for the least squares solver to enable O(1) memory backward propagation using implicit differentiation.

    The forward pass solves the following problem using :func:`deepinv.optim.utils.least_squares`:

    .. math::

        \min_x \|A_\theta x - y \|^2 + \frac{1}{\gamma} \|x - z\|^2

    where :math:`A_\theta` is a linear operator :class:`deepinv.physics.LinearPhysics` parameterized by :math:`\theta`.

    .. note::

        This function uses a :func:`least squares <deepinv.optim.utils.least_squares>` solver under the hood, which supports various solvers such as Conjugate Gradient (CG), BiCGStab, LSQR, and MinRes (see :func:`deepinv.optim.utils.least_squares` for more details).

    The backward pass computes the gradients with respect to the inputs using implicit differentiation.
    """

    # NOTE: the physics parameters are handled as side-effects (not inputs/outputs of the autograd function)
    # we add a dummy input tensor `trigger` to trigger backward when needed (i.e. when physics parameters require grad)
    @staticmethod
    def forward(
        ctx,
        physics,
        y: Tensor,
        z: Tensor,
        init: Tensor,
        gamma: float | Tensor,
        trigger: Tensor = None,
        extra_kwargs: dict = None,
    ):

        kwargs = extra_kwargs if extra_kwargs is not None else {}

        with torch.no_grad():
            solution = least_squares(
                A=physics.A,
                AT=physics.A_adjoint,
                y=y,
                z=z,
                init=init,
                gamma=gamma,
                AAT=physics.A_A_adjoint,
                ATA=physics.A_adjoint_A,
                **kwargs,
            )

        # Save tensors only
        gamma_orig_shape = gamma.shape
        # For broadcasting with other tensors. Note we already have checked gamma shapes
        # in forward, so the following is just for gamma batched but not shaped.
        if gamma.ndim == 1:
            if isinstance(solution, TensorList):
                ndim = solution[0].ndim
            else:
                ndim = solution.ndim

            gamma = gamma.view([gamma.size(0)] + [1] * (ndim - 1))

        ctx.save_for_backward(solution, y, z, gamma)
        # Save other non-tensor contexts
        ctx.physics = physics
        ctx.kwargs = kwargs
        ctx.gamma_orig_shape = gamma_orig_shape

        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, ...]:
        h, y, z, gamma = ctx.saved_tensors
        physics = ctx.physics

        # Solve (A^T A + I/gamma) mv = grad_output
        with torch.no_grad():
            mv = least_squares(
                A=physics.A,
                AT=physics.A_adjoint,
                y=torch.zeros_like(y),
                z=grad_output * gamma,
                gamma=gamma,
                AAT=physics.A_A_adjoint,
                ATA=physics.A_adjoint_A,
                **ctx.kwargs,
            )

        # Build grads aligned with inputs: (physics, y, z, gamma, solver, max_iter, tol, ..., **kwargs)
        needs = ctx.needs_input_grad
        grads = [None] * len(needs)

        # dL/dy = A mv
        if needs[1]:
            grads[1] = physics.A(mv)

        # dL/dz = (1/gamma) mv
        if needs[2]:
            grads[2] = mv / gamma

        # dL/dgamma = <mv, h - z> / gamma^2
        if needs[4]:
            diff = h - z
            # vdot gives correct conjugation for complex; .real keeps real scalar
            grad_gamma = torch.sum(
                mv.conj() * diff, dim=list(range(1, mv.ndim)), keepdim=True
            ).real / (gamma**2)

            # If gamma was batched in the forward, we return a batched grad
            if len(ctx.gamma_orig_shape) > 0:
                grad_gamma = grad_gamma.view(ctx.gamma_orig_shape)
            # gamma was a scalar in the forward, we accumulate all grads to return a scalar, similar to
            # torch autograd behavior for broadcasted inputs
            else:
                grad_gamma = torch.sum(grad_gamma).view(())
            grads[4] = grad_gamma

        # Optional: implicit grads w.r.t physics parameters (side-effect accumulation)
        params = [
            p for p in getattr(physics, "buffers", lambda: [])() if p.requires_grad
        ]
        if params:
            # pseudo-loss = - <mv, A^T (A h - y)>
            h_det = h.detach()
            y_det = y.detach()
            with torch.enable_grad():
                pseudo = -torch.vdot(
                    mv.flatten(), physics.A_adjoint(physics.A(h_det) - y_det).flatten()
                ).real
            g_params = torch.autograd.grad(
                pseudo, params, retain_graph=False, allow_unused=True
            )
            for p, g in zip_strict(
                params,
                g_params,
            ):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.detach()
                    else:
                        p.grad = p.grad + g.detach()

        return tuple(grads)


# wrapper of the autograd function for easier use
def least_squares_implicit_backward(
    physics,
    y: Tensor,
    z: Tensor = None,
    init: Tensor = None,
    gamma: float | Tensor = None,
    **kwargs,
) -> Tensor:
    r"""
    Least squares solver with O(1) memory backward propagation using implicit differentiation.
    The function is similar to :func:`deepinv.optim.utils.least_squares` for the forward pass, but uses implicit differentiation for the backward pass, which reduces memory consumption to O(1) in the number of iterations. 

    This function supports backpropagation with respect to the inputs :math:`y`, :math:`z` and :math:`\gamma` and also with respect to the parameters of the physics operator :math:`A_\theta` if they require gradients. See :ref:`sphx_glr_auto_examples_unfolded_demo_unfolded_constant_memory.py` and the notes below for more details. 
    
    Let :math:`h(z, y, \theta, \gamma)` denote the output of the least squares solver, i.e. the solution of the following problem:
    
    .. math::

        h(z, y, \theta, \gamma) = \underset{x}{\arg\min} \; \frac{\gamma}{2}\|A_\theta x-y\|^2 + \frac{1}{2}\|x-z\|^2

    When the forward least-squares solver converges to the exact minimizer, we have the following closed-form expressions for :math:`h(z, y, \theta, \gamma)`:

    .. math::

        h(z, y, \theta, \gamma) = \left( A_\theta^{\top} A_\theta + \frac{1}{\gamma} I \right)^{-1} \left( A_\theta^{\top} y + \frac{1}{\gamma} z \right)

    Let :math:`M` denote the inverse :math:`\left( A_\theta^T A_\theta + \frac{1}{\gamma} I \right)^{-1}`. In the forward, we need to compute the vector-Jacobian products (VJPs), which can be computed as follows:

    .. math::

        \left( \frac{\partial h}{\partial z} \right)^{\top} v               &= \frac{1}{\gamma} M v \\
        \left( \frac{\partial h}{\partial y} \right)^{\top} v               &= A_\theta M v \\
        \left( \frac{\partial h}{\partial \gamma} \right)^{\top} v          &=   (h - z)^\top M  v / \gamma^2 \\
        \left( \frac{\partial h}{\partial \theta} \right)^{\top} v          &= \frac{\partial p}{\partial \theta} 
        
    where :math:`p =  (y - A_\theta h)^{\top} A_\theta M v` and :math:`\frac{\partial p}{\partial \theta}` can be computed using the standard backpropagation mechanism (autograd).

    .. note::

        This function only supports first-order gradients. Higher-order gradients are not supported. If you need higher-order gradients, please use :func:`deepinv.optim.utils.least_squares` instead but be aware that it requires storing all intermediate iterates, which can be memory-intensive.

    .. note::

        This function also supports implicit gradients with respect to the parameters of the physics operator :math:`A_\theta` if they require gradients. This is useful for learning the physics parameters in an end-to-end fashion. The gradients are accumulated in-place in the `.grad` attribute of the parameters of the physics operator. To make this work, the function takes as input the physics operator itself (not just its matmul functions) and checks if any of its parameters require gradients. If so, it triggers the backward pass accordingly.

    .. warning::

        Implicit gradients can be incorrect if the least squares solver does not converge sufficiently. Make sure to set the `max_iter` and `tol` parameters of the least squares solver appropriately to ensure convergence. You can monitor the convergence by setting `verbose=True` in the least squares solver via `kwargs`. If the solver does not converge, the implicit gradients can be very inaccurate and lead to divergence of the training.

    .. warning::

        This function does not support :class:`deepinv.utils.TensorList` inputs yet. If you use :class:`deepinv.utils.TensorList` as inputs, the function will fall back to standard least squares with full backpropagation.

    .. tip::

        If you do not need gradients with respect to the physics parameters, you can set `requires_grad=False` for all parameters of the physics operator to avoid the additional backward pass. This can save some computation time.

    .. tip::

        Training unfolded network with implicit differentiation can reduce memory consumption significantly, especially when using many iterations. On GPU, we can expect a memory reduction factor of about 2x-3x compared to standard backpropagation and a speed-up of about 1.2x-1.5x. The exact numbers depend on the problem and the number of iterations. 

    :param deepinv.physics.LinearPhysics physics: physics operator :class:`deepinv.physics.LinearPhysics`.
    :param torch.Tensor y: input tensor of shape (B, ...)
    :param torch.Tensor z: input tensor of shape (B, ...). Default is `None`, which corresponds to a zero tensor.
    :param None, torch.Tensor init: Optional initial guess, only used for the forward pass. Default is `None`, which corresponds to a zero initialization.
    :param None, float, torch.Tensor gamma: regularization parameter :math:`\gamma > 0`. Default is `None`. Can be batched (shape (B, ...)) or a scalar.
    :param kwargs: additional arguments to be passed to the least squares solver.

    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...), the solution of the least squares problem.
    """

    if z is None:
        # To get correct shape
        z = zeros_like(physics.A_adjoint(y))
    if init is None:
        init = zeros_like(z)

    # NOTE: TensorList not supported by autograd function, we fall back to standard least_squares in this case for now
    if isinstance(y, TensorList):
        warnings.warn(
            "Warning: least_squares_implicit_backward does not support TensorList inputs. Falling back to standard least_squares with full backpropagation."
        )
        return least_squares(
            A=physics.A,
            AT=physics.A_adjoint,
            y=y,
            z=z,
            init=init,
            gamma=gamma,
            AAT=physics.A_A_adjoint,
            ATA=physics.A_adjoint_A,
            **kwargs,
        )

    physics_requires_grad_params = any(
        p.requires_grad for p in getattr(physics, "buffers", lambda: [])()
    )
    # NOTE: backward is triggered if any of the inputs require grad
    # When input tensors (y, z, gamma) do not require grad and we want to do backward w.r.t physics parameters, we need to trigger it manually by a dummy tensor.
    trigger_backward = (
        y.requires_grad
        or z.requires_grad
        or (isinstance(gamma, Tensor) and gamma.requires_grad)
        or physics_requires_grad_params
    )
    if trigger_backward:
        trigger = torch.ones(1, device=y.device, dtype=y.dtype).requires_grad_(
            True
        )  # Dummy tensor to trigger backward
    else:
        trigger = torch.ones(1, device=y.device, dtype=y.dtype)
    extra_kwargs = kwargs if kwargs else None
    dtype = y.dtype if not torch.is_complex(y) else y.real.dtype
    if gamma is None:
        gamma = torch.zeros((), device=y.device, dtype=dtype)
    if isinstance(gamma, Tensor) and gamma.ndim > 0:
        if gamma.size(0) != y.size(0):
            raise ValueError(
                "If gamma is batched, its batch size must match the one of y."
            )
    if not isinstance(gamma, Tensor):
        gamma = torch.as_tensor(gamma, device=y.device, dtype=dtype)
    return LeastSquaresSolver.apply(physics, y, z, init, gamma, trigger, extra_kwargs)


class GaussianMixtureModel(nn.Module):
    r"""
    Gaussian mixture model including parameter estimation.

    Implements a Gaussian Mixture Model, its negative log likelihood function and an EM algorithm
    for parameter estimation.

    :param int n_components: number of components of the GMM
    :param int dimension: data dimension
    :param str device: gpu or cpu.
    """

    def __init__(self, n_components, dimension, device="cpu"):
        super(GaussianMixtureModel, self).__init__()
        self._covariance_regularization = None
        self.n_components = n_components
        self.dimension = dimension
        self._weights = nn.Parameter(
            torch.ones((n_components,), device=device), requires_grad=False
        )
        self.set_weights(self._weights)
        self.mu = nn.Parameter(
            torch.zeros((n_components, dimension), device=device), requires_grad=False
        )
        self._cov = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._cov_inv = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._cov_inv_reg = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._cov_reg = nn.Parameter(
            0.1
            * torch.eye(dimension, device=device)[None, :, :].tile(n_components, 1, 1),
            requires_grad=False,
        )
        self._logdet_cov = nn.Parameter(self._weights.clone(), requires_grad=False)
        self._logdet_cov_reg = nn.Parameter(self._weights.clone(), requires_grad=False)
        self.set_cov(self._cov)

    def set_cov(self, cov):
        r"""
        Sets the covariance parameters to cov and maintains their log-determinants and inverses

        :param torch.Tensor cov: new covariance matrices in a n_components x dimension x dimension tensor
        """
        self._cov.data = cov.detach().to(self._cov)
        self._logdet_cov.data = torch.logdet(self._cov).detach().clone()
        self._cov_inv.data = torch.linalg.inv(self._cov).detach().clone()
        if self._covariance_regularization:
            self._cov_reg.data = (
                self._cov.detach().clone()
                + self._covariance_regularization
                * torch.eye(self.dimension, device=self._cov.device)[None, :, :].tile(
                    self.n_components, 1, 1
                )
            )
            self._logdet_cov_reg.data = torch.logdet(self._cov_reg).detach().clone()
            self._cov_inv_reg.data = torch.linalg.inv(self._cov_reg).detach().clone()

    def set_cov_reg(self, reg):
        r"""
        Sets covariance regularization parameter for evaluating
        Needed for EPLL.

        :param float reg: covariance regularization parameter
        """
        self._covariance_regularization = reg
        self._cov_reg.data = (
            self._cov.detach().clone()
            + self._covariance_regularization
            * torch.eye(self.dimension, device=self._cov.device)[None, :, :].tile(
                self.n_components, 1, 1
            )
        )
        self._logdet_cov_reg.data = torch.logdet(self._cov_reg).detach().clone()
        self._cov_inv_reg.data = torch.linalg.inv(self._cov_reg).detach().clone()

    def get_cov(self):
        r"""
        get method for covariances
        """
        return self._cov.clone()

    def get_cov_inv_reg(self):
        r"""
        get method for covariances
        """
        return self._cov_inv_reg.clone()

    def set_weights(self, weights):
        r"""
        sets weight parameter while ensuring non-negativity and summation to one

        :param torch.Tensor weights: non-zero weight tensor of size n_components with non-negative entries
        """
        assert torch.min(weights) >= 0.0
        assert torch.sum(weights) > 0.0
        self._weights.data = (weights / torch.sum(weights)).detach().to(self._weights)

    def get_weights(self):
        r"""
        get method for weights
        """
        return self._weights.clone()

    def load_state_dict(self, *args, **kwargs):
        r"""
        Override load_state_dict to maintain internal parameters.
        """
        super().load_state_dict(*args, **kwargs)
        self.set_cov(self._cov)
        self.set_weights(self._weights)

    def component_log_likelihoods(self, x, cov_regularization=False):
        r"""
        returns a tensor containing the log likelihood values of x for each component

        :param torch.Tensor x: input data of shape batch_dimension x dimension
        :param bool cov_regularization: whether using regularized covariance matrices
        """
        if cov_regularization:
            cov_inv = self._cov_inv_reg
            logdet_cov = self._logdet_cov_reg
        else:
            cov_inv = self._cov_inv
            logdet_cov = self._logdet_cov
        centered_x = x[None, :, :] - self.mu[:, None, :]
        exponent = torch.sum(torch.bmm(centered_x, cov_inv) * centered_x, 2)
        component_log_likelihoods = (
            -0.5 * logdet_cov[:, None]
            - 0.5 * exponent
            - 0.5 * self.dimension * torch.log(torch.tensor(2 * torch.pi).to(x))
        )
        return component_log_likelihoods.T

    def forward(self, x):
        r"""
        evaluate negative log likelihood function

        :param torch.Tensor x: input data of shape batch_dimension x dimension
        """
        component_log_likelihoods = self.component_log_likelihoods(x)
        component_log_likelihoods = component_log_likelihoods + torch.log(
            self._weights[None, :]
        )
        log_likelihoods = torch.logsumexp(component_log_likelihoods, -1)
        return -log_likelihoods

    def classify(self, x, cov_regularization=False):
        """
        returns the index of the most likely component

        :param torch.Tensor x: input data of shape batch_dimension x dimension
        :param bool cov_regularization: whether using regularized covariance matrices
        """
        component_log_likelihoods = self.component_log_likelihoods(
            x, cov_regularization=cov_regularization
        )
        component_log_likelihoods = component_log_likelihoods + torch.log(
            self._weights[None, :]
        )
        val, ind = torch.max(component_log_likelihoods, 1)
        return ind

    def fit(
        self,
        dataloader,
        max_iters=100,
        stopping_criterion=None,
        data_init=True,
        cov_regularization=1e-5,
        verbose=False,
    ):
        """
        Batched Expectation Maximization algorithm for parameter estimation.


        :param torch.utils.data.DataLoader dataloader: containing the data
        :param int max_iters: maximum number of iterations
        :param float stopping_criterion: stop when objective decrease is smaller than this number.
            None for performing exactly max_iters iterations
        :param bool data_init: True for initialize mu by the first data points, False for using current values as initialization
        :param bool verbose: Output progress information in the console
        """
        if data_init:
            first_data = next(iter(dataloader))

            if isinstance(first_data, (tuple, list)):
                first_data = first_data[0]

            first_data = first_data[: self.n_components].to(self.mu)

            if first_data.shape[0] == self.n_components:
                self.mu.copy_(first_data)
            else:
                # if the first batch does not contain enough data points, fill up the others randomly...
                self.mu.data[: first_data.shape[0]] = first_data
                self.mu.data[first_data.shape[0] :] = torch.randn_like(
                    self.mu[first_data.shape[0] :]
                ) * torch.std(first_data, 0, keepdim=True) + torch.mean(
                    first_data, 0, keepdim=True
                )

        objective = 1e100
        for step in (progress_bar := tqdm(range(max_iters), disable=not verbose)):
            weights_new, mu_new, cov_new, objective_new = self._EM_step(
                dataloader, verbose
            )
            # stopping criterion
            self.set_weights(weights_new)
            self.mu.data = mu_new
            cov_new_reg = cov_new + cov_regularization * torch.eye(self.dimension)[
                None, :, :
            ].tile(self.n_components, 1, 1).to(cov_new)
            self.set_cov(cov_new_reg)
            if stopping_criterion:
                if objective - objective_new < stopping_criterion:
                    return
            objective = objective_new
            progress_bar.set_description(
                "Step {}, Objective {:.4f}".format(step + 1, objective.item())
            )

    def _EM_step(self, dataloader, verbose):
        """
        one step of the EM algorithm

        :param torch.data.Dataloader dataloader: containing the data
        :param bool verbose: Output progress information in the console
        """
        objective = 0
        weights_new = torch.zeros_like(self._weights)
        mu_new = torch.zeros_like(self.mu)
        C_new = torch.zeros_like(self._cov)
        n = 0
        objective = 0
        for x in tqdm(dataloader, disable=not verbose):
            x = x.to(self.mu)
            n += x.shape[0]
            component_log_likelihoods = self.component_log_likelihoods(x)
            log_betas = component_log_likelihoods + torch.log(self._weights[None, :])
            log_beta_sum = torch.logsumexp(log_betas, -1)
            log_betas = log_betas - log_beta_sum[:, None]
            objective -= torch.sum(log_beta_sum)
            betas = torch.exp(log_betas)
            weights_new += torch.sum(betas, 0)
            beta_times_x = x[None, :, :] * betas.T[:, :, None]
            mu_new += torch.sum(beta_times_x, 1)
            C_new += torch.bmm(
                beta_times_x.transpose(1, 2),
                x[None, :, :].tile(self.n_components, 1, 1),
            )

        # prevents division by zero if weights_new is zero
        weights_new = torch.maximum(weights_new, torch.tensor(1e-5).to(weights_new))

        mu_new = mu_new / weights_new[:, None]
        cov_new = C_new / weights_new[:, None, None] - torch.matmul(
            mu_new[:, :, None], mu_new[:, None, :]
        )
        weights_new = weights_new / n
        objective = objective / n
        return weights_new, mu_new, cov_new, objective
