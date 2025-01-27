from deepinv.utils import zeros_like
import torch
from tqdm import tqdm
import torch.nn as nn
from typing import Callable
from deepinv.utils.tensorlist import TensorList


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
    A,
    AT,
    y,
    z=0.0,
    gamma=None,
    AAT=None,
    ATA=None,
    solver="CG",
    max_iter=100,
    tol=1e-6,
    **kwargs,
):
    r"""
    Solves :math:`\min_x \|Ax-y\|^2 + \frac{1}{\gamma}\|x-z\|^2` using the specified solver.

    The solvers are stopped either when :math:`\|Ax-y\| \leq \text{tol} \times \|y\|` or
    when the maximum number of iterations is reached.

    Available solvers are:

    - `'CG'`: `Conjugate Gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.
    - `'BiCGStab'`: `Biconjugate Gradient Stabilized method <https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method>`_
    - `'lsqr'`: `Least Squares QR <https://www-leland.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf>`_

    .. note::

        Both `'CG'` and `'BiCGStab'` are used for squared linear systems, while `'lsqr'` is used for rectangular systems.

        If the chosen solver requires a squared system, we map to the problem to the normal equations:
        If the size of :math:`y` is larger than :math:`x` (overcomplete problem), it computes :math:`(A^{\top} A)^{-1} A^{\top} y`,
        otherwise (incomplete problem) it computes :math:`A^{\top} (A A^{\top})^{-1} y`.


    :param Callable A: Linear operator :math:`A` as a callable function.
    :param Callable AT: Adjoint operator :math:`A^{\top}` as a callable function.
    :param torch.Tensor y: input tensor of shape (B, ...)
    :param torch.Tensor z: input tensor of shape (B, ...) or scalar.
    :param None, float gamma: (Optional) inverse regularization parameter.
    :param str solver: solver to be used.
    :param Callable AAT: (Optional) Efficient implementation of :math:`A(A^{\top}(x))`. If not provided, it is computed as :math:`A(A^{\top}(x))`.
    :param Callable ATA: (Optional) Efficient implementation of :math:`A^{\top}(A(x))`. If not provided, it is computed as :math:`A^{\top}(A(x))`.
    :param int max_iter: maximum number of iterations.
    :param float tol: relative tolerance for stopping the algorithm.
    :param kwargs: Keyword arguments to be passed to the solver.
    :return: (class:`torch.Tensor`) :math:`x` of shape (B, ...).
    """

    if solver == "lsqr":  # rectangular solver

        if gamma is not None:
            eta = 1 / gamma
        else:
            eta = 0

        x = lsqr(A, AT, y, x0=z, eta=eta, max_iter=max_iter, tol=tol, **kwargs)

    else:  # squared solvers

        if AAT is None:
            AAT = lambda x: A(AT(x))
        if ATA is None:
            ATA = lambda x: AT(A(x))

        Aty = AT(y)

        overcomplete = False
        if gamma is not None:
            b = AT(y) + 1 / gamma * z
            H = lambda x: ATA(x) + 1 / gamma * x
        else:
            overcomplete = Aty.flatten().shape[0] < y.flatten().shape[0]
            if not overcomplete:
                H = lambda x: AAT(x)
                b = y
            else:
                H = lambda x: ATA(x)
                b = Aty
        if solver == "CG":
            x = conjugate_gradient(A=H, b=b, max_iter=max_iter, tol=tol, **kwargs)
        elif solver == "BiCGStab":
            x = bicgstab(A=H, b=b, max_iter=max_iter, tol=tol, **kwargs)
        else:
            raise ValueError(
                f"Solver {solver} not recognized. Choose between 'CG', 'lsqr' and 'BiCGStab'."
            )

        if gamma is None and not overcomplete:
            x = AT(x)

    return x


def dot(a, b):
    ndim = a[0].ndim if isinstance(a, TensorList) else a.ndim
    dot = (a.conj() * b).sum(
        dim=tuple(range(1, ndim)), keepdim=True
    )  # performs batched dot product
    if isinstance(dot, TensorList):
        aux = 0
        for d in dot:
            aux += d
        dot = aux
    return dot


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    max_iter: float = 1e2,
    tol: float = 1e-5,
    eps: float = 1e-8,
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
    :return: torch.Tensor :math:`x` of shape (B, ...) verifying :math:`Ax=b`.

    """

    x = zeros_like(b)

    r = b - A(x)
    p = r
    rsold = dot(r, r)

    bnorm = dot(b, b).real

    tol = tol**2
    flag = False
    for _ in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / (dot(p, Ap) + eps)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = dot(r, r)
        assert rsnew.isfinite().all(), "Conjugate gradient diverged"
        if all(rsnew.abs() < tol * bnorm):
            flag = True
            if verbose:
                print(f"Conjugate gradient converged after {_} iterations")
            break
        p = r + p * (rsnew / (rsold + eps))
        rsold = rsnew

    if not flag and verbose:
        print("Conjugate gradient did not converge")

    return x


def bicgstab(
    A,
    b,
    init=None,
    max_iter=1e2,
    tol=1e-5,
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
    :param bool verbose: Output progress information in the console.
    :param Callable left_precon: left preconditioner as a callable function.
    :param Callable right_precon: right preconditioner as a callable function.
    :return: (class:`torch.Tensor`) :math:`x` of shape (B, ...)
    """
    if init is not None:
        x = init
    else:
        x = torch.zeros_like(b)

    r = b - A(x)
    r_hat = r.clone()  # torch.rand_like(r) # torch.ones_like(r)
    rho = dot(r, r_hat)
    p = r
    max_iter = int(max_iter)

    bnorm = dot(b, b).real

    tol = tol**2
    flag = False
    for i in range(max_iter):
        y = right_precon(left_precon(p))
        v = A(y)
        alpha = rho / dot(r_hat, v)
        h = x + alpha * y
        s = r - alpha * v
        z = right_precon(left_precon(s))
        t = A(z)
        omega = dot(left_precon(t), left_precon(s)) / dot(
            left_precon(t), left_precon(t)
        )

        x = h + omega * z
        r = s - omega * t
        if dot(r, r).real < tol * bnorm:
            flag = True
            if verbose:
                print("BiCGSTAB Converged at iteration", i)
            break

        rho_new = dot(r, r_hat)
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        rho = rho_new

    if not flag and verbose:
        print("BiCGSTAB did not converge")

    return x


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    adapted from https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py

    The routine '_sym_ortho' was added for numerical stability. This is
    recommended by S.-C. Choi in "Iterative Methods for Singular Linear Equations and Least-Squares
    Problems".  It removes the unpleasant potential of
    ``1/eps`` in some important places.

    """
    if b == 0:
        return torch.sign(a), 0, a.abs()
    elif a == 0:
        return 0, torch.sign(b), b.abs()
    elif b.abs() > a.abs():
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
    atol=0.,
    tol=1e-6,
    conlim=1e8,
    max_iter=100,
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
    :param float eta: damping parameter :math:`eta \geq 0`.
    :param None, torch.Tensor x0: Optional :math:`x_0`, which is also used as the initial guess.
    :param float atol: absolute tolerance for stopping the LSQR algorithm.
    :param float tol: relative tolerance for stopping the LSQR algorithm.
    :param float conlim: maximum value of the condition number of the system.
    :param int max_iter: maximum number of LSQR iterations.
    :param bool verbose: Output progress information in the console.
    """

    xt = AT(b)
    # m = b.size(0)
    # n = xt.numel()/xt.size(0)

    itn = 0
    istop = 0
    # ctol = 1 / conlim if conlim > 0 else 0
    anorm = 0.0
    # acond = 0.0
    if isinstance(eta, float) or isinstance(eta, int):
        eta = torch.tensor(eta, device=b.device)
    dampsq = eta
    damp = torch.sqrt(eta)
    ddnorm = 0.0
    # res2 = 0.0
    # xnorm = 0.0
    xxnorm = 0.0
    z = 0.0
    cs2 = -1.0
    sn2 = 0.0

    u = b.clone()
    bnorm = torch.norm(b)

    if x0 is None:
        x = torch.zeros_like(xt)
        beta = bnorm
    else:
        if isinstance(x0, float):
            x = x0 * torch.zeros_like(xt)
        else:
            x = x0.clone()

        u -= A(x)
        beta = torch.norm(u)

    if beta > 0:
        u /= beta
        v = AT(u)
        alfa = torch.norm(v)
    else:
        v = torch.zeros_like(x)
        alfa = 0.0

    if alfa > 0:
        v /= alfa

    w = v.clone()
    rhobar = alfa
    phibar = beta
    arnorm = alfa * beta

    if arnorm == 0:
        return x

    flag = False
    while itn < max_iter:
        itn += 1
        u = A(v) - alfa * u
        beta = torch.norm(u)

        if beta > 0:
            u /= beta
            anorm = torch.sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
            v = AT(u) - beta * v
            alfa = torch.norm(v)
            if alfa > 0:
                v /= alfa

        if eta > 0:
            rhobar1 = torch.sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.0

        cs, sn, rho = _sym_ortho(rhobar1, beta)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x += t1 * w
        w = v + t2 * w
        ddnorm += torch.norm(dk) ** 2

        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = torch.sqrt(xxnorm + zbar**2)
        gamma = torch.sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm += z**2

        acond = anorm * torch.sqrt(ddnorm)
        rnorm = torch.sqrt(phibar**2 + psi**2)
        # arnorm = alfa * abs(tau)

        if rnorm <= tol * bnorm + atol * anorm * xnorm:
            istop = 1
        elif acond >= conlim:
            istop = 4
        elif itn >= max_iter:
            istop = 7

        if istop > 0:
            flag = True
            if verbose:
                print("LSQR converged at iteration", itn)
            break

    if not flag and verbose:
        print("LSQR did not converge")

    return x


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
            first_data = next(iter(dataloader))[0][: self.n_components].to(self.mu)
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
            self.set_weights = weights_new
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
        for x, _ in tqdm(dataloader, disable=not verbose):
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
