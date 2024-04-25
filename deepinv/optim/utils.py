from deepinv.utils import zeros_like
import torch
from tqdm import tqdm
import torch.nn as nn
from typing import Callable
from deepinv.utils import TensorList


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


def conjugate_gradient(
    A: Callable,
    b: torch.Tensor,
    max_iter: float = 1e2,
    tol: float = 1e-5,
    eps: float = 1e-8,
):
    """
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param (callable) A: Linear operator as a callable function, has to be square!
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param int max_iter: maximum number of CG iterations
    :param float tol: absolute tolerance for stopping the CG algorithm.
    :param float eps: a small value for numerical stability
    :return: torch.Tensor :math:`x` of shape (B, ...) verifying :math:`Ax=b`.

    """

    x = zeros_like(b)

    def dot(a, b):
        ndim = a[0].ndim if isinstance(a, TensorList) else a.ndim
        dot = (a.conj() * b).sum(
            dim=tuple(range(1, ndim)), keepdim=False
        )  # performs batched dot product
        if isinstance(dot, TensorList):
            aux = 0
            for d in dot:
                aux += d
            dot = aux
        return dot

    r = b - A(x)
    p = r
    rsold = dot(r, r)

    for _ in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / (dot(p, Ap) + eps)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = dot(r, r)
        if all(rsnew.abs() < tol**2):
            break
        p = r + p * (rsnew / (rsold + eps))
        rsold = rsnew

    return x


def gradient_descent(grad_f, x, step_size=1.0, max_iter=1e2, tol=1e-5):
    """
    Standard gradient descent algorithm`.

    :param callable grad_f: gradient of function to bz minimized as a callable function.
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
