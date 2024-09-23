from abc import abstractmethod
from typing import Callable, Iterable

import torch
from torch import nn

from deepinv.optim import BaseOptim
from deepinv.physics import LinearPhysics


class ChampagneIterator(nn.Module):
    r"""
    Iterator for the Champagne Bayesian framework.
    
    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        \Sigma_{x_k} &= \Gamma_k - \Gamma_k A^T (\Sigma_y)^{-1} A \Gamma_k \\
        \Sigma_{y, k} &= \lambda^2 I + A \Gamma_k A^T \\
        x_{k} &= \Gamma_k A^T (\Sigma_{y,k})^{-1} y \\
        \Gamma_{k+1} &= update(\Gamma_k, x_k, y,\Sigma_{y,k}) \text{, specified per subclass}
        \end{aligned}
        \end{equation*}
    """

    def __init__(self, F_fn: Callable | None = None):
        super().__init__()
        self.F_fn = F_fn
        self.has_cost = self.F_fn is not None

    def forward(self, X_prev, cur_data_fidelity, cur_prior, cur_params, y, A):
        x, gamma = X_prev["est"]
        nb_sensors = y.size(0)

        g_times_gamma = A * gamma  # A @ Γ
        # Measure covariance matrix
        sigma_y = (
            cur_params["lambda"] * torch.eye(nb_sensors, dtype=A.dtype, device=A.device)
            + g_times_gamma @ A.T
        )
        sigma_y_inv = torch.linalg.inv(sigma_y)

        # Source expectation
        x = g_times_gamma.T @ sigma_y_inv @ y

        # Source covariance matrix
        sigma_x = -g_times_gamma.T @ sigma_y_inv @ g_times_gamma
        idx = torch.arange(sigma_x.shape[0])
        sigma_x[idx, idx] += gamma

        # Update gamma
        gamma = self.update(gamma, x, A, sigma_x, sigma_y_inv)

        # FIXME : we're returning x_k and gamma_{k+1}
        return {"est": (x, gamma)}

    @abstractmethod
    def update(
        self,
        gamma: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y_inv: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Get the updated source variances \gamma_{k+1} from \gamma_k.

        :param torch.Tensor gamma: current estimate for the variances of all sources.
        :param torch.Tensor x: current estimate for the expectation of all sources at each timestep.
        :param torch.Tensor A: the forward operator.
        :param torch.Tensor sigma_x: current estimate for the source covariance matrix.
        :param torch.Tensor sigma_y_inv: current estimate for the inverse of the sensor covariance matrix.
        :return: the update estimate for the source variances.
        """
        pass


class Champagne(BaseOptim):
    r"""
    Class to run the Champagne algorithm, notably used for source localization in MEG/EEG.

    It belongs to the class of Sparse Bayesian Learning algorithms, and jointly estimates source activations positions
    and their variances, while encouraging sparsity.

    :param ChampagneIterator iterator: Fixed-point Champagne iterator of the optimization algorithm of interest.
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: 100.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration).
                            Default: `{"lambda": 1.0}`. See :any:`optim-params` for more details.
    :return: a torch model that solves the optimization problem.
    """

    def __init__(
        self,
        iterator: ChampagneIterator,
        params_algo: dict[str, float | Iterable[float]] = {"lambda": 1.0},
        max_iter: int = 100,
        **kwargs,
    ):
        def default_custom_init(y, A):
            x_init = A.T @ y
            time_steps = y.size(1)
            gamma_init = (1 / time_steps) * torch.diag(x_init @ x_init.T)
            return {"est": (x_init, gamma_init)}

        if "custom_init" not in kwargs:
            kwargs["custom_init"] = default_custom_init

        super().__init__(
            iterator=iterator,
            params_algo=params_algo,
            max_iter=max_iter,
            anderson_acceleration=False,
            backtracking=False,
            **kwargs,
        )

    def forward(
        self,
        y: torch.Tensor,
        physics: LinearPhysics,
        x_gt: torch.Tensor | None = None,
        compute_metrics: bool = False,
    ):
        r"""
        Runs the fixed-point iteration algorithm for solving :ref:`(1) <optim>`.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics.LinearPhysics physics: physics of the problem for the acquisition of ``y``.
                                                     The physics must be linear, as the code will attempt to
                                                     extract A such that physics.A(x) = A @ x.
        :param torch.Tensor x_gt: (optional) ground truth data, for plotting the PSNR across optim iterations.
        :param bool compute_metrics: whether to compute the metrics or not. Default: ``False``.
        :return: If ``compute_metrics`` is ``False``,  returns (torch.Tensor) the output of the algorithm.
                Else, returns (torch.Tensor, dict) the output of the algorithm and the metrics.
        """
        nb_sources = physics.A_adjoint(y).size(0)
        A = physics.A(torch.eye(nb_sources, dtype=y.dtype, device=y.device))
        X, metrics = self.fixed_point(y, A, x_gt=x_gt, compute_metrics=compute_metrics)
        x = self.get_output(X)
        if compute_metrics:
            return x, metrics
        else:
            return x


class EMChampagneIterator(ChampagneIterator):
    r"""
    Iterator for the Expectation-Maximization update in Champagne.

    .. math::
        \begin{equation*}
        \begin{aligned}
         \Gamma_{k+1} &= diag(\gamma_{k+1}) \\
        (\gamma_{k+1})_{n} &= [\Sigma_{x_k}]_{n,n} + \dfrac{1}{T}\sum_{t=1}^{T} (x_k)_{n,t}^2
        \end{aligned}
        \end{equation*}
    """

    def update(
        self,
        gamma: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y_inv: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Get the updated source variances \gamma_{k+1} from \gamma_k.

        :param torch.Tensor gamma: current estimate for the variances of all sources.
        :param torch.Tensor x: current estimate for the expectation of all sources at each timestep.
        :param torch.Tensor A: the forward operator.
        :param torch.Tensor sigma_x: current estimate for the source covariance matrix.
        :param torch.Tensor sigma_y_inv: current estimate for the inverse of the sensor covariance matrix.
        :return: the update estimate for the source variances.
        """
        time_steps = x.size(1)
        return torch.diag(sigma_x) + (x**2).sum(1) / time_steps


class ConvexChampagneIterator(ChampagneIterator):
    r"""
    Iterator for the Convex-Bounding update in Champagne.

    .. math::
        \begin{equation*}
        \begin{aligned}
         \Gamma_{k+1} &= diag(\gamma_{k+1}) \\
        (\gamma_{k+1})_{n} &= \sqrt{\left[\dfrac{1}{T}\sum_{t=1}^{T} (x_k)_{n,t}^2 \right]} \left(A^T_n \Sigma_{y,k} ^{-1} A_n \right)^{-\frac{1}{2}}
        \end{aligned}
        \end{equation*}
    """

    def update(
        self,
        gamma: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y_inv: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Get the updated source variances \gamma_{k+1} from \gamma_k.

        :param torch.Tensor gamma: current estimate for the variances of all sources.
        :param torch.Tensor x: current estimate for the expectation of all sources at each timestep.
        :param torch.Tensor A: the forward operator.
        :param torch.Tensor sigma_x: current estimate for the source covariance matrix.
        :param torch.Tensor sigma_y_inv: current estimate for the inverse of the sensor covariance matrix.
        :return: the update estimate for the source variances.
        """
        time_steps = x.size(1)
        return torch.sqrt(
            (1 / time_steps) * (x**2).sum(1) / (A.T @ sigma_y_inv * A.T).sum(1)
        )


class MacKayChampagneIterator(ChampagneIterator):
    r"""
    Iterator for the MacKay update in Champagne.

    .. math::
        \begin{equation*}
        \begin{aligned}
         \Gamma_{k+1} &= diag(\gamma_{k+1}) \\
        (\gamma_{k+1})_{n} &= \left[\dfrac{1}{T}\sum_{t=1}^{T} (x_k)_{n,t}^2 \right] \left((\gamma_k)_n A^T_n \Sigma_{y,k} ^{-1} A_n \right)^{-1}
        \end{aligned}
        \end{equation*}
    """

    def update(
        self,
        gamma: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y_inv: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Get the updated source variances \gamma_{k+1} from \gamma_k.

        :param torch.Tensor gamma: current estimate for the variances of all sources.
        :param torch.Tensor x: current estimate for the expectation of all sources at each timestep.
        :param torch.Tensor A: the forward operator.
        :param torch.Tensor sigma_x: current estimate for the source covariance matrix.
        :param torch.Tensor sigma_y_inv: current estimate for the inverse of the sensor covariance matrix.
        :return: the update estimate for the source variances.
        """
        time_steps = x.size(1)
        return (
            (1 / time_steps)
            * (x**2).sum(1)
            / (gamma * (A.T @ sigma_y_inv * A.T).sum(1))
        )


class LowSNRChampagneIterator(ChampagneIterator):
    r"""
    Iterator for the LowSNR-BSI update in Champagne.

    .. math::
        \begin{equation*}
        \begin{aligned}
         \Gamma_{k+1} &= diag(\gamma_{k+1}) \\
        (\gamma_{k+1})_{n} &= \sqrt{\left[\dfrac{1}{T}\sum_{t=1}^{T} (x_k)_{n,t}^2 \right]} \left(A^T_n A_n \right)^{-\frac{1}{2}}
        \end{aligned}
        \end{equation*}
    """

    def update(
        self,
        gamma: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y_inv: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Get the updated source variances \gamma_{k+1} from \gamma_k.

        :param torch.Tensor gamma: current estimate for the variances of all sources.
        :param torch.Tensor x: current estimate for the expectation of all sources at each timestep.
        :param torch.Tensor A: the forward operator.
        :param torch.Tensor sigma_x: current estimate for the source covariance matrix.
        :param torch.Tensor sigma_y_inv: current estimate for the inverse of the sensor covariance matrix.
        :return: the update estimate for the source variances.
        """
        time_steps = x.size(1)
        return torch.sqrt((1 / time_steps) * (x**2).sum(1) / (A**2).sum(0))
