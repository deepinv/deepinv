import time as time
from typing import Callable, Union
from torch import Tensor
from warnings import warn

import deepinv.optim
from deepinv.sampling import BaseSampling
from deepinv.optim import ScorePrior
from deepinv.physics import Physics
from deepinv.sampling.sampling_iterators import ULAIterator, SKRockIterator


class ULA(BaseSampling):
    r"""
    Projected Plug-and-Play Unadjusted Langevin Algorithm.

    The algorithm runs the following markov chain iteration (Algorithm 2 from :footcite:t:`laumont2022bayesian`):

    .. math::

        x_{k+1} = \Pi_{[a,b]} \left(x_{k} + \eta \nabla \log p(y|A,x_k) +
        \eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1} \right).

    where :math:`x_{k}` is the :math:`k` th sample of the Markov chain,
    :math:`\log p(y|x)` is the log-likelihood function, :math:`\log p(x)` is the log-prior,
    :math:`\eta>0` is the step size, :math:`\alpha>0` controls the amount of regularization,
    :math:`\Pi_{[a,b]}(x)` projects the entries of :math:`x` to the interval :math:`[a,b]` and
    :math:`z\sim \mathcal{N}(0,I)` is a standard Gaussian vector.


    - Projected PnP-ULA assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, ULA required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`

    .. warning::
        This a legacy class provided for convenience. See the example in :ref:`mcmc` for details on how to build a ULA sampler.

    :param deepinv.optim.ScorePrior, torch.nn.Module prior: negative log-prior based on a trained or model-based denoiser.
    :param deepinv.optim.DataFidelity, torch.nn.Module data_fidelity: negative log-likelihood function linked with the
        noise distribution in the acquisition physics.
    :param float step_size: step size :math:`\eta>0` of the algorithm.
        Tip: use :func:`deepinv.physics.LinearPhysics.compute_norm` to compute the Lipschitz constant of a linear forward operator.
    :param float sigma: noise level used in the plug-and-play prior denoiser. A larger value of sigma will result in
        a more regularized reconstruction.
    :param float alpha: regularization parameter :math:`\alpha`
    :param int max_iter: number of Monte Carlo iterations.
    :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
        samples to compute posterior statistics).
    :param float burnin_ratio: percentage of iterations used for burn-in period, should be set between 0 and 1.
        The burn-in samples are discarded constant with a numerical algorithm.
    :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
        If ``None``, the algorithm will not project the samples.
    :param float crit_conv: Threshold for verifying the convergence of the mean and variance estimates.
    :param bool verbose: prints progress of the algorithm.



    """

    def __init__(
        self,
        prior: ScorePrior,
        data_fidelity,
        step_size: float = 1.0,
        sigma: float = 0.05,
        alpha: float = 1.0,
        max_iter: int = 1e3,
        thinning: int = 5,
        burnin_ratio: float = 0.2,
        clip: tuple = (-1.0, 2.0),
        thresh_conv: float = 1e-3,
        save_chain: bool = False,
        verbose: bool = False,
    ):
        algo_params = {"step_size": step_size, "alpha": alpha, "sigma": sigma}
        iterator = ULAIterator(algo_params, clip=clip)
        super().__init__(
            iterator,
            data_fidelity,
            prior,
            max_iter=max_iter,
            thresh_conv=thresh_conv,
            burnin_ratio=burnin_ratio,
            thinning=thinning,
            history_size=save_chain,
            verbose=verbose,
        )

    def forward(
        self,
        y: Tensor,
        physics: Physics,
        seed: Union[None, int] = None,
        x_init: Union[None, Tensor] = None,
        g_statistics: Union[list[Callable], Callable] = lambda d: d["x"],
    ):
        r"""
        Runs the chain to obtain the posterior mean and variance of the reconstruction of the measurements y.

        :param torch.Tensor y: Measurements
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param float seed: Random seed for generating the Monte Carlo samples
        :param list[Callable] | Callable g_statistics: List of functions for which to compute posterior statistics, or a single function.
            The sampler will compute the posterior mean and variance of each function in the list. Note the sampler outputs a dictionary so they must act on `d["x"]`.
            Default: ``lambda d: d["x"]`` (identity function)
        :return: (tuple of torch.tensor) containing the posterior mean and variance.
        """
        warn(
            "Deprecated ULA.forward returns tuple (mean, var). This will return only mean in a future version in line with BaseSampling.forward. Use deepinv.sampling.sampling_builder instead to build an ULA sampler",
            DeprecationWarning,
        )
        return self.sample(
            y, physics, x_init=x_init, seed=seed, g_statistics=g_statistics
        )


class SKRock(BaseSampling):
    r"""
    Plug-and-Play SKROCK algorithm.

    Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
    approximation to accelerate the standard Unadjusted Langevin Algorithm.

    The algorithm was introduced by :footcite:t:`pereyra2020accelerating`.

    - SKROCK assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, SKROCK required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`

    .. warning::
        This a legacy class provided for convenience. See the example in :ref:`mcmc` for details on how to build a SKRock sampler.

    :param deepinv.optim.ScorePrior, torch.nn.Module prior: negative log-prior based on a trained or model-based denoiser.
    :param deepinv.optim.DataFidelity, torch.nn.Module data_fidelity: negative log-likelihood function linked with the
        noise distribution in the acquisition physics.
    :param float step_size: Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
    :param float eta: :math:`\eta` SKROCK damping parameter.
    :param float alpha: regularization parameter :math:`\alpha`.
    :param int inner_iter: Number of inner SKROCK iterations.
    :param int max_iter: Number of outer iterations.
    :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
        samples to compute posterior statistics).
    :param float burnin_ratio: percentage of iterations used for burn-in period. The burn-in samples are discarded
        constant with a numerical algorithm.
    :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
        If ``None``, the algorithm will not project the samples.
    :param bool verbose: prints progress of the algorithm.
    :param float sigma: noise level used in the plug-and-play prior denoiser. A larger value of sigma will result in
        a more regularized reconstruction.



    """

    def __init__(
        self,
        prior: deepinv.optim.ScorePrior,
        data_fidelity,
        step_size: float = 1.0,
        inner_iter: int = 10,
        eta: float = 0.05,
        alpha: float = 1.0,
        max_iter: int = 1e3,
        burnin_ratio: float = 0.2,
        thinning: int = 10,
        clip: tuple[float, float] = (-1.0, 2.0),
        thresh_conv: float = 1e-3,
        save_chain: bool = False,
        verbose: bool = False,
        sigma: float = 0.05,
    ) -> None:
        algo_params = {
            "step_size": step_size,
            "alpha": alpha,
            "sigma": sigma,
            "eta": eta,
            "inner_iter": inner_iter,
        }
        iterator = SKRockIterator(algo_params=algo_params, clip=clip)
        super().__init__(
            iterator,
            data_fidelity,
            prior,
            max_iter=max_iter,
            thresh_conv=thresh_conv,
            burnin_ratio=burnin_ratio,
            thinning=thinning,
            history_size=save_chain,
            verbose=verbose,
        )

    def forward(
        self,
        y: Tensor,
        physics: Physics,
        seed: Union[None, int] = None,
        x_init: Union[None, Tensor] = None,
        g_statistics: Union[list[Callable], Callable] = lambda d: d["x"],
    ) -> tuple[Tensor, Tensor]:
        r"""
        Runs the chain to obtain the posterior mean and variance of the reconstruction of the measurements y.

        :param torch.Tensor y: Measurements
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param float seed: Random seed for generating the Monte Carlo samples
        :param List[Callable] | Callable g_statistics: List of functions for which to compute posterior statistics, or a single function.
            The sampler will compute the posterior mean and variance of each function in the list. Note the sampler outputs a dictionary so they must act on `d["x"]`.
            Default: ``lambda d: d["x"]`` (identity function)
        :return: (tuple of torch.tensor) containing the posterior mean and variance.
        """
        warn(
            "Deprecated ULA.forward returns tuple (mean, var). This will return only mean in a future version in line with BaseSampling.forward. Use deepinv.sampling.sampling_builder instead to build an SKRock sampler",
            DeprecationWarning,
        )
        return self.sample(
            y, physics, x_init=x_init, seed=seed, g_statistics=g_statistics
        )
