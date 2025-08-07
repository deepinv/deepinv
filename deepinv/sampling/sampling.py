import sys
from collections import deque
from typing import Union, Callable

import torch
from tqdm import tqdm

from deepinv.models import Reconstructor
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim.prior import Prior
from deepinv.optim.utils import check_conv
from deepinv.physics import Physics, LinearPhysics
from deepinv.sampling.sampling_iterators import *
from deepinv.sampling.utils import Welford


class BaseSampling(Reconstructor):
    r"""
    Base class for Monte Carlo sampling.

    This class aims to sample from the posterior distribution :math:`p(x|y)`, where :math:`y` represents the observed
    measurements and :math:`x` is the (unknown) image to be reconstructed. The sampling process generates a
    sequence of states (samples) :math:`X_0, X_1, \ldots, X_N` from a Markov chain. Each state :math:`X_k` contains the
    current estimate of the unknown image, denoted :math:`x_k`, and may include other latent variables.
    The class then computes statistics (e.g., image posterior mean, image posterior variance) from the samples :math:`X_k`.

    This class can be used to create new Monte Carlo samplers by implementing the sampling kernel through :class:`deepinv.sampling.SamplingIterator`:

    ::

        # define your sampler (possibly a Markov kernel which depends on the previous sample)
        class MyIterator(SamplingIterator):
            def __init__(self):
                super().__init__()

            def initialize_latent_variables(x, y, physics, data_fidelity, prior):
                # initialize a latent variable
                latent_z = g(x, y, physics, data_fidelity, prior)
                return {"x": x, "z": latent_z}

            def forward(self, X, y, physics, data_fidelity, prior, params_algo):
                # run one sampling kernel iteration
                new_X = f(X, y, physics, data_fidelity, prior, params_algo)
                return new_X

        # create the sampler
        sampler = BaseSampling(MyIterator(), prior, data_fidelity, iterator_params)

        # compute posterior mean and variance of reconstruction of x
        mean, var = sampler.sample(y, physics)

    This class computes the mean and variance of the chain using Welford's algorithm, which avoids storing the whole
    Monte Carlo samples. It can also maintain a history of the `history_size` most recent samples.

    Note on retained sample calculation:
        With the default parameters (max_iter=100, burnin_ratio=0.2, thinning=10), the number
        of samples actually used for statistics is calculated as follows:

        - Total iterations: 100
        - Burn-in period: 100 * 0.2 = 20 iterations (discarded)
        - Remaining iterations: 80
        - With thinning of 10, we keep iterations 20, 30, 40, 50, 60, 70, 80, 90
        - This results in 8 retained samples used for computing the posterior statistics

    :param deepinv.sampling.SamplingIterator iterator: The sampling iterator that defines the MCMC kernel
    :param deepinv.optim.DataFidelity data_fidelity: Negative log-likelihood function linked with the noise distribution in the acquisition physics
    :param deepinv.optim.Prior prior: Negative log-prior
    :param int max_iter: The number of Monte Carlo iterations to perform. Default: 100
    :param float burnin_ratio: Percentage of iterations used for burn-in period (between 0 and 1). Default: 0.2
    :param int thinning: Integer to thin the Monte Carlo samples (keeping one out of `thinning` samples). Default: 10
    :param float thresh_conv: The convergence threshold for the mean and variance. Default: ``1e-3``
    :param Callable callback: A function that is called on every (thinned) sample state dictionary for diagnostics. It is called with the current sample `X`, the current `statistics` (a list of Welford objects), and the current iteration number `iter` as keyword arguments.
    :param history_size: Number of most recent samples to store in memory. If `True`, all samples are stored. If `False`, no samples are stored. If an integer, it specifies the number of most recent samples to store. Default: 5
    :param bool verbose: Whether to print progress of the algorithm. Default: ``False``
    """

    def __init__(
        self,
        iterator: SamplingIterator,
        data_fidelity: DataFidelity,
        prior: Prior,
        max_iter: int = 100,
        callback: Callable = lambda X, **kwargs: None,
        burnin_ratio: float = 0.2,
        thresh_conv: float = 1e-3,
        crit_conv: str = "residual",
        thinning: int = 10,
        history_size: Union[int, bool] = 5,
        verbose: bool = False,
    ):
        super(BaseSampling, self).__init__()
        self.iterator = iterator
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.max_iter = max_iter
        self.burnin_ratio = burnin_ratio
        self.thresh_conv = thresh_conv
        self.crit_conv = crit_conv
        self.callback = callback
        self.mean_convergence = False
        self.var_convergence = False
        self.thinning = thinning
        self.verbose = verbose
        self.history_size = history_size

        # Initialize history to zero
        if history_size is True:
            self.history = []
        elif history_size:
            self.history = deque(maxlen=history_size)
        else:
            self.history = False

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        x_init: Union[torch.Tensor, dict, None] = None,
        seed: Union[int, None] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Run the MCMC sampling chain and return the posterior sample mean.

        :param torch.Tensor y: The observed measurements
        :param Physics physics: Forward operator of your inverse problem
        :param Union[torch.Tensor, dict, None] x_init: Optional initial state of the Markov chain. This can be a ``torch.Tensor`` to initialize the image ``X["x"]``, or a ``dict`` to initialize the entire state ``X`` including any latent variables. In most cases, providing a tensor to initialize ``X["x"]`` will be sufficient.
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :return: Posterior sample mean
        :rtype: torch.Tensor
        """

        # pass back out sample mean
        return self.sample(y, physics, x_init=x_init, seed=seed, **kwargs)[0]

    def sample(
        self,
        y: torch.Tensor,
        physics: Physics,
        x_init: Union[torch.Tensor, dict, None] = None,
        seed: Union[int, None] = None,
        g_statistics: Union[Callable, list[Callable]] = [lambda d: d["x"]],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Execute the MCMC sampling chain and compute posterior statistics.

        This method runs the main MCMC sampling loop to generate samples from the posterior
        distribution and compute their statistics using Welford's online algorithm.

        :param torch.Tensor y: The observed measurements/data tensor
        :param Physics physics: Forward operator of your inverse problem.
        :param Union[torch.Tensor, dict, None] x_init: Optional initial state of the Markov chain. This can be a ``torch.Tensor`` to initialize the image ``X["x"]``, or a ``dict`` to initialize the entire state ``X`` including any latent variables. In most cases, providing a tensor to initialize ``X["x"]`` will be sufficient.
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :param list g_statistics: List of functions for which to compute posterior statistics.
            The sampler will compute the posterior mean and variance of each function in the list.
            The input to these functions is a dictionary `d` which contains the current state of the sampler alongside any latent variables. `d["x"]` will always be the current image. See specific iterators for details on what (if any) latent variables they provide.
            Default: ``lambda d: d["x"]`` (identity function on the image).
        :param Union[List[Callable], Callable] g_statistics: List of functions for which to compute posterior statistics, or a single function.
        :param kwargs: Additional arguments passed to the sampling iterator (e.g., proposal distributions)
        :return: | If a single g_statistic was specified: Returns tuple (mean, var) of torch.Tensors
            | If multiple g_statistics were specified: Returns tuple (means, vars) of lists of torch.Tensors

        Example::

            from deepinv.sampling import BaseSampling, ULAIterator

            iterator = ULAIterator(...) # define iterator

            # Basic usage with default settings
            sampler = BaseSampling(iterator, data_fidelity, prior)
            mean, var = sampler.sample(measurements, forward_operator)

            # Using multiple statistics
            sampler = BaseSampling(
                iterator, data_fidelity, prior,
                g_statistics=[lambda X: X["x"], lambda X: X["x"]**2]
            )
            means, vars = sampler.sample(measurements, forward_operator)

        """

        # Don't store computational graphs
        with torch.no_grad():
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)

            # Initialization of both our image chain and any latent variables
            if x_init is None:
                # if linear take adjoint (pseudo-inverse can be a bit unstable) else fall back to pseudoinverse
                if isinstance(physics, LinearPhysics):
                    X = self.iterator.initialize_latent_variables(
                        physics.A_adjoint(y), y, physics, self.data_fidelity, self.prior
                    )
                else:
                    X = self.iterator.initialize_latent_variables(
                        physics.A_dagger(y), y, physics, self.data_fidelity, self.prior
                    )
            else:
                if isinstance(x_init, dict):
                    X = x_init
                else:
                    X = self.iterator.initialize_latent_variables(
                        x_init, y, physics, self.data_fidelity, self.prior
                    )

            if self.history_size:
                if isinstance(self.history, deque):
                    self.history = deque([X], maxlen=self.history_size)
                else:
                    self.history = [X]

            self.mean_convergence = False
            self.var_convergence = False

            if not isinstance(g_statistics, list):
                g_statistics = [g_statistics]

            # Initialize Welford trackers for each g_statistic
            statistics = []
            for g in g_statistics:
                statistics.append(Welford(g(X)))

            # Initialize for convergence checking
            mean_prevs = [stat.mean().clone() for stat in statistics]
            var_prevs = [stat.var().clone() for stat in statistics]

            # Run the chain
            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                X = self.iterator(
                    X,
                    y,
                    physics,
                    self.data_fidelity,
                    self.prior,
                    it,
                    **kwargs,
                )

                if (
                    it >= (self.max_iter * self.burnin_ratio)
                    and it % self.thinning == 0
                ):
                    self.callback(X, statistics=statistics, iter=it)
                    # Store previous means and variances for convergence check
                    if it >= (self.max_iter - self.thinning):
                        mean_prevs = [stat.mean().clone() for stat in statistics]
                        var_prevs = [stat.var().clone() for stat in statistics]

                    if self.history_size:
                        self.history.append(X)

                    for _, (g, stat) in enumerate(
                        zip(g_statistics, statistics, strict=True)
                    ):
                        stat.update(g(X))

            # Check convergence for all statistics
            self.mean_convergence = True
            self.var_convergence = True

            if it > 1:
                # Check convergence for each statistic
                for j, stat in enumerate(statistics):
                    if not check_conv(
                        {"est": (mean_prevs[j],)},
                        {"est": (stat.mean(),)},
                        it,
                        self.crit_conv,
                        self.thresh_conv,
                        self.verbose,
                    ):
                        self.mean_convergence = False

                    if not check_conv(
                        {"est": (var_prevs[j],)},
                        {"est": (stat.var(),)},
                        it,
                        self.crit_conv,
                        self.thresh_conv,
                        self.verbose,
                    ):
                        self.var_convergence = False

            # Return means and variances for all g_statistics
            means = [stat.mean() for stat in statistics]
            vars = [stat.var() for stat in statistics]

            # Unwrap single statistics
            if len(g_statistics) == 1:
                return means[0], vars[0]
        return means, vars

    def get_chain(self) -> list[torch.Tensor]:
        r"""
        Retrieve the stored history of samples.

        Returns a list of dictionaries, where each dictionary contains the state of the sampler.

        Only includes samples after the burn-in period and thinning.

        :return: List of stored sample states (dictionaries) from oldest to newest. Each dictionary contains the sample `"x": x` along with any latent variables.
        :rtype: list[dict]
        :raises RuntimeError: If history storage was disabled (history_size=False)

        Example::

            from deepinv.sampling import BaseSampling, SamplingIterator

            sampler = BaseSampling(SamplingIterator(...), data_fidelity, prior, history_size=5)
            _ = sampler(measurements, forward_operator)
            history = sampler.get_chain()
            latest_state = history[-1]  # Get most recent state dictionary
            latest_sample = latest_state["x"] # Get sample from state

        """
        if self.history is False:
            raise RuntimeError(
                "Cannot get chain: history storage is disabled (history_size=False)"
            )
        return list(self.history)

    @property
    def mean_has_converged(self) -> bool:
        r"""
        Returns a boolean indicating if the posterior mean verifies the convergence criteria.
        """
        return self.mean_convergence

    @property
    def var_has_converged(self) -> bool:
        r"""
        Returns a boolean indicating if the posterior variance verifies the convergence criteria.
        """
        return self.var_convergence


def create_iterator(
    iterator: Union[SamplingIterator, str], cur_params, **kwargs
) -> SamplingIterator:
    r"""
    Helper function for creating an iterator instance of the :class:`deepinv.sampling.SamplingIterator` class.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :return: SamplingIterator instance
    """
    if isinstance(iterator, str):
        # If a string is provided, create an instance of the named class
        iterator_fn = getattr(sys.modules[__name__], iterator + "Iterator")
        return iterator_fn(cur_params, **kwargs)
    else:
        # If already a SamplingIterator instance, return as is
        return iterator


def sampling_builder(
    iterator: Union[SamplingIterator, str],
    data_fidelity: DataFidelity,
    prior: Prior,
    params_algo: dict = {},
    max_iter: int = 100,
    thresh_conv: float = 1e-3,
    burnin_ratio: float = 0.2,
    thinning: int = 10,
    history_size: Union[int, bool] = 5,
    verbose: bool = False,
    callback: Callable = lambda X, **kwargs: None,
    **kwargs,
) -> BaseSampling:
    r"""
    Helper function for building an instance of the :class:`deepinv.sampling.BaseSampling` class.

    See :ref:`sphx_glr_auto_examples_sampling_demo_sampling.py` and :ref:`mcmc` for example usage.

    See the docs for :class:`deepinv.sampling.BaseSampling` for further examples and information.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :param data_fidelity: Negative log-likelihood function
    :param prior: Negative log-prior
    :param params_algo: Dictionary containing the parameters for the algorithm
    :param max_iter: Number of Monte Carlo iterations
    :param burnin_ratio: Percentage of iterations for burn-in
    :param thinning: Integer to thin the Monte Carlo samples
    :param history_size: Number of most recent samples to store in memory. If `True`, all samples are stored. If `False`, no samples are stored. If an integer, it specifies the number of most recent samples to store. Default: 5
    :param verbose: Whether to print progress
    :param Callable callback: A function that is called on every (thinned) sample state dictionary for diagnostics. It is called with the current sample `X`, the current `statistics` (a list of Welford objects), and the current iteration number `iter` as keyword arguments.
    :param kwargs: Additional keyword arguments passed to the iterator constructor when a string is provided as the iterator parameter
    :return: Configured BaseSampling instance in eval mode
    """
    iterator = create_iterator(iterator, params_algo, **kwargs)
    # Note we put the model in evaluation mode (.eval() is a PyTorch method inherited from nn.Module)
    return BaseSampling(
        iterator,
        data_fidelity=data_fidelity,
        prior=prior,
        max_iter=max_iter,
        thresh_conv=thresh_conv,
        burnin_ratio=burnin_ratio,
        thinning=thinning,
        history_size=history_size,
        verbose=verbose,
        callback=callback,
    ).eval()
