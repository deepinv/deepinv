import sys
from collections import deque
import torch
from tqdm import tqdm
from deepinv.physics import Physics
from deepinv.optim.optim_iterators import *
from deepinv.optim.prior import Prior
from deepinv.models import Reconstructor
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator
from deepinv.sampling.utils import Welford
from deepinv.sampling.sampling_iterators import *
from deepinv.optim.utils import check_conv
from typing import Union, Dict, Callable, List, Tuple


class BaseSample(Reconstructor):
    r"""
    Base class for Monte Carlo sampling.

    This class can be used to create new Monte Carlo samplers by implementing the sampling kernel through :class:`deepinv.sampling.SamplingIterator`:

    ::

        # define your sampler (possibly a Markov kernel which depends on the previous sample)
        class MyIterator(SamplingIterator):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, physics, data_fidelity, prior, params_algo):
                # run one sampling kernel iteration
                new_x = f(x, y, physics, data_fidelity, prior, params_algo)
                return new_x

        # create the sampler
        sampler = BaseSampler(MyIterator(), prior, data_fidelity, iterator_params)

        # compute posterior mean and variance of reconstruction of x
        mean, var = sampler(y, physics)

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
    :param Callable callback: A funciton that is called on every (thinned) sample for diagnostics.
    :param int history_size: Number of most recent samples to store in memory. Default: 5
    :param bool verbose: Whether to print progress of the algorithm. Default: ``False``
    """

    def __init__(
        self,
        iterator: SamplingIterator,
        data_fidelity: DataFidelity,
        prior: Prior,
        max_iter: int = 100,
        callback: Callable = lambda x: x,
        burnin_ratio: float = 0.2,
        thresh_conv: float = 1e-3,
        crit_conv: str = "residual",
        thinning: int = 10,
        history_size: Union[int, bool] = 5,
        verbose: bool = False,
    ):
        super(BaseSample, self).__init__()
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

        # initialize history to zero
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
        X_init: Union[torch.Tensor, None] = None,
        seed: Union[int, None] = None,
    ) -> torch.Tensor:
        r"""
        Run the MCMC sampling chain and return the posterior sample mean.

        :param torch.Tensor y: The observed measurements
        :param Physics physics: Forward operator of your inverse problem
        :param torch.Tensor X_init: Initial state of the Markov chain. If None, uses ``physics.A_adjoint(y)`` as the starting point
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :return: Posterior sample mean
        :rtype: torch.Tensor
        """

        # pass back out sample mean
        return self.sample(y, physics, X_init=X_init, seed=seed)[0]

    def sample(
        self,
        y: torch.Tensor,
        physics: Physics,
        X_init: Union[torch.Tensor, None] = None,
        seed: Union[int, None] = None,
        g_statistics: Union[Callable, List[Callable]] = [lambda x: x],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Execute the MCMC sampling chain and compute posterior statistics.

        This method runs the main MCMC sampling loop to generate samples from the posterior
        distribution and compute their statistics using Welford's online algorithm.

        :param torch.Tensor y: The observed measurements/data tensor
        :param Physics physics: Forward operator of your inverse problem.
        :param torch.Tensor X_init: Initial state of the Markov chain. If None, uses ``physics.A_adjoint(y)`` as the starting point
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :param list g_statistics: List of functions for which to compute posterior statistics. Default: ``[lambda x: x]``
            The sampler will compute the posterior mean and variance of each function in the list.
            Default: ``lambda x: x`` (identity function)
        :param Union[List[Callable], Callable] g_statistics: List of functions for which to compute posterior statistics, or a single function.
        :param kwargs: Additional arguments passed to the sampling iterator (e.g., proposal distributions)
        :return: | If a single g_statistic was specified: Returns tuple (mean, var) of torch.Tensors
            | If multiple g_statistics were specified: Returns tuple (means, vars) of lists of torch.Tensors

        Example:
            >>> # Basic usage with default settings
            >>> sampler = BaseSample(iterator, data_fidelity, prior)
            >>> mean, var = sampler(measurements, forward_operator)

            >>> # Using multiple statistics
            >>> sampler = BaseSample(
            ...     iterator, data_fidelity, prior,
            ...     g_statistics=[lambda x: x, lambda x: x**2]
            ... )
            >>> means, vars = sampler(measurements, forward_operator)
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Initialization
        if X_init is None:
            X_t = physics.A_adjoint(y)
        else:
            X_t = X_init

        if self.history:
            if isinstance(self.history, deque):
                self.history = deque([X_t], maxlen=self.history_size)
            else:
                self.history = [X_t]

        self.mean_convergence = False
        self.var_convergence = False

        if not isinstance(g_statistics, List):
            g_statistics = [g_statistics]

        # Initialize Welford trackers for each g_statistic
        statistics = []
        for g in g_statistics:
            statistics.append(Welford(g(X_t)))

        # Initialize for convergence checking
        mean_prevs = [stat.mean().clone() for stat in statistics]
        var_prevs = [stat.var().clone() for stat in statistics]

        # Run the chain
        for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
            X_t = self.iterator(
                X_t,
                y,
                physics,
                self.data_fidelity,
                self.prior,
                it,
                **kwargs,
            )

            if it >= (self.max_iter * self.burnin_ratio) and it % self.thinning == 0:
                self.callback(X_t)
                # Store previous means and variances for convergence check
                if it >= (self.max_iter - self.thinning):
                    mean_prevs = [stat.mean().clone() for stat in statistics]
                    var_prevs = [stat.var().clone() for stat in statistics]

                if self.history:
                    self.history.append(X_t)

                for _, (g, stat) in enumerate(zip(g_statistics, statistics)):
                    stat.update(g(X_t))

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

    def get_chain(self) -> List[torch.Tensor]:
        r"""
        Retrieve the stored history of samples.

        Returns a list of samples.

        Only includes samples after the burn-in period and, thinning.

        :return: List of stored samples from oldest to newest
        :rtype: list[torch.Tensor]
        :raises RuntimeError: If history storage was disabled (history_size=False)

        Example:
            >>> sampler = BaseSample(iterator, data_fidelity, prior, history_size=5)
            >>> _ = sampler(measurements, forward_operator)
            >>> samples = sampler.get_history()
            >>> latest_sample = samples[-1]  # Get most recent sample
        """
        if self.history is False:
            raise RuntimeError(
                "Cannot get chain: history storage is disabled (history_size=False)"
            )
        return list(self.history)

    def mean_has_converged(self) -> bool:
        r"""
        Returns a boolean indicating if the posterior mean verifies the convergence criteria.
        """
        return self.mean_convergence

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
        iterator_fn = str_to_class(iterator + "Iterator")
        return iterator_fn(cur_params, **kwargs)
    else:
        # If already a SamplingIterator instance, return as is
        return iterator


def sample_builder(
    iterator: Union[SamplingIterator, str],
    data_fidelity: DataFidelity,
    prior: Prior,
    params_algo: Dict = {},
    max_iter: int = 100,
    thresh_conv: float = 1e-3,
    burnin_ratio: float = 0.2,
    thinning: int = 10,
    history_size: int = 5,
    verbose: bool = False,
    **kwargs,
) -> BaseSample:
    r"""
    Helper function for building an instance of the :class:`deepinv.sampling.BaseSample` class.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :param data_fidelity: Negative log-likelihood function
    :param prior: Negative log-prior
    :param params_algo: Dictionary containing the parameters for the algorithm
    :param max_iter: Number of Monte Carlo iterations
    :param burnin_ratio: Percentage of iterations for burn-in
    :param thinning: Integer to thin the Monte Carlo samples
    :param history_size: Number of recent samples to store
    :param verbose: Whether to print progress
    :param kwargs: Additional keyword arguments passed to the iterator constructor when a string is provided as the iterator parameter
    :return: Configured BaseSample instance in eval mode
    """
    iterator = create_iterator(iterator, params_algo, **kwargs)
    # Note we put the model in evaluation mode (.eval() is a PyTorch method inherited from nn.Module)
    return BaseSample(
        iterator,
        data_fidelity=data_fidelity,
        prior=prior,
        max_iter=max_iter,
        thresh_conv=thresh_conv,
        burnin_ratio=burnin_ratio,
        thinning=thinning,
        history_size=history_size,
        verbose=verbose,
    ).eval()


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
