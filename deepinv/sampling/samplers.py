import sys
import warnings
from collections import deque
from collections.abc import Iterable
import torch
from tqdm import tqdm
from deepinv.physics import Physics
from deepinv.optim.optim_iterators import *
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.prior import Zero, Prior
from deepinv.loss.metric.distortion import PSNR
from deepinv.models import Reconstructor
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator
from deepinv.sampling.utils import Welford


class BaseSample(Reconstructor):
    def __init__(
        self,
        iterator: SamplingIterator,
        data_fidelity: DataFidelity,
        prior: Prior,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        num_iter=100,
        burnin_ratio=0.2,
        thinning=10,
        g_statistics=[lambda x: x],
        history_size=5,
        verbose=False,
    ):
        super(BaseSample, self).__init__()
        self.iterator = iterator
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.params_algo = params_algo
        self.num_iter = num_iter
        self.burnin_ratio = burnin_ratio
        self.thinning = thinning
        self.g_statistics = g_statistics
        self.verbose = verbose
        self.history_size = history_size
        # Stores last history_size samples note float('inf') => we store the whole chain
        self.history = deque(maxlen=history_size)

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        X_init: torch.Tensor | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Run the sampling chain to generate samples from the posterior and return averages of relevant statistics.

        :param torch.Tensor y: Measurements
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param torch.Tensor X_init: Initial state. If None, will use A^T(y) as initial state
        :param kwargs: Additional arguments passed to the iterator (e.g. proposal distributions)
        :return: Mean and variance of g_statistics. If single g_statistic, returns (mean, var) as tensors.
                If multiple g_statistics, returns (means, vars) as lists of tensors.
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Initialization
        if X_init is None:
            x = physics.A_adjoint(y)
        else:
            x = X_init

        self.history = deque([x], maxlen=self.history_size)

        # Initialize Welford trackers for each g_statistic
        statistics = []
        for g in self.g_statistics:
            statistics.append(Welford(g(x)))

        # Run the chain
        for i in tqdm(range(self.num_iter), disable=(not self.verbose)):
            x = self.iterator(
                x,
                y,
                physics,
                self.data_fidelity,
                self.prior,
                self.params_algo,
                **kwargs,
            )

            if i >= (self.num_iter * self.burnin_ratio) and i % self.thinning == 0:
                self.history.append(x)

                for j, (g, stat) in enumerate(zip(self.g_statistics, statistics)):
                    stat.update(g(x))

            if self.verbose and i % (self.num_iter // 10) == 0:
                print(f"Iteration {i}/{self.num_iter}")

        # Return means and variances for all g_statistics
        means = [stat.mean() for stat in statistics]
        vars = [stat.var() for stat in statistics]

        # Unwrap single statistics
        if len(self.g_statistics) == 1:
            return means[0], vars[0]
        return means, vars

    def get_history(self) -> list[torch.Tensor]:
        """
        Returns the current history of (thinned) samples.

        Returns:
            list[torch.Tensor]: List of stored samples from oldest to newest.
        """
        return list(self.history)

