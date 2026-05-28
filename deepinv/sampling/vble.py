from __future__ import annotations

import torch

from deepinv.models.latent_optimizer import VBLExzOptimizer, VBLEzOptimizer
from deepinv.sampling import BaseSampling
from deepinv.sampling.sampling_iterators.vble_iterator import VBLEIterator


class VBLESampling(BaseSampling):
    r"""
    Turns :class:`deepinv.models.VBLEzOptimizer` and :class:`deepinv.models.VBLExzOptimizer` into a Monte-Carlo
    sampler using the optimized variational parameters.  Unlike MCMC methods, the resulting sampler
    computes the mean and variance of the distribution by sampling the approximate posterior distribution
    multiple times. This sampling procedure can be parallelized by drawing samples in batches by setting
    the `batch_size` parameter.

    :param VBLEzOptimizer | VBLExzOptimizer vble_model: Variational Bayesian Latent Estimator model.
    :param int max_iter: Number of posterior samples to draw.
    :param tuple[int, int] clip: Min and max values for clipping the samples.
    :param float thres_conv: Convergence threshold for the mean and variance.
    :param bool verbose: If `True`, prints progress information during sampling.
    :param bool save_chain: If `True`, saves all samples drawn during the sampling process.
    :param int batch_size: Number of samples to draw in each batch.
    """

    def __init__(
        self,
        vble_model: VBLEzOptimizer | VBLExzOptimizer,
        max_iter: int = 100,
        clip: tuple[int, int] = (0, 1),
        thres_conv=1e-1,
        verbose=True,
        save_chain=False,
        batch_size: int = 16,
    ):
        vble_iterator = VBLEIterator(clip=clip)
        super().__init__(
            vble_iterator,
            None,
            vble_model,
            max_iter=max_iter // batch_size + 1,
            thinning=1,
            thresh_conv=thres_conv,
            history_size=save_chain,
            burnin_ratio=0.0,
            verbose=verbose,
        )

        self.g_statistics = [lambda d: d["x"]]
        self.batch_size = batch_size

    def sample(
        self,
        seed: int | None = None,
        **kwargs,
    ):
        r"""
        Execute the sampling chain and compute posterior statistics.

        :param int | None seed: Random seed for reproducibility.
        """
        batch_means, batch_vars = super().sample(
            None,
            None,
            x_init=self.iterator(prior=self.prior, batch_size=self.batch_size),
            seed=seed,
            g_statistics=self.g_statistics,
            batch_size=self.batch_size,
            **kwargs,
        )

        # Compute overall mean and variance across all batches
        if isinstance(batch_means, list):
            means = [torch.mean(bm, dim=0, keepdim=True) for bm in batch_means]
            vars = [torch.mean(bv, dim=0, keepdim=True) for bv in batch_vars]
        else:
            means = torch.mean(batch_means, dim=0, keepdim=True)
            vars = torch.mean(batch_vars, dim=0, keepdim=True)
        return means, vars

    def get_chain(self) -> list[torch.Tensor]:
        r"""
        Retrieve the full sampling chain as a list of samples.

        :returns: List of samples drawn during the sampling process.
        """
        if self.history is False:
            raise RuntimeError(
                "Cannot get chain: history storage is disabled (history_size=False)"
            )
        history = list(self.history)[1:]
        return [
            {"x": s} for h in history for s in h["x"].chunk(self.batch_size, dim=0)
        ]  # split samples if batched
