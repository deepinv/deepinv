import torch.nn as nn
import torch
import numpy as np
import time as time

from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator


class ULAIterator(SamplingIterator):
    r"""
    Single iteration of the Unadjusted Langevin Algorithm.


    Expected cur_params dict:
    :param float step_size: step size :math:`\eta>0` of the algorithm.
    :param float alpha: regularization parameter :math:`\alpha`.
    :param float sigma: noise level used in the plug-and-play prior denoiser.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, x, y, physics, cur_data_fidelity, cur_prior, cur_params, *args, **kwargs
    ):
        noise = torch.randn_like(x) * np.sqrt(2 * cur_params["step_size"])
        lhood = -cur_data_fidelity.grad(x, y, physics)
        lprior = -cur_prior.grad(x, cur_params["sigma"]) * cur_params["alpha"]
        return x + cur_params["step_size"] * (lhood + lprior) + noise
