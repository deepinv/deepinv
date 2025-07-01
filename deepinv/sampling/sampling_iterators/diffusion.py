from deepinv.sampling.utils import projbox
import torch
import time as time
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator


class DiffusionIterator(SamplingIterator):
    r"""
    Helper class used by :class:`deepinv.sampling.DiffusionSampler` to interface diffusion models with the
    :class:`deepinv.sampling.BaseSampling` framework.

    .. note::
        Users should typically interact with :class:`deepinv.sampling.DiffusionSampler` rather than this class directly.
    """

    def __init__(self, cur_params=None, clip=None):
        super(SamplingIterator, self).__init__()
        self.clip = clip

    def forward(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        physics,
        cur_data_fidelity,
        prior,
        iteration,
    ) -> dict[str, torch.Tensor]:
        x = X["x"]
        # run one sampling kernel iteration
        x = prior(y, physics)
        if self.clip:
            x = projbox(x, self.clip[0], self.clip[1])
        return {"x": x}  # return the updated x
