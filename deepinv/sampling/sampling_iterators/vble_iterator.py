import torch

from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator
from deepinv.sampling.utils import projbox


class VBLEIterator(SamplingIterator):
    r"""
    Helper class used by :class:`deepinv.sampling.VBLESampling` to interface diffusion models with the
    :class:`deepinv.sampling.BaseSampling` framework.
    """

    def __init__(self, cur_params=None, clip=None):
        super(VBLEIterator, self).__init__(cur_params)
        self.clip = clip

    def forward(
        self,
        X: dict[str, torch.Tensor] = None,
        y: torch.Tensor = None,
        physics=None,
        cur_data_fidelity=None,
        prior=None,
        iteration=None,
        batch_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        # run one sampling kernel iteration
        x = prior.generate_solution(n_samples=batch_size)["x_rec"]

        if self.clip:
            x = projbox(x, self.clip[0], self.clip[1])
        return {"x": x}  # return the updated x
