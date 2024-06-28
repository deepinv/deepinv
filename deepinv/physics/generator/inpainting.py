import torch
import numpy as np
from typing import Tuple
from deepinv.physics.generator import PhysicsGenerator


class BernoulliSplittingMaskGenerator(PhysicsGenerator):
    """Base generator for splitting masks.

    Generates binary masks with a given split ratio, according to a Bernoulli distribution.

    :param Tuple tensor_size: size of the tensor to be masked.
    :param float split_ratio: ratio of values to be kept.
    :param torch.device device: device where the tensor is stored (default: 'cpu').
    :param np.random.Generator, torch.Generator rng: random number generator.
    """

    def __init__(
        self,
        tensor_size: Tuple,
        split_ratio: float,
        device: torch.device = torch.device("cpu"),
        rng: torch.Generator = torch.Generator().manual_seed(0),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tensor_size = tensor_size
        self.split_ratio = split_ratio
        self.device = device
        self.rng = rng

    def step(self, batch_size=1, input_mask=None) -> dict:
        r"""
        Create a bernoulli mask.

        :param int batch_size: batch_size.
        :param torch.Tensor, None input_mask: optional mask to be split. If None, all pixels are considered. If not None, only pixels where mask==1 are considered.
        :return: dictionary with key **'mask'**: tensor of size (batch_size, *tensor_size) with values in {0, 1}.
        :rtype: dict
        """
        if not isinstance(input_mask, torch.Tensor) or tuple(input_mask.shape) != (batch_size, *self.tensor_size):
            # Sample pixels from a uniform distribution
            mask = torch.ones((batch_size, *self.tensor_size), device=self.device)
            mask[torch.empty_like(mask).uniform_(generator=self.rng) > self.split_ratio] = 0
        else:
            # Sample indices from input mask
            idx = input_mask.nonzero(as_tuple=False)
            shuff = idx[torch.randperm(len(idx), generator=self.rng)]
            idx_out = shuff[:int(self.split_ratio * len(idx))]

            mask = torch.zeros_like(input_mask)
            mask[tuple(idx_out.t())] = 1

        return {"mask": mask}
