from typing import Tuple
from warnings import warn

import numpy as np
import torch
from deepinv.physics.generator import PhysicsGenerator


class BernoulliSplittingMaskGenerator(PhysicsGenerator):
    """Base generator for splitting masks.

    Generates binary masks with a given split ratio, according to a Bernoulli distribution.

    :param Tuple tensor_size: size of the tensor to be masked without batch dimension e.g. of shape (C, H, W) or (C, M) or (M,)
    :param float split_ratio: ratio of values to be kept.
    :param bool pixelwise: Apply the mask in a pixelwise fashion, i.e., zero all channels in a given pixel simultaneously.
    :param torch.device device: device where the tensor is stored (default: 'cpu').
    :param np.random.Generator, torch.Generator rng: random number generator.
    """

    def __init__(
        self,
        tensor_size: Tuple,
        split_ratio: float,
        pixelwise: bool = True,
        device: torch.device = torch.device("cpu"),
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tensor_size = tensor_size
        self.split_ratio = split_ratio
        self.pixelwise = pixelwise
        self.device = device
        self.rng = (
            rng
            if rng is not None
            else torch.Generator(device=self.device).manual_seed(0)
        )

        if self.pixelwise and len(self.tensor_size) == 2:
            warn(
                "Generating pixelwise mask assumes channel in first dimension. For 2D images (i.e. of shape (H,W)) ensure tensor_size is at least 3D (i.e. C,H,W). However, for tensor_size of shape (C,M), this will work as expected."
            )
        elif self.pixelwise and len(self.tensor_size) == 1:
            warn("For 1D tensor_size, pixelwise must be False.")
            self.pixelwise = False

    def step(self, batch_size=1, input_mask=None) -> dict:
        r"""
        Create a bernoulli mask.

        If ``input_mask`` is None, generates a standard random mask that can be used for :class:`deepinv.physics.Inpainting`.
        If ``input_mask`` is specified, splits the input mask into subsets given the split ratio.

        :param int batch_size: batch_size.
        :param torch.Tensor, None input_mask: optional mask to be split. If None, all pixels are considered. If not None, only pixels where mask==1 are considered.
        :return: dictionary with key **'mask'**: tensor of size ``(batch_size, *tensor_size)`` with values in {0, 1}.
        :rtype: dict
        """
        if isinstance(input_mask, torch.Tensor) and input_mask.numel() > 1:
            # Sample indices from given input mask

            # Check if pixelwise can be used
            batch_size = (
                0  # NOTE batch_size argument not used when input_mask is passed
            )
            pixelwise = self.pixelwise

            if len(input_mask.shape) == 1:
                warn("input_mask is only 1D so pixelwise cannot be used.")
                pixelwise = False
            elif len(input_mask.shape) == 2 and len(input_mask.shape) < len(
                self.tensor_size
            ):
                # When input_mask 2D, this can either be shape C,M or H,W.
                # When input_mask C,M, tensor_size will also be C,M (as passed in from SplittingLoss) and pixelwise can be used safely.
                # When input_mask H,W but tensor_size higher-dimensional e.g. C,H,W, then pixelwise should be set to False as it will happen anyway.
                pixelwise = False
            elif len(input_mask.shape) > len(self.tensor_size):
                # Batch size in input_mask
                batch_size = input_mask.shape[0]
            elif not all(
                torch.equal(input_mask[i], input_mask[0])
                for i in range(1, input_mask.shape[0])
            ):
                warn("To use pixelwise, all channels must be same.")
                pixelwise = False

            # Sample indices
            if pixelwise:
                if batch_size > 0:
                    idx = input_mask[:, 0, ...].nonzero(as_tuple=False)
                else:
                    idx = input_mask[0, ...].nonzero(as_tuple=False)
            else:
                idx = input_mask.nonzero(as_tuple=False)

            shuff = idx[torch.randperm(len(idx), generator=self.rng)]
            idx_out = shuff[: int(self.split_ratio * len(idx))].t()

            mask = torch.zeros_like(input_mask)

            if pixelwise:
                if batch_size > 0:
                    mask = mask[:, 0, ...]
                    mask[tuple(idx_out)] = 1
                    mask = torch.stack([mask] * input_mask.shape[1], dim=1)
                else:
                    mask = mask[0, ...]
                    mask[tuple(idx_out)] = 1
                    mask = torch.stack([mask] * input_mask.shape[0])
            else:
                mask[tuple(idx_out)] = 1

        else:

            # Sample pixels from a uniform distribution
            mask = (
                torch.ones(self.tensor_size, device=self.device)
                if batch_size is None
                else torch.ones((batch_size, *self.tensor_size), device=self.device)
            )
            aux = torch.rand(self.tensor_size, generator=self.rng, device=self.device)
            if not self.pixelwise:
                mask[aux > self.split_ratio] = 0
            else:
                mask[:, aux[0, ...] > self.split_ratio] = 0

        return {"mask": mask}
