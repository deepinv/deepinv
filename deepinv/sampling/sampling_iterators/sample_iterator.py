import torch.nn as nn

from deepinv.physics import Physics
from deepinv.optim import Prior
from deepinv.optim import DataFidelity
from typing import Dict, Any
import torch


class SamplingIterator(nn.Module):
    r"""
    Base class for sampling iterators.

    All samplers should implement the forward method which performs a single sampling step.
    """

    def __init__(self, **kwargs):
        super(SamplingIterator, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        cur_params: Dict[str, Any],
        *args,
        **kwargs,
    ):
        r"""
        Performs a single sampling step: X_t -> X_{t+1}

        Args:
            x: Current state X_t
            y: Observed data
            physics: Forward operator
            cur_data_fidelity: Data fidelity term
            cur_prior: Prior term
            cur_params: Dictionary of sampling parameters
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Next state X_{t+1}
        """
        pass
