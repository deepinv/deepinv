import torch.nn as nn

from deepinv.physics import Physics
from deepinv.optim import Prior
from deepinv.optim import DataFidelity
from typing import Dict, Any
import torch

# TODO: maybe remove the *args, **kwargs (at very least document what they are for e.g. proposal dists, clip etc)
class SamplingIterator(nn.Module):
    r"""
    Base class for sampling iterators.

    All samplers should implement the forward method which performs a single sampling step
    in the Markov chain :math:`X_t \rightarrow X_{t+1}`.
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
    ) -> torch.Tensor:
        r"""
        Performs a single sampling step: :math:`X_t \rightarrow X_{t+1}`

        This method implements one step of the Markov chain Monte Carlo sampling process,
        generating the next state :math:`X_{t+1}` given the current state :math:`X_t`.

        :param torch.Tensor x: Current state :math:`X_t` of the Markov chain
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator
        :param DataFidelity cur_data_fidelity: Negative log-likelihood
        :param Prior cur_prior: Negative log-prior term
        :param dict cur_params: Dictionary containing the parameters for the sampling algorithm
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Next state :math:`X_{t+1}` in the Markov chain
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Subclasses of SamplingIterator must implement the forward method")
