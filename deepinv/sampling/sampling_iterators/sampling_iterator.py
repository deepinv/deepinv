import torch.nn as nn

from deepinv.physics import Physics
from deepinv.optim import Prior
from deepinv.optim import DataFidelity
from typing import Any
import torch


class SamplingIterator(nn.Module):
    r"""
    Base class for sampling iterators.

    All samplers should implement the `forward` method which performs one step of the Markov chain Monte Carlo sampling process,
    generating the next state :math:`X_{t+1}` given the current state :math:`X_t`.
    Where :math:`X_t` is a `dict` containing the image :math:`x_t` as well as any latent variables.
    See the docs for :class:`deepinv.sampling.BaseSampling` for an example along with more information.

    :param dict algo_params: Dictionary containing the parameters for the sampling algorithm
    """

    def __init__(
        self,
        algo_params: dict[str, Any],
        **kwargs,
    ):
        super(SamplingIterator, self).__init__()
        self.algo_params = algo_params

    def initialize_latent_variables(
        self,
        x_init: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
    ) -> dict[str, Any]:
        r"""
        Initializes latent variables for the sampling iterator.

        This method is intended to be overridden by subclasses to initialize any latent variables
        required by the specific sampling algorithm. The default implementation simply returns the
        initial state `x` in a dictionary.

        :param torch.Tensor x_init: Initial state tensor.
        :param torch.Tensor y: Observed measurements/data tensor.
        :param Physics physics: Forward operator.
        :param DataFidelity cur_data_fidelity: Negative log-likelihood.
        :param Prior cur_prior: Negative log-prior term.
        :return: Dictionary containing the initial state `x` and any latent variables.
        """
        return {"x": x_init}

    def forward(
        self,
        X: dict[str, Any],
        y: torch.Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        iteration: int,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        r"""
        Performs a single sampling step: :math:`X_t \rightarrow X_{t+1}, where :math:`X_t` is a `dict` containing the image :math:`x_t` as well as any latents`

        :param Dict X: Dictionary containing the current image :math:`X_t` of the Markov chain along with any latent variables.
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator
        :param DataFidelity cur_data_fidelity: Negative log-likelihood
        :param Prior cur_prior: Negative log-prior term
        :param int iteration: Current iteration number in the sampling process (zero-indexed)
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Dictionary `{"x": x, ...}` containing the next state along with any latent variables.
        """
        raise NotImplementedError(
            "Subclasses of SamplingIterator must implement the forward method"
        )
