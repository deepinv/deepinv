import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import PoissonLikelihood, L2
from deepinv.optim.prior import RED
from deepinv.unfolded import unfolded_builder
from deepinv.optim import Bregman
from torchvision import transforms
from deepinv.utils.demo import load_dataset
from deepinv.models.icnn import ICNN
from deepinv.optim import OptimIterator
from deepinv.optim.optim_iterators import MDIteration
from deepinv.optim.optim_iterators.gradient_descent import gStepGD, fStepGD


class DeepBregman(Bregman):
    r"""
    Module for the using a deep NN as Bregman potential.
    """

    def __init__(self, forw_model, conj_model=None):
        super().__init__()
        self.forw_model = forw_model
        self.conj_model = conj_model

    def fn(self, x):
        r"""
        Computes the Bregman potential.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`\phi(x)`.
        """
        return self.forw_model(x)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        if self.conj_model is not None:
            return self.conj_model(x)
        else:
            super().conjugate(x)
        return


def get_unrolled_architecture(max_iter, device):

    # Select the data fidelity term
    data_fidelity = L2()
    # Set up the prior
    prior = dinv.optim.WaveletPrior(wv="db8", level=3, device=device)

    # Unrolled optimization algorithm parameters
    max_iter = 5  # number of unfolded layers
    stepsize = [1] * max_iter  # stepsize of the algorithm
    lamb = [1] * max_iter

    forw_bregman = ICNN(in_channels=3, dim_hidden=256, device=device)
    back_bregman = ICNN(in_channels=3, dim_hidden=256, device=device)
    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "lambda": lamb,
        "bregman_potential": DeepBregman(
            forw_model=forw_bregman, conj_model=back_bregman
        ),
    }
    trainable_params = [
        "lambda",
        "stepsize",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.
    model = unfolded_builder(
        iteration=MDIteration(F_fn=None, has_cost=False),
        params_algo=params_algo.copy(),
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
    )

    return model.to(device)
