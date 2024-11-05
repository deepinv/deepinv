import deepinv as dinv
import torch
from deepinv.optim.prior import RED
from deepinv.unfolded import unfolded_builder
from deepinv.optim import Bregman
from deepinv.models.icnn import ICNN
from deepinv.optim.optim_iterators import MDIteration
from deepinv.loss.loss import Loss

class DeepBregman(Bregman):
    r"""
    Module for the using a deep NN as Bregman potential.
    """

    def __init__(self, forw_model, conj_model = None):
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
    

class MirrorLoss(Loss):
    def __init__(self, metric=torch.nn.MSELoss()):
        super(MirrorLoss, self).__init__()
        self.name = "mirror"
        self.metric = metric

    def forward(self, x, x_net, y, physics, model, *args, **kwargs):
        bregman_potential = model.params_algo.bregman_potential[0]
        return self.metric(bregman_potential.grad_conj(bregman_potential.grad(x_net)), x_net)


def get_unrolled_architecture(max_iter = 10, device = "cpu"):

    # Select the data fidelity term
    data_fidelity = dinv.optim.data_fidelity.L2()
    # Set up the prior
    prior = dinv.optim.WaveletPrior(wv="db8", level=3, device=device)

     # Unrolled optimization algorithm parameters
    max_iter = 5  # number of unfolded layers
    stepsize = [1] * max_iter  # stepsize of the algorithm
    lamb = [1] * max_iter

    forw_bregman = ICNN(in_channels=3, dim_hidden=256, device = device).to(device)
    back_bregman = ICNN(in_channels=3, dim_hidden=256, device = device).to(device)
    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "lambda": lamb,
    }
    trainable_params = [
        "lambda",
        "stepsize",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.
    model = unfolded_builder(
        iteration = MDIteration(F_fn=None, has_cost=False),
        params_algo=params_algo.copy(),
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        bregman_potential = DeepBregman(forw_model = forw_bregman, conj_model = back_bregman)
    )

    return model.to(device)
