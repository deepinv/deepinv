import deepinv as dinv
import torch
from deepinv.optim.prior import RED
from deepinv.unfolded import unfolded_builder
from deepinv.optim import Bregman
from deepinv.models.icnn import ICNN
from deepinv.optim.optim_iterators import MDIteration
from deepinv.loss.loss import Loss

class MDIteration(OptimIterator):
    r"""
    Iterator for Mirror Descent.

    Class for a single iteration of the mirror descent (GD) algorithm for minimising :math:`f(x) + \lambda g(x)`.

    For a given convex potential :math:`h`, the iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        v_{k} &= \nabla f(x_k) + \nabla g(x_k) \\
        x_{k+1} &= \nabla h^*(\nabla h(x_k) - \gamma v_{k})
        \end{aligned}
        \end{equation*}


   where :math:`\gamma` is a stepsize.
   The potential :math:`h` should be specified in the cur_params dictionary.
    """

    def __init__(self, bregman_potential=BregmanL2(), **kwargs):
        super(MDIteration, self).__init__(**kwargs)
        self.g_step = gStepGD(**kwargs)
        self.f_step = fStepGD(**kwargs)
        self.requires_grad_g = True
        self.bregman_potential = bregman_potential

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single mirror descent iteration on the objective :math:`f(x) + \lambda g(x)`.
        The Bregman potential, which is an intance of the deepinv.optim.Bregman class, is used to compute the mirror descent step.

        :param dict X: Dictionary containing the current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the `Physics` class defining the current physics.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"][0]
        grad = cur_params["stepsize"] * (
            self.g_step(x_prev, cur_prior, cur_params)
            + self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
        )
        x = self.bregman_potential.grad_conj(self.bregman_potential.grad(x_prev) - grad)
        F = (
            self.F_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.has_cost
            else None
        )
        return {"est": (x,), "cost": F}


class fStepGD(fStep):
    r"""
    GD fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        r"""
        Single gradient descent iteration on the data fit term :math:`f`.

        :param torch.Tensor x: current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        return cur_data_fidelity.grad(x, y, physics)


class gStepGD(gStep):
    r"""
    GD gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        return cur_params["lambda"] * cur_prior.grad(x, cur_params["g_param"])


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


def get_unrolled_architecture(max_iter = 10, data_fidelity="L2", prior_name="wavelet", denoiser_name="DRUNET", stepsize_init=1.0, lamb_init=1.0, device = "cpu", 
                                use_mirror_loss=False, use_dual_iterations = False, strong_convexity_backward=0.5, strong_convexity_forward=0.1, strong_convexity_potential='L2'):

    # Select the data fidelity term
    if data_fidelity == 'L2':
        data_fidelity = dinv.optim.data_fidelity.L2()
    elif data_fidelity == 'KL':
        data_fidelity = dinv.optim.data_fidelity.KL()
        
    # Set up the prior
    if prior_name == 'wavelet':
        prior = dinv.optim.WaveletPrior(wv="db8", level=3, device=device)

    # Unrolled optimization algorithm parameters
    stepsize = [stepsize_init] * max_iter  # stepsize of the algorithm
    lamb = [lamb_init] * max_iter

    if use_mirror_loss: 
        forw_bregman = ICNN(in_channels=3,
                            num_filters=64,
                            kernel_dim=5,
                            num_layers=10,
                            strong_convexity=strong_convexity_forward,
                            pos_weights=True,
                            device="cpu").to(device)
    back_bregman = ICNN(in_channels=3,
                            num_filters=64,
                            kernel_dim=5,
                            num_layers=10,
                            strong_convexity=strong_convexity_backward,
                            pos_weights=True,
                            device="cpu").to(device)

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
