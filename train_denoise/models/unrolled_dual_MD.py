import deepinv as dinv
import torch
from deepinv.optim.prior import RED
from deepinv.unfolded import unfolded_builder
from deepinv.optim import Bregman
from deepinv.models.icnn import ICNN
from deepinv.optim.optim_iterators import MDIteration, OptimIterator
from deepinv.loss.loss import Loss
from deepinv.optim.utils import gradient_descent

class DualMDIteration(MDIteration):

    def __init__(self, bregman_potential=dinv.optim.bregman.BregmanL2(), **kwargs):
        super(MDIteration, self).__init__(**kwargs)
        self.g_step = dinv.optim.optim_iterators.gradient_descent.gStepGD(**kwargs)
        self.f_step = dinv.optim.optim_iterators.gradient_descent.fStepGD(**kwargs)
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
        y_prev = X["est"][0]
        x_prev = self.bregman_potential.grad_conj(y_prev)
        grad = cur_params["stepsize"] * (
            self.g_step(x_prev, cur_prior, cur_params)
            + self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
        )
        y = y_prev - grad
        return {"est": (y,)}


class DeepBregman(Bregman):
    r"""
    Module for the using a deep NN as Bregman potential.
    """

    def __init__(self, forw_model = None, conj_model = None):
        super().__init__()
        self.forw_model = forw_model
        self.conj_model = conj_model

    def fn(self, x, *args, init = None, **kwargs):
        r"""
        Computes the Bregman potential.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`\phi(x)`.
        """
        if self.forw_model is not None:
            return self.forw_model(x, *args, **kwargs)
        else:
            return self.conjugate_conjugate(x, *args, init = init, **kwargs)

    def grad(self, x, *args, init = None, **kwargs):
        if self.forw_model is not None:
            return super().grad(x, *args, **kwargs)
        else:
            return self.grad_conj_conj(x, *args, init = init, **kwargs)

    def conjugate(self, x, *args, **kwargs):
        r"""
        Computes the convex conjugate potential.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        if self.conj_model is not None:
            return self.conj_model(x, *args, **kwargs)
        else:
            super().conjugate(x, *args, **kwargs)
        return

    def conjugate_conjugate(self, x, *args, init = None, **kwargs)
        grad = lambda z: self.grad_conj(z, *args, **kwargs) - x
        init = x if init is None else init
        z = gradient_descent(-grad, init)
        return self.conjugate(z, *args, **kwargs) - torch.sum(
            x.reshape(x.shape[0], -1) * z.reshape(z.shape[0], -1), dim=-1
        ).view(x.shape[0], 1)

    def grad_conj_conj(self, x, *args, method = 'fixed-point', init = None, **kwargs):
        if method == 'backprop':
            with torch.enable_grad():
                x = x.requires_grad_()
                h = self.conjugate_conjugate(x, *args, init = init, **kwargs)
                grad = torch.autograd.grad(
                    h,
                    x,
                    torch.ones_like(h),
                    create_graph=True,
                    only_inputs=True,
                )[0]
            return grad
        else: 
            init = x if init is None else init
            grad = lambda z: self.grad_conj(z, *args, **kwargs) - x
            return gradient_descent(grad, init)
            

class MirrorLoss(Loss):
    def __init__(self, metric=torch.nn.MSELoss()):
        super(MirrorLoss, self).__init__()
        self.name = "mirror"
        self.metric = metric

    def forward(self, x, x_net, y, physics, model, *args, **kwargs):
        bregman_potential = model.fixed_point.iterator.bregman_potential
        return self.metric(bregman_potential.grad_conj(bregman_potential.grad(x_net)), x_net)


def get_unrolled_architecture(max_iter = 10, data_fidelity="L2", prior_name="wavelet", denoiser_name="DRUNET", stepsize_init=1.0, lamb_init=1.0, device = "cpu", 
                                use_mirror_loss=False, use_dual_iterations = False, strong_convexity_backward=0.5, strong_convexity_forward=0.1, strong_convexity_potential='L2'):

    # Select the data fidelity term
    if data_fidelity == 'L2':
        data_fidelity = dinv.optim.data_fidelity.L2()
    elif data_fidelity == 'KL':
        data_fidelity = dinv.optim.data_fidelity.PoissonLikelihood()
        
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
    else:
        forw_bregman = None
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

    bregman_potential = DeepBregman(forw_model = forw_bregman, conj_model = back_bregman)

    if use_dual_iterations:
        iteration = DualMDIteration(bregman_potential = bregman_potential)
        custom_init = lambda y, physics : {'est' : bregman_potential(physics.A_adjoint(y))}
        custom_output = lambda X : bregman_potential.grad_conj(X["est"][0])
    else:
        iteration = MDIteration(bregman_potential = bregman_potential)
        custom_init = lambda y, physics : {'est' : physics.A_adjoint(y)}
        custom_output = lambda X: X["est"][0]

    # Define the unfolded trainable model.
    model = unfolded_builder(
        iteration = MDIteration(F_fn=None, has_cost=False),
        params_algo=params_algo.copy(),
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        custom_init=custom_init,
        get_output = custom_output
    )

    return model.to(device)
