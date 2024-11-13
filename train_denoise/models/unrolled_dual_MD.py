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
            return super().conjugate(x, *args, **kwargs)

    def grad_conj(self, x, *args, **kwargs):
        with torch.enable_grad():
            x = x.requires_grad_()
            h = self.conjugate(x, *args, **kwargs)
            grad = torch.autograd.grad(
                h,
                x,
                torch.ones_like(h),
                create_graph=True,
                only_inputs=True,
            )[0]
        return grad
        
    def conjugate_conjugate(self, x, *args, init = None, **kwargs):
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


class NoLipLoss(Loss):
    def __init__(self, L = 1., eps_jacobian_loss = 0.05, jacobian_loss_weight = 1e-2, max_iter_power_it=10, tol_power_it=1e-3, verbose=False, eval_mode=False, use_interpolation=False):
        super(NoLipLoss, self).__init__()
        self.spectral_norm_module = dinv.loss.JacobianSpectralNorm(
            max_iter=max_iter_power_it, tol=tol_power_it, verbose=verbose, eval_mode=eval_mode
        )
        self.L = L
        self.use_interpolation = use_interpolation
        self.eps_jacobian_loss = eps_jacobian_loss
        self.jacobian_loss_weight = jacobian_loss_weight

    def forward(self, x, x_net, y, physics, model, *args, **kwargs):

        if self.use_interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * x.detach() + (1 - eta) * x_net.detach()
        else:
            x = x
        
        bregman_potential = model.fixed_point.iterator.bregman_potential
    
        x.requires_grad_()
        x = bregman_potential.grad_conj(x)
        # We need to apply to the loss at each iteration.For now we assume the model is fixed along iterations
        it = 0
        cur_params = model.update_params_fn(it)
        cur_data_fidelity = model.update_data_fidelity_fn(it)
        cur_prior = model.update_prior_fn(it)
        nabla_F = cur_data_fidelity.grad(x, y, physics) # + cur_params["lambda"] * cur_prior.grad(x, cur_params["g_param"])
        jacobian_norm = (1 / self.L) * self.spectral_norm_module(nabla_F, x)
        jacobian_loss = self.jacobian_loss_weight * torch.maximum(jacobian_norm, torch.ones_like(jacobian_norm)-self.eps_jacobian_loss)
        return jacobian_loss




def get_unrolled_architecture(max_iter = 10, data_fidelity="L2", prior_name="wavelet", denoiser_name="DRUNET", stepsize_init=1.0, lamb_init=1.0, sigma_denoiser_init = 0.03, device = "cpu", 
                                use_mirror_loss=False, use_dual_iterations = False, strong_convexity_backward=0.5, strong_convexity_forward=0.1, strong_convexity_potential='L2'):

    # Select the data fidelity term
    if data_fidelity.lower() == 'l2':
        data_fidelity = dinv.optim.data_fidelity.L2()
    elif data_fidelity.lower() == 'kl':
        data_fidelity = dinv.optim.data_fidelity.PoissonLikelihood()
        
    # Set up the prior
    if prior_name.lower() == 'wavelet':
        prior = dinv.optim.WaveletPrior(wv="db8", level=3, device=device)
    elif prior_name.lower() == 'red':
        if denoiser_name.lower() == 'drunet':
            denoiser = dinv.models.DRUNet(pretrained="ckpts/drunet_deepinv_color.pth", device=device)
        if denoiser_name.lower() == 'dncnn':
            denoiser = dinv.models.DnCNN(pretrained="ckpts/dncnn_sigma2_color.pth", device=device)
            sigma_denoiser_init = 2/255.
        prior = dinv.optim.prior.RED(denoiser = denoiser)
        sigma_denoiser = [sigma_denoiser_init] # only one sigma

    # Freeze prior for now
    for param in prior.parameters():
        param.requires_grad = False

    # Unrolled optimization algorithm parameters
    stepsize = [stepsize_init]  # stepsize of the algorithm, only one stepsize.
    lamb = [lamb_init] # only one lambda.
    
    if use_mirror_loss: 
        forw_bregman = ICNN(in_channels=3,
                            num_filters=32,
                            kernel_dim=3,
                            num_layers=5,
                            strong_convexity=strong_convexity_forward,
                            pos_weights=False,
                            device="cpu").to(device)
    else:
        forw_bregman = None
    back_bregman = ICNN(in_channels=3,
                            num_filters=32,
                            kernel_dim=3,
                            num_layers=5,
                            strong_convexity=strong_convexity_backward,
                            pos_weights=True,
                            device="cpu").to(device)

    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "lambda": lamb,
        "g_param": sigma_denoiser if prior_name == 'RED' else None
    }
    trainable_params = [
        "lambda",
        "stepsize",
    ]  # define which parameters from 'params_algo' are trainable

    bregman_potential = DeepBregman(forw_model = forw_bregman, conj_model = back_bregman)

    if use_dual_iterations:
        iteration = DualMDIteration(bregman_potential = bregman_potential)
        def custom_init(y, physics, stop_grad = True):
            if stop_grad:
                with torch.no_grad():
                    return {'est' : [bregman_potential.grad(physics.A_adjoint(y))]}
            else:
                return {'est' : [bregman_potential.grad(physics.A_adjoint(y))]}
        custom_output = lambda X : bregman_potential.grad_conj(X["est"][0])
    else:
        iteration = MDIteration(bregman_potential = bregman_potential)
        custom_init = lambda y, physics : {'est' : [physics.A_adjoint(y)]}
        custom_output = lambda X: X["est"][0]

    # Define the unfolded trainable model.
    model = unfolded_builder(
        iteration=iteration,
        params_algo=params_algo.copy(),
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        custom_init=custom_init,
        get_output = custom_output
    )

    return model.to(device)
