
import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration

class DEQ(nn.Module):
    '''
    Deep Equilibrium Model. Strongly inspired from http://implicit-layers-tutorial.org/deep_equilibrium_models/. 

    Args : 
    iterator : y = iterator(x,it) forward model.
    PnP_module : nn.Module, Plug&Play module with trainable parameters.
    anderson_acceleration : bool, if True, use Anderson acceleration for the fixed-point forward and backward resolutions.
    max_iter_forward: int, maximum number of iterations for the fixed-point forward resolution.
    max_iter_backward: int, maximum number of iterations for the fixed-point backward resolution.
    anderson_beta: float, beta parameter for the Anderson acceleration.
    anderson_history_size: int, history size parameter for the Anderson acceleration.
    early_stop: bool, if True, stop the fixed-point resolution when the relative error is below crit_conv.
    crit_conv: float, relative error criterion for the fixed-point resolution.
    verbose: bool, if True, print the relative error at each iteration of the fixed-point resolution.
    '''
    def __init__(self, iterator, PnP_module, anderson_acceleration=False, max_iter_forward=50, max_iter_backward=50, 
                anderson_beta=1., anderson_history_size=5, early_stop=True, crit_conv=1-5, verbose=False) :
        super().__init__()
        self.iterator = iterator
        self.max_iter_backward = max_iter_backward
        self.anderson_acceleration = anderson_acceleration
        if anderson_acceleration :
            self.anderson_beta = anderson_beta
            self.anderson_history_size = anderson_history_size
            self.forward_FP = AndersonAcceleration(iterator, max_iter=max_iter_forward, history_size=anderson_history_size, beta=anderson_beta,
                            early_stop=early_stop, crit_conv=crit_conv, verbose=verbose)
        else :
            self.forward_FP = FixedPoint(iterator, max_iter=max_iter_forward, early_stop=early_stop, crit_conv=crit_conv, verbose=verbose)
        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose
        
        self.parameters = PnP_module.parameters

    def forward(self,x,physics):
        with torch.no_grad():
            z = self.forward_FP(x, x, physics)
        z =  self.iterator(z, 0, x, physics)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.iterator(z0, 0, x, physics)
        def backward_hook(grad):
            iterator = lambda x,it: torch.autograd.grad(f0, z0, x, retain_graph=True)[0] + grad
            if self.anderson_acceleration :
                backward_FP = AndersonAcceleration(iterator, max_iter=self.max_iter_backward, history_size=self.anderson_history_size, beta=self.anderson_beta,
                            early_stop=self.early_stop, crit_conv=self.crit_conv, verbose=self.verbose)
            else :
                backward_FP = FixedPoint(iterator, max_iter=self.max_iter_backward, early_stop=self.early_stop, crit_conv=self.crit_conv, verbose=self.verbose)
            g = backward_FP(grad)
            return g
        if z.requires_grad:
            z.register_hook(backward_hook)
        return z