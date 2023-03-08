import torch
import torch.nn as nn

from deepinv.optim.fixed_point import FixedPoint, AndersonAcceleration


class Unfolded(nn.Module):
    '''
    Unfolded module
    '''

    def __init__(self, iterator, init=None, max_iter=50, crit_conv=1e-3, learn_stepsize=False, learn_g_param=False,
                 trainable=None, custom_g_step=None, custom_f_step=None, device=torch.device('cpu'), verbose=True,
                 constant_stepsize=False, constant_g_param=False,
                 anderson_acceleration=False, anderson_beta=1., anderson_history_size=5, early_stop=True,
                 deep_equilibrium=False, max_iter_backward=50):
        super(Unfolded, self).__init__()

        self.early_stop = early_stop
        self.crit_conv = crit_conv
        self.verbose = verbose

        # model parameters
        self.iterator = iterator
        if trainable is not None:
            self.trainable = trainable

        self.deep_equilibrium = deep_equilibrium
        if self.deep_equilibrium:  # DEQ requires a "real" constant fixed-point operator
            constant_stepsize = True
            constant_g_param = True
            self.max_iter_backward = max_iter_backward

        if learn_stepsize:
            if constant_stepsize:
                self.step_size = nn.Parameter(torch.tensor(iterator.stepsize[0], device=device))
                self.stepsize_list = [self.step_size] * max_iter
            else:
                self.stepsize_list = nn.ParameterList([nn.Parameter(torch.tensor(iterator.stepsize[i], device=device))
                                                       for i in range(max_iter)])
            self.iterator.stepsize = self.stepsize_list
        if learn_g_param:
            if constant_g_param:
                self.g_param = nn.Parameter(torch.tensor(iterator.g_param[0], device=device))
                self.g_param_list = [self.g_param] * max_iter
            else:
                self.g_param_list = nn.ParameterList([nn.Parameter(torch.tensor(iterator.g_param[i], device=device))
                                                      for i in range(max_iter)])
            self.iterator.g_param = self.g_param_list

        if custom_g_step is not None:
            self.iterator.g_step = custom_g_step
        # else:
        #     self.iterator.g_step = self.iterator._g_step

        if custom_f_step is not None:
            self.iterator.f_step = custom_f_step
        # else:
        #     self.iterator.f_step = self.iterator._f_step


        # fixed-point iterations
        self.anderson_acceleration = anderson_acceleration
        if anderson_acceleration:
            self.anderson_beta = anderson_beta
            self.anderson_history_size = anderson_history_size
            self.forward_FP = AndersonAcceleration(self.iterator, max_iter=max_iter, history_size=anderson_history_size,
                                                   beta=anderson_beta,
                                                   early_stop=early_stop, crit_conv=crit_conv, verbose=verbose)
        else:
            self.forward_FP = FixedPoint(self.iterator, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv,
                                         verbose=verbose)

    def get_init(self, y, physics):
        return physics.A_adjoint(y), y

    def get_primal_variable(self, x):
        return x[0]

    def forward(self, y, physics, **kwargs):
        x = self.get_init(y, physics)
        if self.deep_equilibrium:
            with torch.no_grad():
                x = self.forward_FP(x, y, physics, **kwargs)[0]
            x = self.iterator(x, 0, y, physics)[0]
            x0 = x.clone().detach().requires_grad_()
            f0 = self.iterator(x0, 0, y, physics)[0]

            def backward_hook(grad):
                grad = (grad,)
                iterator = lambda y, it: (torch.autograd.grad(f0, x0, y, retain_graph=True)[0] + grad[0],)
                if self.anderson_acceleration:
                    backward_FP = AndersonAcceleration(iterator, max_iter=self.max_iter_backward,
                                                       history_size=self.anderson_history_size, beta=self.anderson_beta,
                                                       early_stop=self.early_stop, crit_conv=self.crit_conv,
                                                       verbose=self.verbose)
                else:
                    backward_FP = FixedPoint(iterator, max_iter=self.max_iter_backward, early_stop=self.early_stop,
                                             crit_conv=self.crit_conv, verbose=self.verbose)
                g = backward_FP(grad)[0]
                return g

            if x.requires_grad:
                x.register_hook(backward_hook)
        else:
            x = self.forward_FP(x, y, physics, **kwargs)
            x = self.get_primal_variable(x)
        return x
