import torch
import torch.nn as nn
from deepinv.optim.utils import check_conv

class FixedPoint(nn.Module):
    '''
    Fixed-point iterations. 

        iterator : function that takes as input the current iterate and the iteration number and returns the next iterate.
        max_iter : maximum number of iterations. Default = 50 
        early_stop : if True, the acceleration stops when the convergence criterion is reached. Default = True
        crit_conv : stopping criterion.  Default = 1e-5
        verbose: if True, print the relative error at each iteration. Default = False
    '''
    def __init__(self, iterator=None, update_params_fn=None, max_iter=50, early_stop=True, crit_conv='residual', thres_conv=1e-5, verbose=False):
        super().__init__()
        self.iterator = iterator
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.thres_conv = thres_conv
        self.verbose = verbose
        self.early_stop = early_stop
        self.has_converged = False
        self.update_params_fn = update_params_fn

    def forward(self, x, init_params, *args, return_params=False, **kwargs):
        cur_params = init_params
        for it in range(self.max_iter):
            x_prev = x
            cur_prior = self.get_prior(it)
            x = self.iterator(x, cur_prior, cur_params, *args, **kwargs)
            if check_conv(x_prev, x, it, self.crit_conv, self.thres_conv, verbose=self.verbose) and it>1:
                self.has_converged = True
                if self.early_stop:
                    if self.verbose:
                        print('Convergence reached at iteration ', it)
                    break
            if it < self.max_iter - 1 and self.update_params_fn:
                cur_params = self.update_params_fn(it, x, x_prev)
        if return_params : 
            return x, cur_params
        else: 
            return x

    def get_prior(self, it):
        if isinstance(next(iter(self.iterator.prior.items()))[1], nn.ModuleList):
            prior_cur = {key: value[it] if len(value) > 1 else value[0]
                               for key, value in zip(self.iterator.prior.keys(), self.iterator.prior.values())}
        else:
            prior_cur = self.iterator.prior
        return prior_cur

class AndersonAcceleration(FixedPoint):
    '''
    Anderson Acceleration for accelerated fixed-point resolution. Strongly inspired from http://implicit-layers-tutorial.org/deep_equilibrium_models/. 
    Foward is called with init a tuple (x,) with x the initialization tensor of shape BxCxHxW and iterator optional arguments. 

    Args :
        iterator : function that takes as input the current iterate and the iteration number and returns the next iterate.
        history_size : size of the history used for the acceleration. Default = 5
        max_iter : maximum number of iterations. Default = 50 
        early_stop : if True, the acceleration stops when the convergence criterion is reached. Default = True
        crit_conv : stopping criterion.  Default = 1e-5
        ridge: ridge regularization in solver. Default = 1e-4
        beta: momentum in Anderson updates. Default = 1.
        verbose: if True, print the relative error at each iteration. Default = False
    '''
    def __init__(self, history_size=5, ridge=1e-4, beta=1.0, **kwargs):
        super(AndersonAcceleration, self).__init__(**kwargs)
        self.history_size = history_size
        if isinstance(beta, float):
            beta = [beta] * self.max_iter
        self.beta = beta
        self.ridge = ridge
        
    def forward(self, x, init_params, *args):
        cur_params = init_params
        init =  x['est'][0]
        B, C, H, W = init.shape
        X = torch.zeros(B, self.history_size, C * H * W, dtype=init.dtype, device=init.device)
        F = torch.zeros(B, self.history_size, C * H * W, dtype=init.dtype, device=init.device)
        X[:, 0] = init.reshape(B, -1)
        F[:, 0] = self.iterator(init, 0, *args)[0].reshape(B, -1)
        X[:, 1] = F[:, 0]
        x = self.iterator(F[:, 0].reshape(init.shape),1,*args)[0]
        F[:, 1] = x.reshape(B, -1)

        H = torch.zeros(B, self.history_size + 1, self.history_size + 1, dtype=init.dtype, device=init.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(B, self.history_size + 1, dtype=init.dtype, device=init.device)
        y[:, 0] = 1
        for it in range(2, self.max_iter):
            n = min(it, self.history_size)
            G = F[:, :n] - X[:, :n]
            H[:, 1:n+1, 1:n+1] = torch.bmm(G, G.transpose(1, 2)) + self.ridge * torch.eye(n, dtype=init.dtype, device=init.device).unsqueeze(0)
            alpha = torch.linalg.solve(H[:, :n+1, :n+1],y[:, :n+1])[:, 1:n+1]
            X[:, it % self.history_size] = self.beta[it] * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - self.beta[it]) * (alpha[:, None] @ X[:, :n])[:, 0]
            F[:, it % self.history_size] = self.iterator(X[:, it % self.history_size].reshape(init.shape), it, *args)[0].reshape(B, -1)
            x_prev = X[:, it % self.history_size].reshape(init.shape)
            x = F[:, it % self.history_size].reshape(init.shape)
            if check_conv(x_prev, x, it, self.crit_conv, self.thres_conv, verbose=self.verbose) and it>1:
                self.has_converged = True
                if self.early_stop:
                    if self.verbose:
                        print('Convergence reached at iteration ', it)
                    break
            if it < self.max_iter - 1 :
                cur_params = self.update_params(cur_params, it+1, x, x_prev)
        return (x,)


