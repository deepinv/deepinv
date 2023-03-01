import torch
import torch.nn as nn
from deepinv.optim.utils import check_conv

class FixedPoint(nn.Module):
    '''
    '''
    def __init__(self, iterator, max_iter=50, early_stop=True, crit_conv=None, use_anderson=False, 
                anderson_m = 5, anderson_lamb=1e-4, anderson_beta = 1., verbose=False) :
        super().__init__()
        self.iterator = iterator
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.early_stop = early_stop
        self.use_anderson = use_anderson
        if self.use_anderson : 
            solver = AndersonExp(m = anderson_m, lamb = anderson_lamb, max_iter=max_iter, beta = anderson_beta, tol = crit_conv)

    def forward(self, init, *args):
        x = init
        for it in range(self.max_iter):
            x_prev = x.clone() if type(x) is not tuple else x[0]
            x = self.iterator(x, it, *args)
            x_out = x if type(x) is not tuple else x[0]
            if self.early_stop and check_conv(x_prev, x_out, it, self.crit_conv, self.verbose):
                    break
        return x_out



class AndersonExp(nn.Module):
    """ Anderson acceleration for fixed point iteration. """
    def __init__(self, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
        super(AndersonExp, self).__init__()
        self.m = m
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta

    def forward(self, f, x0):
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, self.m, d * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, self.m, d * H * W, dtype=x0.dtype, device=x0.device)
        X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
        X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape)).reshape(bsz, -1)

        H = torch.zeros(bsz, self.m + 1, self.m + 1, dtype=x0.dtype, device=x0.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, self.m + 1, 1, dtype=x0.dtype, device=x0.device)
        y[:, 0] = 1

        current_k = 0
        for k in range(2, self.max_iter):
            current_k = k
            n = min(k, self.m)
            G = F[:, :n] - X[:, :n]
            H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + self.lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
                None]
            alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

            X[:, k % self.m] = self.beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - self.beta) * (alpha[:, None] @ X[:, :n])[:, 0]
            F[:, k % self.m] = f(X[:, k % self.m].reshape(x0.shape)).reshape(bsz, -1)
            res = (F[:, k % self.m] - X[:, k % self.m]).norm().item() / (1e-5 + F[:, k % self.m].norm().item())

            if (res < self.tol):
                break
        # tt += bsz
        return X[:, current_k % self.m].view_as(x0), res

