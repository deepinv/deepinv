import torch.nn as nn
import torch
import numpy as np
import time as time


# Welford's algorithm for calculating mean and variance
# https://doi.org/10.2307/1266577
class Welford:
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self):
        return self.M

    def var(self):
        return self.S / (self.k - 1)


def refl_projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=lower, max=upper)


class MYULA(nn.Module):
    def __init__(self, denoiser, sigma=None, max_iter=1e3, burnin_ratio=.1, step_size=1, alpha=0.9, clip=(0, 1), test_mode=True, verbose=False):
        '''
                 Moreau Yosida Unadjusted Langevin Algorithm
                 - For convergence, its required step_size <= 1/ ||A||_2^2
                 - Assumes that the denoiser is L-Lipschitz differentiable

        :param denoiser: Denoiser network
        :param sigma: Noise level to run diffusion
        :param max_iter: Number of iterations
        :param burnin_ratio: percentage of iterations used for burn-in period. The burn-in samples are discarded
        :param step_size: Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
        constant with a numerical algorithm.
        :param alpha: Momentum parameter
        :param clip: Tuple of box-constraints for the generated samples. e.g., images
         should be normalised between (0,1). If None, the algorithm will not project.
        :param test_mode: Enables torch.grad at train time and disable it at test time
        :param verbose: prints progress of the algorithm
        '''
        super(MYULA, self).__init__()

        self.denoiser = denoiser
        self.sigma = sigma
        self.C_set = clip
        self.max_iter = int(max_iter)
        self.step_size = step_size
        self.step_size = alpha
        self.burnin_iter = int(burnin_ratio*max_iter)
        self.test_mode = test_mode
        self.verbose = verbose

    def forward(self, y, physics, seed=None):
        def grad(z):
            return (1 / self.sigma ** 2) * physics.A_adjoint(physics.A(z) - y)

        with torch.set_grad_enabled(not self.test_mode):  # Enable grad at train time and disable it at test time

            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            # Algorithm parameters
            delta = self.stepsize
            C_upper_lim = self.C_set[1]
            C_lower_lim = self.C_set[0]

            # Initialization
            x = physics.A_adjoint(y) #.cuda(device).detach().clone()

            # PnP-ULA loop
            start_time = time.time()
            statistics = Welford(x)

            for it in range(self.max_iter):
                noise = np.sqrt(2 * delta) * torch.randn_like(x)
                momentum = self.alpha * delta / self.sigma * (self.denoiser(x) - x)
                x_new = x - delta * grad(x) + momentum + noise

                if self.C_set:
                    x = projbox(x_new, C_lower_lim, C_upper_lim)

                if self.verbose:
                    print(f'iteration {it} out of {self.max_iter}', '\r')

                if it < self.burnin_iter:
                    statistics.update(x)

            if self.verbose:
                torch.cuda.synchronize()
                end_time = time.time()
                elapsed = end_time - start_time
                print(f'MYULA finished! elapsed time={elapsed} seconds')

        return statistics.mean(), statistics.var()