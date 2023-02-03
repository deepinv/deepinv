import torch.nn as nn
import torch
import numpy as np
import time as time
from tqdm import tqdm


# Welford's algorithm for calculating mean and variance
# https://doi.org/10.2307/1266577
class Welford:
    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
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


class PnPULA(nn.Module):
    def __init__(self, denoiser, sigma=.1, max_iter=1e3, burnin_ratio=.1, step_size=1, alpha=1., clip=None, test_mode=True, verbose=False):
        '''
                 Moreau Yosida Unadjusted Langevin Algorithm
                 - For convergence, its required step_size <= 1/ ||A||_2^2
                 - Assumes that the denoiser is L-Lipschitz differentiable

        :param denoiser: Denoiser network
        :param sigma: Standard deviation of the noise in the measurements
        :param max_iter: Number of iterations
        :param burnin_ratio: percentage of iterations used for burn-in period. The burn-in samples are discarded
        :param step_size: Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
        constant with a numerical algorithm.
        :param alpha: (float) Regularization parameter
        :param clip: Tuple of box-constraints for the generated samples. e.g., images
         should be normalised between (0,1). If None, the algorithm will not project.
        :param test_mode: Enables torch.grad at train time and disable it at test time
        :param verbose: prints progress of the algorithm
        '''
        super(PnPULA, self).__init__()

        self.denoiser = denoiser
        self.sigma = sigma
        self.C_set = clip
        self.max_iter = int(max_iter)
        self.step_size = step_size
        self.alpha = alpha
        self.burnin_iter = int(burnin_ratio*max_iter)
        self.test_mode = test_mode
        self.verbose = verbose

    def forward(self, y, physics, seed=None):
        def grad_likelihood(z):
            return (1 / self.sigma ** 2) * physics.A_adjoint(physics.A(z) - y)

        with torch.set_grad_enabled(not self.test_mode):  # Enable grad at train time and disable it at test time

            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            # Algorithm parameters
            delta = self.step_size

            if self.C_set:
                C_upper_lim = self.C_set[1]
                C_lower_lim = self.C_set[0]

            # Initialization
            x = physics.A_adjoint(y) #.cuda(device).detach().clone()

            # PnP-ULA loop
            start_time = time.time()
            statistics = Welford(x)

            deltasqrt = torch.sqrt(2 * delta)

            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                noise = deltasqrt * torch.randn_like(x)
                denoised = self.alpha * delta / self.sigma * (self.denoiser(x) - x)
                x = x - delta * grad_likelihood(x) + denoised + noise

                if self.C_set:
                    x = projbox(x, C_lower_lim, C_upper_lim)

                if it > self.burnin_iter:
                    statistics.update(x)

            if self.verbose:
                torch.cuda.synchronize()
                end_time = time.time()
                elapsed = end_time - start_time
                print(f'PnP ULA finished! elapsed time={elapsed} seconds')

        return statistics.mean(), statistics.var()