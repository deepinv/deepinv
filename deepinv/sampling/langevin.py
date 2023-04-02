import torch.nn as nn
import torch
import numpy as np
import time as time
from deepinv.models import ScoreDenoiser
from tqdm import tqdm


class Welford:
    r'''
     Welford's algorithm for calculating mean and variance

     https://doi.org/10.2307/1266577
    '''
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


class MCMC(nn.Module):
    r'''
    Base class for Markov Chain Monte Carlo sampling.

    ..
        class MyIterator(torch.nn.Module):
            def __init__(self, iterator_params)
                super().__init__()
                self.iterator_params = iterator_params

            def forward(self, x):
                # run one sampling kernel iteration
                new_x = f(x, iterator_params)
                return new_x

        class MySampler(MCMC):
            def __init__(self, prior, data_fidelity, iterator_params, max_iter=1e3, burnin_ratio=.1, clip=(-1,2), verbose=True):
                # generate an iterator
                iterator = myIterator(step_size=step_size, alpha=alpha)
                super().__init__(iterator, prior, data_fidelity, alpha=alpha,  max_iter=max_iter,
                                 burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)

        # create the sampler
        sampler = MySampler(...)

        # compute posterior mean and variance of reconstruction of measurement y
        mean, var = sampler(y, physics)

    '''
    def __init__(self, iterator:torch.nn.Module, prior:ScoreDenoiser, data_fidelity, max_iter=1e3, burnin_ratio=.2,
                 clip=(-1., 2.), verbose=False):
        super(MCMC, self).__init__()

        self.iterator = iterator
        self.prior = prior
        self.likelihood = data_fidelity
        self.C_set = clip
        self.max_iter = int(max_iter)
        self.burnin_iter = int(burnin_ratio*max_iter)
        self.verbose = verbose

    def forward(self, y, physics, seed=None):
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            # Algorithm parameters
            if self.C_set:
                C_lower_lim = self.C_set[0]
                C_upper_lim = self.C_set[1]

            # Initialization
            x = physics.A_adjoint(y) #.cuda(device).detach().clone()

            # MCMC loop
            start_time = time.time()
            statistics = Welford(x)

            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                x = self.iterator(x, y, physics, likelihood=self.likelihood,
                                  prior=self.prior)

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


class ULAIterator(nn.Module):
    def __init__(self, step_size, alpha):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.noise_std = np.sqrt(2*step_size)

    def forward(self, x, y, physics, likelihood, prior):
        noise = torch.randn_like(x)*self.noise_std
        lhood = - likelihood.grad(x, y, physics)
        lprior = - prior(x) * self.alpha
        return x + self.step_size * (lhood+lprior) + noise


class ULA(MCMC):
    r'''
        Plug-and-Play Unadjusted Langevin Algorithm.

        The algorithm runs the following markov chain iteration
        https://arxiv.org/abs/2103.04715 :

        .. math::

            x_{k+1} = \Pi_{[a,b]} \left(x_{k} + \eta \nabla \log p(y|A,x_k) +
            \eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1} \right).

        where :math:`x_{k}` is the :math:`k`th sample of the Markov chain,
        :math:`\log p(y|x)` is the log-likelihood function, :math:`\log p(x)` is the log-prior
        :math:`\eta>0` is the step size, :math:`\alpha>0` controls the amount of regularization,
        :math:`\Pi_{[a,b]}(x)` projects the entries of :math:`x` to the interval :math:`[a,b]` and
        :math:`z\sim \mathcal{N}(0,I)` is a standard Gaussian vector.


        - PnP-ULA assumes that the denoiser is :math:`L`-Lipschitz differentiable
        - For convergence, PnP-ULA required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`

        :param deepinv.models.ScoreDenoiser prior: negative log-prior based on a trained or model-based denoiser.
        :param deepinv.optim.DataFidelity data_fidelity: negative log-likelihood function linked with the
            noise distribution in the acquisition physics.
        :param float step_size: step size :math:`\eta>0` of the algorithm.
            Tip: use :meth:`deepinv.physics.Physics.compute_norm()` to compute the Lipschitz constant of the forward operator.
        :param float alpha: regularization parameter :math:`\alpha`
        :param int max_iter: number of Monte Carlo iterations.
        :param float burnin_ratio: percentage of iterations used for burn-in period, should be set between 0 and 1.
            The burn-in samples are discarded constant with a numerical algorithm.
        :param tuple clip: Tuple containing the box-constraints :math:`\[a,b]`. If None, the algorithm will not project.
        :param bool verbose: prints progress of the algorithm.

    '''
    def __init__(self, prior, data_fidelity, step_size=1., alpha=1.,  max_iter=1e3, burnin_ratio=.2,
                 clip=(-1.,2.), verbose=False):

        iterator = ULAIterator(step_size=step_size, alpha=alpha)
        super().__init__(iterator, prior, data_fidelity, max_iter=max_iter,
                         burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)


class SKRockIterator(nn.Module):
    def __init__(self, step_size, alpha, inner_iter, eta):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.eta = eta
        self.inner_iter = inner_iter
        self.noise_std = np.sqrt(2*step_size)

    def forward(self, x, y, physics, likelihood, prior):

        posterior = lambda u:  likelihood.grad(u, y, physics) \
                               + self.alpha * prior(u)

        # First kind Chebyshev function
        T_s = lambda s, u: np.cosh(s*np.arccosh(u))
        # First derivative Chebyshev polynomial first kind
        T_prime_s = lambda s, u: s*np.sinh(s*np.arccosh(u))/np.sqrt(u**2-1)

        w0 = 1 + self.eta/(self.inner_iter**2)  # parameter \omega_0
        w1 = T_s(self.inner_iter, w0)/T_prime_s(self.inner_iter, w0)  # parameter \omega_1
        mu1 = w1/w0  # parameter \mu_1
        nu1 = self.inner_iter*w1/2  # parameter \nu_1
        kappa1 = self.inner_iter*(w1/w0)  # parameter \kappa_1

        # sampling the variable x
        noise = np.sqrt(2*self.step_size)*torch.randn_like(x)  # diffusion term

        # first internal iteration (s=1)
        xts_2 = x.clone()
        xts = x.clone() - mu1*self.step_size*posterior(x + nu1*noise) + kappa1*noise

        for js in range(2, self.inner_iter+1):  # s=2,...,self.inner_iter SK-ROCK internal iterations
            xts_1 = xts.clone()
            mu = 2 * w1 * T_s(js-1, w0) / T_s(js, w0)  # parameter \mu_js
            nu = 2 * w0 * T_s(js-1, w0) / T_s(js, w0)  # parameter \nu_js
            kappa = 1-nu  # parameter \kappa_js
            xts = -mu * self.step_size*posterior(xts) + nu*xts + kappa*xts_2
            xts_2 = xts_1

        return xts  # new sample produced by the SK-ROCK algorithm


class SKRock(MCMC):
    r'''
        Plug-and-Play SKROCK algorithm.

        Obtains samples of the posterior distribution

        .. math::

            - \log p(x|y,A) \propto f(y,A(x))+\alpha g(x)

        where :math:`x` is the image to be reconstructed, :math:`y` are the measurements.
        :math:`f(y,A(x))` is the negative log-likelihood and :math:`g(x)` is the negative log-prior

        https://arxiv.org/abs/1908.08845

        The step size should be chosen smaller than

        .. math:: \frac{c}{L_f + L_g}

        where

         - For convergence, PnPULA required step_size smaller than :math:`{1}\frac{\|A|\_2^2}`
         - PnPULA assumes that the denoiser is :math:`L`-Lipschitz differentiable

        :param deepinv.models.ScoreDenoiser prior: negative log-prior based on a trained or model-based denoiser.
        :param deepinv.optim.DataFidelity data_fidelity: negative log-likelihood function linked with the
            noise distribution in the acquisition physics.
        :param float step_size: Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
        :param float alpha: regularization parameter :math:`\alpha`
        :param int max_iter: Number of iterations
        :param float burnin_ratio: percentage of iterations used for burn-in period. The burn-in samples are discarded
            constant with a numerical algorithm.
        :param tuple clip: Tuple containing the box-constraints :math:`\[a,b]`. If None, the algorithm will not project.
        :param bool verbose: prints progress of the algorithm

    '''
    def __init__(self, prior: ScoreDenoiser, data_fidelity, step_size=1., inner_iter=10, eta=0.05, alpha=1.,  max_iter=1e3, burnin_ratio=.2,
                 clip=(-1., 2.), verbose=False):

        iterator = SKRockIterator(step_size=step_size, alpha=alpha, inner_iter=inner_iter, eta=eta)
        super().__init__(iterator, prior, data_fidelity, max_iter=max_iter,
                         burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)


if __name__ == "__main__":

    import deepinv as dinv
    import torchvision
    from deepinv.optim.data_fidelity import L2

    x = torchvision.io.read_image('../../datasets/celeba/img_align_celeba/085307.jpg')
    x = x.unsqueeze(0).float().to(dinv.device) / 255
    #physics = dinv.physics.CompressedSensing(m=50000, fast=True, img_shape=(3, 218, 178), device=dinv.device)
    #physics = dinv.physics.Denoising()
    physics = dinv.physics.Inpainting(mask=.95, tensor_size=(3, 218, 178), device=dinv.device)
    #physics = dinv.physics.BlurFFT(filter=dinv.physics.blur.gaussian_blur(sigma=(2,2)), img_size=x.shape[1:], device=dinv.device)

    sigma = .1
    physics.noise_model = dinv.physics.GaussianNoise(sigma)

    y = physics(x)

    likelihood = L2(sigma=sigma)

    model_spec = {'name': 'median_filter', 'args': {'kernel_size': 3}}
    #model_spec = {'name': 'waveletprior', 'args': {'wv': 'db8', 'level': 4, 'device': dinv.device}}
    #model_spec = {'name': 'dncnn', 'args': {'device': dinv.device, 'in_channels': 3, 'out_channels': 3,
    #                                        'pretrained': 'download_lipschitz'}}

    prior = ScoreDenoiser(model_spec=model_spec, sigma_denoiser=2/255)

    #f = ULA(prior, likelihood, max_iter=10000, burnin_ratio=.3, verbose=True,
    #           alpha=.1, step_size=.1*(sigma**2), clip=(-1, 2))
    f = SKRock(prior, likelihood, max_iter=100, burnin_ratio=.3, verbose=True,
               alpha=.1, step_size=.1*(sigma**2), clip=(-1, 2))

    xmean, xvar = f(y, physics)

    xnstd = xvar.sqrt()
    xnstd = xnstd/xnstd.flatten().max()

    dinv.utils.plot_debug([physics.A_adjoint(y), x, xmean, xnstd], titles=['meas.', 'ground-truth', 'mean', 'norm. std'])
