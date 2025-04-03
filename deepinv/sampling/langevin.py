import torch.nn as nn
import torch
import numpy as np
import time as time

import deepinv.optim
from tqdm import tqdm
from deepinv.optim.utils import check_conv
from deepinv.sampling.utils import Welford, projbox, refl_projbox


class MonteCarlo(nn.Module):
    r"""
    Base class for Monte Carlo sampling.

    This class can be used to create new Monte Carlo samplers, by only defining their kernel inside a torch.nn.Module:

    ::

        # define custom sampling kernel (possibly a Markov kernel which depends on the previous sample).
        class MyKernel(torch.torch.nn.Module):
            def __init__(self, iterator_params):
                super().__init__()
                self.iterator_params = iterator_params

            def forward(self, x, y, physics, likelihood, prior):
                # run one sampling kernel iteration
                new_x = f(x, y, physics, likelihood, prior, self.iterator_params)
                return new_x

        class MySampler(MonteCarlo):
            def __init__(self, prior, data_fidelity, iterator_params,
                         max_iter=1e3, burnin_ratio=.1, clip=(-1,2), verbose=True):
                # generate an iterator
                iterator = MyKernel(step_size=step_size, alpha=alpha)
                # set the params of the base class
                super().__init__(iterator, prior, data_fidelity, max_iter=max_iter,
                                 burnin_ratio=burnin_ratio, clip=clip, verbose=verbose)

        # create the sampler
        sampler = MySampler(prior, data_fidelity, iterator_params)

        # compute posterior mean and variance of reconstruction of measurement y
        mean, var = sampler(y, physics)


    This class computes the mean and variance of the chain using Welford's algorithm, which avoids storing the whole
    Monte Carlo samples.

    :param deepinv.optim.ScorePrior prior: negative log-prior based on a trained or model-based denoiser.
    :param deepinv.optim.DataFidelity data_fidelity: negative log-likelihood function linked with the
        noise distribution in the acquisition physics.
    :param int max_iter: number of Monte Carlo iterations.
    :param int thinning: thins the Monte Carlo samples by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
        samples to compute posterior statistics).
    :param float burnin_ratio: percentage of iterations used for burn-in period, should be set between 0 and 1.
        The burn-in samples are discarded constant with a numerical algorithm.
    :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
        If ``None``, the algorithm will not project the samples.
    :param float crit_conv: Threshold for verifying the convergence of the mean and variance estimates.
    :param Callable g_statistic: The sampler will compute the posterior mean and variance
        of the function g_statistic. By default, it is the identity function (lambda x: x),
        and thus the sampler computes the posterior mean and variance.
    :param bool verbose: prints progress of the algorithm.

    """

    def __init__(
        self,
        iterator: torch.nn.Module,
        prior: deepinv.optim.ScorePrior,
        data_fidelity: deepinv.optim.DataFidelity,
        max_iter=1e3,
        burnin_ratio=0.2,
        thinning=10,
        clip=(-1.0, 2.0),
        thresh_conv=1e-3,
        crit_conv="residual",
        save_chain=False,
        g_statistic=lambda x: x,
        verbose=False,
    ):
        super(MonteCarlo, self).__init__()

        self.iterator = iterator
        self.prior = prior
        self.likelihood = data_fidelity
        self.C_set = clip
        self.thinning = thinning
        self.max_iter = int(max_iter)
        self.thresh_conv = thresh_conv
        self.crit_conv = crit_conv
        self.burnin_iter = int(burnin_ratio * max_iter)
        self.verbose = verbose
        self.mean_convergence = False
        self.var_convergence = False
        self.g_function = g_statistic
        self.save_chain = save_chain
        self.chain = []

    def forward(self, y, physics, seed=None, x_init=None):
        r"""
        Runs an Monte Carlo chain to obtain the posterior mean and variance of the reconstruction of the measurements y.

        :param torch.Tensor y: Measurements
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param float seed: Random seed for generating the Monte Carlo samples
        :return: (tuple of torch.tensor) containing the posterior mean and variance.
        """
        with torch.no_grad():
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)

            # Algorithm parameters
            if self.C_set:
                C_lower_lim = self.C_set[0]
                C_upper_lim = self.C_set[1]

            # Initialization
            if x_init is None:
                x = physics.A_adjoint(y)
            else:
                x = x_init

            # Monte Carlo loop
            start_time = time.time()
            statistics = Welford(self.g_function(x))

            self.mean_convergence = False
            self.var_convergence = False
            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                x = self.iterator(
                    x, y, physics, likelihood=self.likelihood, prior=self.prior
                )

                if self.C_set:
                    x = projbox(x, C_lower_lim, C_upper_lim)

                if it >= self.burnin_iter and (it % self.thinning) == 0:
                    if it >= (self.max_iter - self.thinning):
                        mean_prev = statistics.mean().clone()
                        var_prev = statistics.var().clone()
                    statistics.update(self.g_function(x))

                    if self.save_chain:
                        self.chain.append(x.clone())

            if self.verbose:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                elapsed = end_time - start_time
                print(
                    f"Monte Carlo sampling finished! elapsed time={elapsed:.2f} seconds"
                )

            if (
                check_conv(
                    {"est": (mean_prev,)},
                    {"est": (statistics.mean(),)},
                    it,
                    self.crit_conv,
                    self.thresh_conv,
                    self.verbose,
                )
                and it > 1
            ):
                self.mean_convergence = True

            if (
                check_conv(
                    {"est": (var_prev,)},
                    {"est": (statistics.var(),)},
                    it,
                    self.crit_conv,
                    self.thresh_conv,
                    self.verbose,
                )
                and it > 1
            ):
                self.var_convergence = True

        return statistics.mean(), statistics.var()

    def get_chain(self):
        r"""
        Returns the thinned Monte Carlo samples (after burn-in iterations).
        Requires ``save_chain=True``.
        """
        return self.chain

    def reset(self):
        r"""
        Resets the Markov chain.
        """
        self.chain = []
        self.mean_convergence = False
        self.var_convergence = False

    def mean_has_converged(self):
        r"""
        Returns a boolean indicating if the posterior mean verifies the convergence criteria.
        """
        return self.mean_convergence

    def var_has_converged(self):
        r"""
        Returns a boolean indicating if the posterior variance verifies the convergence criteria.
        """
        return self.var_convergence


class ULAIterator(nn.Module):
    r"""
    Single iteration of the Unadjusted Langevin Algorithm.

    :param float step_size: step size :math:`\eta>0` of the algorithm.
    :param float alpha: regularization parameter :math:`\alpha`.
    :param float sigma: noise level used in the plug-and-play prior denoiser.
    """

    def __init__(self, step_size, alpha, sigma):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.noise_std = np.sqrt(2 * step_size)
        self.sigma = sigma

    def forward(self, x, y, physics, likelihood, prior):
        noise = torch.randn_like(x) * self.noise_std
        lhood = -likelihood.grad(x, y, physics)
        lprior = -prior.grad(x, self.sigma) * self.alpha
        return x + self.step_size * (lhood + lprior) + noise


class ULA(MonteCarlo):
    r"""
    Projected Plug-and-Play Unadjusted Langevin Algorithm.

    The algorithm runs the following markov chain iteration
    (Algorithm 2 from https://arxiv.org/abs/2103.04715):

    .. math::

        x_{k+1} = \Pi_{[a,b]} \left(x_{k} + \eta \nabla \log p(y|A,x_k) +
        \eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1} \right).

    where :math:`x_{k}` is the :math:`k` th sample of the Markov chain,
    :math:`\log p(y|x)` is the log-likelihood function, :math:`\log p(x)` is the log-prior,
    :math:`\eta>0` is the step size, :math:`\alpha>0` controls the amount of regularization,
    :math:`\Pi_{[a,b]}(x)` projects the entries of :math:`x` to the interval :math:`[a,b]` and
    :math:`z\sim \mathcal{N}(0,I)` is a standard Gaussian vector.


    - Projected PnP-ULA assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, ULA required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`


    :param deepinv.optim.ScorePrior, torch.nn.Module prior: negative log-prior based on a trained or model-based denoiser.
    :param deepinv.optim.DataFidelity, torch.nn.Module data_fidelity: negative log-likelihood function linked with the
        noise distribution in the acquisition physics.
    :param float step_size: step size :math:`\eta>0` of the algorithm.
        Tip: use :func:`deepinv.physics.LinearPhysics.compute_norm` to compute the Lipschitz constant of a linear forward operator.
    :param float sigma: noise level used in the plug-and-play prior denoiser. A larger value of sigma will result in
        a more regularized reconstruction.
    :param float alpha: regularization parameter :math:`\alpha`
    :param int max_iter: number of Monte Carlo iterations.
    :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
        samples to compute posterior statistics).
    :param float burnin_ratio: percentage of iterations used for burn-in period, should be set between 0 and 1.
        The burn-in samples are discarded constant with a numerical algorithm.
    :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
        If ``None``, the algorithm will not project the samples.
    :param float crit_conv: Threshold for verifying the convergence of the mean and variance estimates.
    :param Callable g_statistic: The sampler will compute the posterior mean and variance
        of the function g_statistic. By default, it is the identity function (lambda x: x),
        and thus the sampler computes the posterior mean and variance.
    :param bool verbose: prints progress of the algorithm.

    """

    def __init__(
        self,
        prior,
        data_fidelity,
        step_size=1.0,
        sigma=0.05,
        alpha=1.0,
        max_iter=1e3,
        thinning=5,
        burnin_ratio=0.2,
        clip=(-1.0, 2.0),
        thresh_conv=1e-3,
        save_chain=False,
        g_statistic=lambda x: x,
        verbose=False,
    ):
        iterator = ULAIterator(step_size=step_size, alpha=alpha, sigma=sigma)
        super().__init__(
            iterator,
            prior,
            data_fidelity,
            max_iter=max_iter,
            thresh_conv=thresh_conv,
            g_statistic=g_statistic,
            burnin_ratio=burnin_ratio,
            clip=clip,
            thinning=thinning,
            save_chain=save_chain,
            verbose=verbose,
        )


class SKRockIterator(nn.Module):
    def __init__(self, step_size, alpha, inner_iter, eta, sigma):
        super().__init__()
        self.step_size = step_size
        self.alpha = alpha
        self.eta = eta
        self.inner_iter = inner_iter
        self.noise_std = np.sqrt(2 * step_size)
        self.sigma = sigma

    def forward(self, x, y, physics, likelihood, prior):
        posterior = lambda u: likelihood.grad(u, y, physics) + self.alpha * prior.grad(
            u, self.sigma
        )

        # First kind Chebyshev function
        T_s = lambda s, u: np.cosh(s * np.arccosh(u))
        # First derivative Chebyshev polynomial first kind
        T_prime_s = lambda s, u: s * np.sinh(s * np.arccosh(u)) / np.sqrt(u**2 - 1)

        w0 = 1 + self.eta / (self.inner_iter**2)  # parameter \omega_0
        w1 = T_s(self.inner_iter, w0) / T_prime_s(
            self.inner_iter, w0
        )  # parameter \omega_1
        mu1 = w1 / w0  # parameter \mu_1
        nu1 = self.inner_iter * w1 / 2  # parameter \nu_1
        kappa1 = self.inner_iter * (w1 / w0)  # parameter \kappa_1

        # sampling the variable x
        noise = np.sqrt(2 * self.step_size) * torch.randn_like(x)  # diffusion term

        # first internal iteration (s=1)
        xts_2 = x.clone()
        xts = (
            x.clone()
            - mu1 * self.step_size * posterior(x + nu1 * noise)
            + kappa1 * noise
        )

        for js in range(
            2, self.inner_iter + 1
        ):  # s=2,...,self.inner_iter SK-ROCK internal iterations
            xts_1 = xts.clone()
            mu = 2 * w1 * T_s(js - 1, w0) / T_s(js, w0)  # parameter \mu_js
            nu = 2 * w0 * T_s(js - 1, w0) / T_s(js, w0)  # parameter \nu_js
            kappa = 1 - nu  # parameter \kappa_js
            xts = -mu * self.step_size * posterior(xts) + nu * xts + kappa * xts_2
            xts_2 = xts_1

        return xts  # new sample produced by the SK-ROCK algorithm


class SKRock(MonteCarlo):
    r"""
    Plug-and-Play SKROCK algorithm.

    Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
    approximation to accelerate the standard Unadjusted Langevin Algorithm.

    The algorithm was introduced in "Accelerating proximal Markov chain Monte Carlo by using an explicit stabilised method"
    by L. Vargas, M. Pereyra and K. Zygalakis (https://arxiv.org/abs/1908.08845)

    - SKROCK assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, SKROCK required step_size smaller than :math:`\frac{1}{L+\|A\|_2^2}`

    :param deepinv.optim.ScorePrior, torch.nn.Module prior: negative log-prior based on a trained or model-based denoiser.
    :param deepinv.optim.DataFidelity, torch.nn.Module data_fidelity: negative log-likelihood function linked with the
        noise distribution in the acquisition physics.
    :param float step_size: Step size of the algorithm. Tip: use physics.lipschitz to compute the Lipschitz
    :param float eta: :math:`\eta` SKROCK damping parameter.
    :param float alpha: regularization parameter :math:`\alpha`.
    :param int inner_iter: Number of inner SKROCK iterations.
    :param int max_iter: Number of outer iterations.
    :param int thinning: Thins the Markov Chain by an integer :math:`\geq 1` (i.e., keeping one out of ``thinning``
        samples to compute posterior statistics).
    :param float burnin_ratio: percentage of iterations used for burn-in period. The burn-in samples are discarded
        constant with a numerical algorithm.
    :param tuple clip: Tuple containing the box-constraints :math:`[a,b]`.
        If ``None``, the algorithm will not project the samples.
    :param bool verbose: prints progress of the algorithm.
    :param float sigma: noise level used in the plug-and-play prior denoiser. A larger value of sigma will result in
        a more regularized reconstruction.
    :param Callable g_statistic: The sampler will compute the posterior mean and variance
        of the function g_statistic. By default, it is the identity function (lambda x: x),
        and thus the sampler computes the posterior mean and variance.

    """

    def __init__(
        self,
        prior: deepinv.optim.ScorePrior,
        data_fidelity,
        step_size=1.0,
        inner_iter=10,
        eta=0.05,
        alpha=1.0,
        max_iter=1e3,
        burnin_ratio=0.2,
        thinning=10,
        clip=(-1.0, 2.0),
        thresh_conv=1e-3,
        save_chain=False,
        g_statistic=lambda x: x,
        verbose=False,
        sigma=0.05,
    ):
        iterator = SKRockIterator(
            step_size=step_size,
            alpha=alpha,
            inner_iter=inner_iter,
            eta=eta,
            sigma=sigma,
        )
        super().__init__(
            iterator,
            prior,
            data_fidelity,
            max_iter=max_iter,
            thresh_conv=thresh_conv,
            thinning=thinning,
            burnin_ratio=burnin_ratio,
            clip=clip,
            g_statistic=g_statistic,
            save_chain=save_chain,
            verbose=verbose,
        )


# if __name__ == "__main__":
#     import deepinv as dinv
#     import torchvision
#     from deepinv.optim.data_fidelity import L2
#
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(dinv.device) / 255
#     # physics = dinv.physics.CompressedSensing(m=50000, fast=True, img_shape=(3, 218, 178), device=dinv.device)
#     # physics = dinv.physics.Denoising()
#     physics = dinv.physics.Inpainting(
#         mask=0.95, tensor_size=(3, 218, 178), device=dinv.device
#     )
#     # physics = dinv.physics.BlurFFT(filter=dinv.physics.blur.gaussian_blur(sigma=(2,2)), img_size=x.shape[1:], device=dinv.device)
#
#     sigma = 0.1
#     physics.noise_model = dinv.physics.GaussianNoise(sigma)
#
#     y = physics(x)
#
#     likelihood = L2(sigma=sigma)
#
#     # model_spec = {'name': 'median_filter', 'args': {'kernel_size': 3}}
#     model_spec = {
#         "name": "dncnn",
#         "args": {
#             "device": dinv.device,
#             "in_channels": 3,
#             "out_channels": 3,
#             "pretrained": "download_lipschitz",
#         },
#     }
#     # model_spec = {'name': 'waveletprior', 'args': {'wv': 'db8', 'level': 4, 'device': dinv.device}}
#
#     prior = ScorePrior(model_spec=model_spec, sigma_normalize=True)
#
#     sigma_den = 2 / 255
#     f = ULA(
#         prior,
#         likelihood,
#         max_iter=5000,
#         sigma=sigma_den,
#         burnin_ratio=0.3,
#         verbose=True,
#         alpha=0.3,
#         step_size=0.5 * 1 / (1 / (sigma**2) + 1 / (sigma_den**2)),
#         clip=(-1, 2),
#         save_chain=True,
#     )
#     # f = SKRock(prior, likelihood, max_iter=1000, burnin_ratio=.3, verbose=True,
#     #           alpha=.9, step_size=.1*(sigma**2), clip=(-1, 2))
#
#     xmean, xvar = f(y, physics)
#
#     print(str(f.mean_has_converged()))
#     print(str(f.var_has_converged()))
#
#     chain = f.get_chain()
#     distance = np.zeros((len(chain)))
#     for k, xhat in enumerate(chain):
#         dist = (xhat - xmean).pow(2).mean()
#         distance[k] = dist
#     distance = np.sort(distance)
#     thres = distance[int(len(distance) * 0.95)]  #
#     err = (x - xmean).pow(2).mean()
#     print(f"Confidence region: {thres:.2e}, error: {err:.2e}")
#
#     xstdn = xvar.sqrt()
#     xstdn_plot = xstdn.sum(dim=1).unsqueeze(1)
#
#     error = (xmean - x).abs()  # per pixel average abs. error
#     error_plot = error.sum(dim=1).unsqueeze(1)
#
#     print(f"Correct std: {(xstdn*3>error).sum()/np.prod(xstdn.shape)*100:.1f}%")
#
#     dinv.utils.plot(
#         [physics.A_adjoint(y), x, xmean, xstdn_plot, error_plot],
#         titles=["meas.", "ground-truth", "mean", "norm. std", "abs. error"],
#     )
