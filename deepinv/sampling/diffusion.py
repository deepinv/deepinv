from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
from deepinv.models import Reconstructor, Denoiser
from deepinv.physics import Blur

import deepinv as dinv
from deepinv.sampling import BaseSampling
from deepinv.sampling.sampling_iterators import DiffusionIterator
from deepinv.sampling.diffusion_sde import (
    BaseSDE,
    EDMDiffusionSDE,
    PosteriorDiffusion,
    VariancePreservingDiffusion,
    VarianceExplodingDiffusion,
)
from deepinv.sampling.noisy_datafidelity import DPSDataFidelity
from deepinv.sampling.sde_solver import BaseSDESolver, EulerSolver


class DiffusionSampler(BaseSampling):
    r"""
    Turns a diffusion method into a Monte Carlo sampler.

    Unlike diffusion methods, the resulting sampler computes the mean and variance of the distribution
    by running the diffusion multiple times.

    See the docs for :class:`deepinv.sampling.BaseSampling` for more information. It uses the helper class :class:`deepinv.sampling.DiffusionIterator`.

    :param torch.nn.Module diffusion: a diffusion model
    :param int max_iter: the number of samples to generate
    :param tuple clip: the clip range
    :param Callable g_statistic: the algorithm computes mean and variance of the g function, by default :math:`g(x) = x`.
    :param float thres_conv: the convergence threshold for the mean and variance
    :param bool verbose: whether to print the progress
    :param bool save_chain: whether to save the chain
    :param int thinning: the thinning factor
    :param float burnin_ratio: the burnin ratio
    """

    def __init__(
        self,
        diffusion,
        max_iter=1e2,
        clip=(-1, 2),
        thres_conv=1e-1,
        g_statistic=lambda x: x,
        verbose=True,
        save_chain=False,
    ):
        # generate an iterator
        # set the params of the base class
        data_fidelity = None
        diffusion.verbose = False
        prior = diffusion
        iterator = DiffusionIterator(clip=clip)

        super().__init__(
            iterator,
            data_fidelity,
            prior,
            max_iter=max_iter,
            thinning=1,
            thresh_conv=thres_conv,
            history_size=save_chain,
            burnin_ratio=0.0,
            verbose=verbose,
            # thresh_conv=thres_conv,
        )
        self.g_statistics = [lambda d: g_statistic(d["x"])]

    def forward(self, y, physics, seed=None):
        r"""
        Runs the diffusion model to obtain the posterior mean and variance of the reconstruction of the measurements y.

        :param torch.Tensor y: Measurements
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param float seed: Random seed for generating the samples
        :return: (tuple of torch.Tensor) containing the posterior mean and variance.
        """
        return self.sample(y, physics, seed=seed, g_statistics=self.g_statistics)


class DDRM(Reconstructor):
    r"""
    Denoising Diffusion Restoration Models (DDRM).

    This class implements the Denoising Diffusion Restoration Model (DDRM) described in :footcite:t:`kawar2022denoising`.

    The DDRM is a sampling method that uses a denoiser to sample from the posterior distribution of the inverse problem.

    It requires that the physics operator has a singular value decomposition, i.e.,
    it is :class:`deepinv.physics.DecomposablePhysics` class.

    :param torch.nn.Module denoiser: a denoiser model that can handle different noise levels.
    :param list[int] sigmas: a list of noise levels to use in the diffusion, they should be in decreasing
        order from 1 to 0. Defaults to ``np.linspace(1, 0, 100)``, i.e., 100 equally spaced noise levels from 1 to 0.
    :param float eta: hyperparameter
    :param float etab: hyperparameter
    :param bool verbose: if True, print progress

    |sep|

    :Examples:

        Denoising diffusion restoration model using a pretrained DRUNet denoiser:

    ::

        import deepinv as dinv
        device = dinv.utils.get_device(verbose=False)
        seed = torch.manual_seed(0) # Random seed for reproducibility
        seed = torch.cuda.manual_seed(0) # Random seed for reproducibility on GPU
        x = 0.5 * torch.ones(1, 3, 32, 32, device=device) # Define plain gray 32x32 image
        physics = dinv.physics.Inpainting(
           mask=0.5, img_size=(3, 32, 32),
           noise_model=dinv.physics.GaussianNoise(0.1),
           device=device,
        )
        y = physics(x) # measurements
        denoiser = dinv.models.DRUNet(pretrained="download").to(device)
        model = dinv.sampling.DDRM(denoiser=denoiser, sigmas=np.linspace(1, 0, 10), verbose=True) # define the DDRM model
        xhat = model(y, physics) # sample from the posterior distribution
        (dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)).cpu() # tensor([True])



    """

    def __init__(
        self,
        denoiser,
        sigmas=None,
        eta=0.85,
        etab=1.0,
        verbose=False,
        eps=1e-6,
    ):
        if sigmas is None:
            sigmas = np.linspace(1, 0, 100)
        super(DDRM, self).__init__()
        self.denoiser = denoiser
        self.sigmas = sigmas
        self.max_iter = len(sigmas)
        self.eta = eta
        self.verbose = verbose
        self.etab = etab
        self.eps = eps

    def forward(self, y, physics: dinv.physics.DecomposablePhysics, seed=None):
        r"""
        Runs the diffusion to obtain a random sample of the posterior distribution.

        :param torch.Tensor y: the measurements.
        :param deepinv.physics.DecomposablePhysics physics: the physics operator, which must have a singular value
            decomposition.
        :param int seed: the seed for the random number generator.
        """
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            if hasattr(physics.noise_model, "sigma"):
                sigma_noise = physics.noise_model.sigma
            else:
                sigma_noise = 0.01

            if isinstance(physics, dinv.physics.Denoising):
                mask = torch.ones_like(
                    y
                )  # TODO: fix for economic SVD decompositions (eg. Decolorize)
            else:
                mask = torch.cat([physics.mask.abs()] * y.shape[0], dim=0)

            c = np.sqrt(1 - self.eta**2)
            y_bar = physics.U_adjoint(y)
            case = mask > sigma_noise
            y_bar[case] = y_bar[case] / (mask[case] + self.eps)
            nsr = torch.zeros_like(mask)
            nsr[case] = sigma_noise / (mask[case] + self.eps)

            # iteration 1
            # compute init noise
            mean = torch.zeros_like(y_bar)
            std = torch.ones_like(y_bar) * self.sigmas[0]
            mean[case] = y_bar[case]
            std[case] = (self.sigmas[0] ** 2 - nsr[case].pow(2)).sqrt()
            x_bar = mean + std * torch.randn_like(y_bar) / np.sqrt(2.0)
            x_bar_prev = x_bar

            # denoise
            x = self.denoiser(physics.V(x_bar), self.sigmas[0])

            for t in tqdm(range(1, self.max_iter), disable=(not self.verbose)):
                # add noise in transformed domain
                x_bar = physics.V_adjoint(x)

                case2 = torch.logical_and(case, (self.sigmas[t] < nsr))
                case3 = torch.logical_and(case, (self.sigmas[t] >= nsr))

                mean = (
                    x_bar
                    + c * self.sigmas[t] * (x_bar_prev - x_bar) / self.sigmas[t - 1]
                )
                mean[case2] = x_bar[case2] + c * self.sigmas[t] * (
                    y_bar[case2] - x_bar[case2]
                ) / (nsr[case2] + self.eps)
                mean[case3] = (1.0 - self.etab) * x_bar[case3] + self.etab * y_bar[
                    case3
                ]

                std = torch.ones_like(x_bar) * self.eta * self.sigmas[t]
                std[case3] = (
                    (self.sigmas[t] ** 2 - (nsr[case3] * self.etab).pow(2))
                    .clamp(min=0)
                    .sqrt()
                )

                x_bar = mean + std * torch.randn_like(x_bar) / np.sqrt(2.0)
                x_bar_prev = x_bar
                # denoise
                x = self.denoiser(physics.V(x_bar), self.sigmas[t])

        return x


class DiffPIR(Reconstructor):
    r"""
    Diffusion PnP Image Restoration (DiffPIR).

    This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described in :footcite:t:`zhu2023denoising`.

    The DiffPIR algorithm is inspired on a half-quadratic splitting (HQS) plug-and-play algorithm, where the denoiser
    is a conditional diffusion denoiser, combined with a diffusion process. The algorithm writes as follows,
    for :math:`t` decreasing from :math:`T` to :math:`1`:

     .. math::
             x_{0}^{t} &= D_{\theta}(x_t, \frac{\sqrt{1-\overline{\alpha}_t}}{\sqrt{\overline{\alpha}_t}}) \\
             \widehat{x}_{0}^{t} &= \operatorname{prox}_{2 f(y, \cdot) /{\rho_t}}(x_{0}^{t}) \\
             \widehat{\varepsilon} &= \left(x_t - \sqrt{\overline{\alpha}_t} \,\,
             \widehat{x}_{0}^t\right)/\sqrt{1-\overline{\alpha}_t} \\
             \varepsilon_t &= \mathcal{N}(0, \mathbf{I}) \\
             x_{t-1} &= \sqrt{\overline{\alpha}_t} \,\, \widehat{x}_{0}^t + \sqrt{1-\overline{\alpha}_t}
             \left(\sqrt{1-\zeta} \,\, \widehat{\varepsilon} + \sqrt{\zeta} \,\, \varepsilon_t\right)

    where :math:`D_\theta(\cdot,\sigma)` is a Gaussian denoiser network with noise level :math:`\sigma`
    and :math:`f(y, \cdot)` is the data fidelity
    term.

    .. note::

            The algorithm might require careful tunning of the hyperparameters :math:`\lambda` and :math:`\zeta` to
            obtain optimal results.

    :param torch.nn.Module model: a conditional noise estimation model
    :param float sigma: the noise level of the data
    :param deepinv.optim.DataFidelity data_fidelity: the data fidelity operator
    :param int max_iter: the number of iterations to run the algorithm (default: 100)
    :param float zeta: hyperparameter :math:`\zeta` for the sampling step (must be between 0 and 1). Default: 1.0.
    :param float lambda_: hyperparameter :math:`\lambda` for the data fidelity step
        (:math:`\rho_t = \lambda \frac{\sigma_n^2}{\bar{\sigma}_t^2}` in the paper where the optimal value range
        between 3.0 and 25.0 depending on the problem). Default: ``7.0``.
    :param bool verbose: if ``True``, print progress
    :param str device: the device to use for the computations

    |sep|

    :Examples:

        Denoising diffusion restoration model using a pretrained DRUNet denoiser:

    ::

        import deepinv as dinv
        device = dinv.utils.get_device(verbose=False)
        x = 0.5 * torch.ones(1, 3, 32, 32, device=device) # Define a plain gray 32x32 image
        physics = dinv.physics.Inpainting(mask=0.5, img_size=(3, 32, 32),
           noise_model=dinv.physics.GaussianNoise(0.1), device=device)
        y = physics(x) # Measurements
        denoiser = dinv.models.DRUNet(device=device)
        model = dinv.sampling.DiffPIR(model=denoiser, data_fidelity=dinv.optim.data_fidelity.L2(),
           device=device) # Define the DiffPIR model
        xhat = model(y, physics) # Run the DiffPIR algorithm
        print((dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x))) # should be True


    """

    def __init__(
        self,
        model,
        data_fidelity,
        sigma=0.05,
        max_iter=100,
        zeta=0.1,
        lambda_=7.0,
        verbose=False,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity
        self.max_iter = max_iter
        self.zeta = zeta
        self.verbose = verbose
        self.device = device
        self.beta_start, self.beta_end = 0.1 / 1000, 20 / 1000
        self.num_train_timesteps = 1000
        self.sigma = sigma

        (
            self.sqrt_1m_alphas_cumprod,
            self.reduced_alpha_cumprod,
            self.sqrt_alphas_cumprod,
            self.sqrt_recip_alphas_cumprod,
            self.sqrt_recipm1_alphas_cumprod,
            self.betas,
        ) = self.get_alpha_beta()

        self.rhos, self.sigmas, self.seq = self.get_noise_schedule(sigma=sigma)

    def get_alpha_beta(self):
        """
        Get the alpha and beta sequences for the algorithm. This is necessary for mapping noise levels to timesteps.
        """
        betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_train_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        reduced_alpha_cumprod = torch.div(
            sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
        )  # equivalent noise sigma on image
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        return (
            sqrt_1m_alphas_cumprod,
            reduced_alpha_cumprod,
            sqrt_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            betas,
        )

    def get_noise_schedule(self, sigma):
        """
        Get the noise schedule for the algorithm.
        """
        lambda_ = self.lambda_
        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(self.num_train_timesteps):
            sigmas.append(self.reduced_alpha_cumprod[self.num_train_timesteps - 1 - i])
            sigma_ks.append(
                (self.sqrt_1m_alphas_cumprod[i] / self.sqrt_alphas_cumprod[i])
            )
            rhos.append(lambda_ * (sigma**2) / (sigma_ks[i] ** 2))
        rhos, sigmas = (
            torch.tensor(rhos).to(self.device),
            torch.tensor(sigmas).to(self.device),
        )

        seq = torch.sqrt(
            torch.linspace(
                0.0, self.num_train_timesteps**2, self.max_iter, device=self.device
            )
        ).type(torch.int32)
        seq[-1] = seq[-1] - 1

        return rhos, sigmas, seq

    def find_nearest(self, array, value):
        """
        Find the argmin of the nearest value in an array.
        """
        idx = torch.abs(array - value).argmin()
        return idx

    def compute_alpha(self, betas, t):
        """
        Compute the alpha sequence from the beta sequence.
        """
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        at = alphas_cumprod[t]
        return at

    def get_alpha_prod(
        self, beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000
    ):
        """
        Get the alpha sequences; this is necessary for mapping noise levels to timesteps when performing pure denoising.
        """
        betas = torch.linspace(
            beta_start,
            beta_end,
            num_train_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        return (
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
        )

    def forward(
        self,
        y,
        physics: dinv.physics.LinearPhysics,
        seed=None,
        x_init=None,
    ):
        r"""
        Runs the diffusion to obtain a random sample of the posterior distribution.

        :param torch.Tensor y: the measurements.
        :param deepinv.physics.LinearPhysics physics: the physics operator.
        :param float sigma: the noise level of the data.
        :param int seed: the seed for the random number generator.
        :param torch.Tensor x_init: the initial guess for the reconstruction.
        """

        if seed:
            torch.manual_seed(seed)

        if hasattr(physics.noise_model, "sigma"):
            sigma = physics.noise_model.sigma  # Then we overwrite the default values
            self.rhos, self.sigmas, self.seq = self.get_noise_schedule(sigma=sigma)

        # Initialization
        if x_init is None:  # Necessary when x and y don't live in the same space
            x = 2 * physics.A_adjoint(y) - 1
        else:
            x = 2 * x_init - 1

        sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod = self.get_alpha_prod()

        with torch.no_grad():
            for i in tqdm(range(len(self.seq)), disable=(not self.verbose)):
                # Current noise level
                curr_sigma = self.sigmas[self.seq[i]]

                # time step associated with the noise level sigmas[i]
                t_i = self.find_nearest(self.reduced_alpha_cumprod, curr_sigma)
                at = 1 / sqrt_recip_alphas_cumprod[t_i] ** 2

                if (
                    i == 0
                ):  # Initialization (simpler than the original code, may be suboptimal)
                    x = (
                        x
                        + (curr_sigma**2 - 4.0 * self.sigma**2).sqrt()
                        * torch.randn_like(x)
                    ) / sqrt_recip_alphas_cumprod[-1]

                sigma_cur = curr_sigma

                # Denoising step
                x_aux = x / (2 * at.sqrt()) + 0.5  # renormalize in [0, 1]
                out = self.model(x_aux, sigma_cur / 2)
                denoised = 2 * out - 1
                x0 = denoised.clamp(-1, 1)

                if not self.seq[i] == self.seq[-1]:
                    # Data fidelity step
                    x0_p = x0 / 2 + 0.5
                    x0_p = self.data_fidelity.prox(
                        x0_p, y, physics, gamma=1.0 / (2 * self.rhos[t_i])
                    )
                    x0 = x0_p * 2 - 1

                    # Sampling step
                    t_im1 = self.find_nearest(
                        self.reduced_alpha_cumprod,
                        self.sigmas[self.seq[i + 1]],
                    )  # time step associated with the next noise level

                    eps = (
                        x - self.sqrt_alphas_cumprod[t_i] * x0
                    ) / self.sqrt_1m_alphas_cumprod[
                        t_i
                    ]  # effective noise

                    x = (
                        self.sqrt_alphas_cumprod[t_im1] * x0
                        + self.sqrt_1m_alphas_cumprod[t_im1]
                        * (1 - self.zeta) ** 0.5
                        * eps
                        + self.sqrt_1m_alphas_cumprod[t_im1]
                        * self.zeta**0.5
                        * torch.randn_like(x)
                    )  # sampling

        out = x / 2 + 0.5  # back to [0, 1] range

        return out


class DPS(PosteriorDiffusion):
    r"""
    Diffusion Posterior Sampling (DPS).

    This class implements the Diffusion Posterior Sampling algorithm (DPS) described in :footcite:t:`chung2022diffusion`.

    DPS is an approximation of a gradient-based posterior sampling algorithm,
    which has minimal assumptions on the forward model. The only restriction is that
    the measurement model has to be differentiable, which is generally the case.

    The algorithm solves the reverse-time SDE specified by the `schedule` argument, using the Euler solver, and approximating the conditional score by the DPS data fidelity term, which is defined as follows:

    .. math::

        \nabla_{x_t} \log p_t(y|x_t) \approx -\lambda \nabla_{x_t} \|y - A D_{\sigma_t}(x_t)\|

    where :math:`\denoiser{\cdot}{\sigma}` is a denoising network for noise level :math:`\sigma`, and :math:`\lambda` is a hyperparameter that controls the weight of the data fidelity term in the approximation of the likelihood gradient.

    .. note::

        This method is a particular instance of the general posterior sampling framework described in :class:`deepinv.sampling.PosteriorDiffusion`, by specifying the data fidelity term as the DPS data fidelity, a SDE and the Euler solver. The user can thus easily modify the algorithm by changing the SDE or the solver, for instance to use a different noise schedule or a different sampling scheme.
        Please refer to the example :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for a full demonstration of how to modify the algorithm.

    :param deepinv.models.Denoiser denoiser: a denoiser network that can handle different noise levels
    :param str schedule: the noise schedule to use, either `"vp"` (default, which matches the original implementation) for the variance preserving noise schedule, or `"ve"` for the variance exploding noise schedule.
    :param int num_steps: the number of diffusion iterations to run the algorithm (default: 1000)
    :param float alpha: DDIM hyperparameter which controls the stochasticity. Default to 1.0, which corresponds to the original DDPM sampling scheme. Setting it to 0 corresponds to the deterministic DDIM sampling scheme.
    :param float weight: the weight of the data fidelity term in the approximation of the likelihood gradient. Default to 1.0.
    :param str guidance: the form of the guidance, passed to :class:`deepinv.sampling.DPSDataFidelity`.
        `"norm"` (default) differentiates the residual norm, as in the original paper; `"annealed"` differentiates
        the Gaussian negative log-likelihood with the annealed variance :math:`\sigma_y^2 + \sigma_t^2`, which puts
        `weight` on the same scale as the other noisy data-fidelity terms.
    :param bool verbose: if `True`, print the progress of the algorithm
    :param str device: the device to use for the computations

    """

    def __init__(
        self,
        denoiser: Denoiser,
        schedule: str = "vp",
        alpha: float = 1.0,
        num_steps: int = 1000,
        weight: float = 1.0,
        guidance: str = "norm",
        verbose: bool = False,
        device: str | torch.device = "cpu",
        dtype=torch.float64,
        rng: torch.Generator | None = None,
        **kwargs,
    ):
        data_fidelity = DPSDataFidelity(
            denoiser=denoiser, clip=[-1.0, 1.0], weight=weight, guidance=guidance
        )

        solver = EulerSolver(
            timesteps=torch.linspace(1, 0.001, num_steps, device=device, dtype=dtype),
            rng=rng,
        )
        if schedule.lower() == "vp":
            sde = VariancePreservingDiffusion(
                alpha=alpha,
                device=device,
                dtype=dtype,
            )
        elif schedule.lower() == "ve":
            sde = VarianceExplodingDiffusion(
                alpha=alpha,
                device=device,
                dtype=dtype,
            )

        else:
            raise ValueError(
                f"Only 'vp' and 've' schedules are supported, got {schedule}"
            )

        super().__init__(
            sde=sde,
            denoiser=denoiser,
            data_fidelity=data_fidelity,
            solver=solver,
            verbose=verbose,
            device=device,
            dtype=dtype,
            **kwargs,
        )


class BlindDPS(Reconstructor):
    r"""
    Blind diffusion posterior sampling (BlindDPS).

    Jointly samples an image and a blur kernel using independent diffusion
    priors, following :footcite:t:`chung2023parallel`. For the measurement model
    :math:`y=A(x,k)+\varepsilon`, the likelihood scores are approximated by

    .. math::

        \nabla_{x_t}\log p_t(y\mid x_t,k_t)
        &\approx -\lambda_x\nabla_{x_t}\|A(\hat{x}_0,\hat{k}_0)-y\|, \\
        \nabla_{k_t}\log p_t(y\mid x_t,k_t)
        &\approx -\lambda_k\nabla_{k_t}\|A(\hat{x}_0,\hat{k}_0)-y\|,

    where both clean estimates are provided by denoisers. The likelihood
    gradient is backpropagated through both denoisers and the forward operator.
    Kernel estimates are clipped to :math:`[0,1]` and normalized to be
    nonnegative with unit sum before applying the physics. The kernel prior
    score uses the denoised estimate before normalization.

    Like :class:`deepinv.sampling.DPS`, this implementation uses the continuous
    reverse-time SDE framework with an Euler solver by default, rather than
    the discrete DDPM updates of the reference implementation. The image and
    kernel follow the same diffusion schedule. Both denoisers must accept
    inputs and noise levels in the :math:`[0,1]` data convention.

    :param deepinv.models.Denoiser denoiser: image denoiser.
    :param deepinv.models.Denoiser kernel_denoiser: blur-kernel denoiser.
    :param int, tuple[int, int] kernel_size: spatial kernel size. An integer
        specifies a square kernel. Default: ``64``.
    :param str schedule: ``"vp"`` (default) or ``"ve"``. Ignored if ``sde`` is supplied.
    :param int num_steps: number of solver time points. Default: ``1000``.
    :param float alpha: stochasticity of the SDE; zero gives deterministic
        sampling. Ignored if ``sde`` is supplied. Default: ``1.0``.
    :param float, Callable weight: image likelihood-guidance weight, either a
        constant or a function of solver time ``t``. Default: ``1.0``.
    :param float, Callable kernel_weight: kernel likelihood-guidance weight,
        either a constant or a function of solver time ``t``. Default: ``1.0``.
    :param deepinv.sampling.EDMDiffusionSDE sde: optional custom diffusion SDE,
        defining the shared image and kernel schedule.
    :param deepinv.sampling.BaseSDESolver solver: optional custom SDE solver.
    :param torch.device, str device: computation device.
    :param torch.dtype dtype: SDE computation dtype. Denoisers are evaluated in
        ``torch.float32``. Default: ``torch.float64``.
    :param torch.Generator rng: random number generator for the default solver.
        If omitted, a generator is created on ``device``.
    :param bool verbose: whether to show sampling progress. Default: ``False``.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        kernel_denoiser: Denoiser,
        kernel_size: int | tuple[int, int] = 64,
        schedule: str = "vp",
        num_steps: int = 1000,
        alpha: float = 1.0,
        weight: float | Callable = 1.0,
        kernel_weight: float | Callable = 1.0,
        sde: EDMDiffusionSDE | None = None,
        solver: BaseSDESolver | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        rng: torch.Generator | None = None,
        verbose: bool = False,
    ):
        super().__init__(device=device)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if len(kernel_size) != 2 or any(size <= 0 for size in kernel_size):
            raise ValueError("kernel_size must contain two positive spatial sizes.")
        self.kernel_size = tuple(kernel_size)
        self.denoiser = denoiser
        self.kernel_denoiser = kernel_denoiser
        self.weight = weight
        self.kernel_weight = kernel_weight
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose

        if sde is None:
            if schedule.lower() == "vp":
                sde_class = VariancePreservingDiffusion
            elif schedule.lower() == "ve":
                sde_class = VarianceExplodingDiffusion
            else:
                raise ValueError(
                    f"Only 'vp' and 've' schedules are supported, got {schedule}."
                )
            sde = sde_class(alpha=alpha, device=self.device, dtype=dtype)
        self.sde = sde

        if solver is None:
            if num_steps < 2:
                raise ValueError("num_steps must be at least two.")
            if rng is None:
                rng = torch.Generator(device=self.device)
            solver = EulerSolver(
                timesteps=torch.linspace(
                    1, 0.001, num_steps, device=self.device, dtype=dtype
                ),
                rng=rng,
            )
        self.solver = solver

    @staticmethod
    def _normalize_kernel(kernel: Tensor) -> Tensor:
        """Normalize each nonnegative kernel, using a uniform zero-kernel fallback."""
        total = kernel.sum(dim=(-2, -1), keepdim=True)
        normalized = kernel / torch.where(total > 0, total, torch.ones_like(total))
        uniform = torch.full_like(kernel, 1 / (kernel.shape[-2] * kernel.shape[-1]))
        return torch.where(total > 0, normalized, uniform)

    @torch.no_grad()
    def forward(
        self,
        y: Tensor,
        physics: Blur,
        x_init: Tensor | tuple[int, ...] | None = None,
        kernel_init: Tensor | None = None,
        seed: int | None = None,
        timesteps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Sample an image and kernel conditioned on the blurred measurements.

        :param torch.Tensor y: measurements of shape ``(B, C, H, W)``.
        :param deepinv.physics.Blur physics: blur operator, supporting both
            spatial and FFT convolution through ``use_fft``. Its original
            filter is restored after sampling.
        :param torch.Tensor, tuple[int, ...] x_init: initial noisy image state,
            in the internal :math:`[-1,1]` coordinates, or its shape. If omitted,
            Gaussian initialization uses ``y.shape``. Supply the image shape
            explicitly when the physics changes spatial dimensions.
        :param torch.Tensor kernel_init: initial noisy kernel state in internal
            :math:`[-1,1]` coordinates, of shape ``(B, 1, h, w)``, matching
            ``kernel_size``. If omitted, initialize from the diffusion prior.
        :param int seed: seed for the solver's random number generator. Custom
            solvers require an initialized generator for this to take effect.
        :param torch.Tensor timesteps: optional decreasing solver time points,
            overriding the default schedule.

        :return: an image in :math:`[0,1]` and a nonnegative unit-sum kernel.
            Both are denoised at the final solver time point and returned in
            the SDE computation dtype.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if not isinstance(physics, Blur):
            raise ValueError(
                "BlindDPS requires deepinv.physics.Blur; use Blur(use_fft=True) "
                "for FFT-based convolution."
            )
        self.solver.rng_manual_seed(seed)
        if timesteps is None:
            timesteps = self.solver.timesteps
        timesteps = timesteps.to(device=self.device, dtype=self.dtype)
        if timesteps.ndim != 1 or len(timesteps) < 2:
            raise ValueError("timesteps must contain at least two time points.")
        if not torch.all(timesteps[:-1] > timesteps[1:]):
            raise ValueError("timesteps must be strictly decreasing.")

        if x_init is None:
            x_init = y.shape
        if isinstance(x_init, (tuple, list, torch.Size)):
            x_init = self.sde.sample_init(x_init, rng=self.solver.rng, t=timesteps[0])
        x_init = x_init.to(device=self.device, dtype=self.dtype)
        image_shape = x_init.shape
        if x_init.ndim != 4 or image_shape[0] != y.shape[0]:
            raise ValueError("x_init must have shape (B, C, H, W), matching y's batch.")
        kernel_shape = (image_shape[0], 1, *self.kernel_size)
        if kernel_init is None:
            kernel_init = self.sde.sample_init(
                kernel_shape, rng=self.solver.rng, t=timesteps[0]
            )
        if tuple(kernel_init.shape) != kernel_shape:
            raise ValueError(f"kernel_init must have shape {kernel_shape}.")
        kernel_init = kernel_init.to(device=self.device, dtype=self.dtype)
        image_numel = x_init[0].numel()
        state = torch.cat((x_init.flatten(1), kernel_init.flatten(1)), dim=1)

        def unpack(z):
            return (
                z[:, :image_numel].reshape(image_shape),
                z[:, image_numel:].reshape(kernel_shape),
            )

        def denoise(z, t):
            image, kernel = unpack(z)
            scale = self.sde.scale_t(t)
            sigma = self.sde.sigma_t(t).to(torch.float32) / 2
            image = ((image / scale + 1) / 2).to(torch.float32)
            kernel = ((kernel / scale + 1) / 2).to(torch.float32)
            return (
                self.denoiser(image, sigma).clamp(0, 1),
                self.kernel_denoiser(kernel, sigma).clamp(0, 1),
            )

        def backward_drift(z, t):
            with torch.enable_grad():
                z_grad = z.detach().requires_grad_(True)
                image, kernel = denoise(z_grad, t)
                normalized_kernel = self._normalize_kernel(kernel)
                current_filter = physics.filter
                physics_dtype = (
                    current_filter.dtype
                    if isinstance(current_filter, Tensor)
                    else image.dtype
                )
                difference = (
                    physics.A(
                        image.to(physics_dtype),
                        filter=normalized_kernel.to(physics_dtype),
                    )
                    - y
                )
                loss = torch.linalg.vector_norm(difference.flatten(1), dim=1).sum()
                gradient = torch.autograd.grad(loss, z_grad)[0]

            model_output = torch.cat(
                (image.detach().flatten(1), kernel.detach().flatten(1)), dim=1
            ).to(self.dtype)
            score = self.sde._score_from_model_output(
                z,
                2 * model_output - 1,
                self.sde.sigma_t(t),
                self.sde.scale_t(t),
            )
            weight = self.weight(t) if callable(self.weight) else self.weight
            kernel_weight = (
                self.kernel_weight(t)
                if callable(self.kernel_weight)
                else self.kernel_weight
            )
            guidance = torch.cat(
                (
                    weight * gradient[:, :image_numel],
                    kernel_weight * gradient[:, image_numel:],
                ),
                dim=1,
            )
            return -self.sde.forward_drift(z, t) + (
                (1 + self.sde.alpha(t)) / 2
            ) * self.sde.forward_diffusion(t) ** 2 * (score - guidance)

        posterior = BaseSDE(
            drift=backward_drift,
            diffusion=self.sde.diffusion,
            device=self.device,
            dtype=self.dtype,
        )
        original_filter = physics.filter
        try:
            solution = self.solver.sample(
                posterior, state, timesteps=timesteps, verbose=self.verbose
            )
            image, kernel = denoise(solution.sample, timesteps[-1])
        finally:
            physics.filter = original_filter
        return image.to(self.dtype), self._normalize_kernel(kernel).to(self.dtype)
