import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from deepinv.models import Reconstructor

import deepinv.physics
from deepinv.sampling.langevin import MonteCarlo
from deepinv.utils.plotting import plot


class DiffusionSampler(MonteCarlo):
    r"""
    Turns a diffusion method into a Monte Carlo sampler

    Unlike diffusion methods, the resulting sampler computes the mean and variance of the distribution
    by running the diffusion multiple times.

    :param torch.nn.Module diffusion: a diffusion model
    :param int max_iter: the maximum number of iterations
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

        def iterator(x, y, physics, likelihood, prior):
            # run one sampling kernel iteration
            x = prior(y, physics)
            return x

        super().__init__(
            iterator,
            prior,
            data_fidelity,
            max_iter=max_iter,
            thinning=1,
            save_chain=save_chain,
            burnin_ratio=0.0,
            clip=clip,
            verbose=verbose,
            thresh_conv=thres_conv,
            g_statistic=g_statistic,
        )


class DDRM(Reconstructor):
    r"""DDRM(self, denoiser, sigmas=np.linspace(1, 0, 100), eta=0.85, etab=1.0, verbose=False)
    Denoising Diffusion Restoration Models (DDRM).

    This class implements the denoising diffusion restoration model (DDRM) described in https://arxiv.org/abs/2201.11793.

    The DDRM is a sampling method that uses a denoiser to sample from the posterior distribution of the inverse problem.

    It requires that the physics operator has a singular value decomposition, i.e.,
    it is :class:`deepinv.physics.DecomposablePhysics` class.

    :param torch.nn.Module denoiser: a denoiser model that can handle different noise levels.
    :param list[int] sigmas: a list of noise levels to use in the diffusion, they should be in decreasing
        order from 1 to 0.
    :param float eta: hyperparameter
    :param float etab: hyperparameter
    :param bool verbose: if True, print progress

    |sep|

    :Examples:

        Denoising diffusion restoration model using a pretrained DRUNet denoiser:

        >>> import deepinv as dinv
        >>> device = dinv.utils.get_freer_gpu(verbose=False) if torch.cuda.is_available() else 'cpu'
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> seed = torch.cuda.manual_seed(0) # Random seed for reproducibility on GPU
        >>> x = 0.5 * torch.ones(1, 3, 32, 32, device=device) # Define plain gray 32x32 image
        >>> physics = dinv.physics.Inpainting(
        ...   mask=0.5, tensor_size=(3, 32, 32),
        ...   noise_model=dinv.physics.GaussianNoise(0.1),
        ...   device=device,
        ... )
        >>> y = physics(x) # measurements
        >>> denoiser = dinv.models.DRUNet(pretrained="download").to(device)
        >>> model = dinv.sampling.DDRM(denoiser=denoiser, sigmas=np.linspace(1, 0, 10), verbose=True) # define the DDRM model
        >>> xhat = model(y, physics) # sample from the posterior distribution
        >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x) # Should be closer to the original
        tensor([True])

    """

    def __init__(
        self,
        denoiser,
        sigmas=np.linspace(1, 0, 100),
        eta=0.85,
        etab=1.0,
        verbose=False,
    ):
        super(DDRM, self).__init__()
        self.denoiser = denoiser
        self.sigmas = sigmas
        self.max_iter = len(sigmas)
        self.eta = eta
        self.verbose = verbose
        self.etab = etab

    def forward(self, y, physics: deepinv.physics.DecomposablePhysics, seed=None):
        r"""
        Runs the diffusion to obtain a random sample of the posterior distribution.

        :param torch.Tensor y: the measurements.
        :param deepinv.physics.DecomposablePhysics physics: the physics operator, which must have a singular value
            decomposition.
        :param int seed: the seed for the random number generator.
        """
        # assert physics.__class__ == deepinv.physics.DecomposablePhysics, 'The forward operator requires a singular value decomposition'
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            if hasattr(physics.noise_model, "sigma"):
                sigma_noise = physics.noise_model.sigma
            else:
                sigma_noise = 0.01

            if physics.__class__ == deepinv.physics.Denoising:
                mask = torch.ones_like(
                    y
                )  # TODO: fix for economic SVD decompositions (eg. Decolorize)
            else:
                mask = torch.cat([physics.mask.abs()] * y.shape[0], dim=0)

            c = np.sqrt(1 - self.eta**2)
            y_bar = physics.U_adjoint(y)
            case = mask > sigma_noise
            y_bar[case] = y_bar[case] / mask[case]
            nsr = torch.zeros_like(mask)
            nsr[case] = sigma_noise / mask[case]

            # iteration 1
            # compute init noise
            mean = torch.zeros_like(y_bar)
            std = torch.ones_like(y_bar) * self.sigmas[0]
            mean[case] = y_bar[case]
            std[case] = (self.sigmas[0] ** 2 - nsr[case].pow(2)).sqrt()
            x_bar = mean + std * torch.randn_like(y_bar)
            x_bar_prev = x_bar.clone()

            # denoise
            x = self.denoiser(physics.V(x_bar), self.sigmas[0])

            for t in tqdm(range(1, self.max_iter), disable=(not self.verbose)):
                # add noise in transformed domain
                x_bar = physics.V_adjoint(x)

                case2 = torch.logical_and(case, (self.sigmas[t] < nsr))
                case3 = torch.logical_and(case, (self.sigmas[t] >= nsr))

                # n = np.prod(mask.shape)
                # print(f'case: {case.sum()/n*100:.2f}, case2: {case2.sum()/n*100:.2f}, case3: {case3.sum()/n*100:.2f}')

                mean = (
                    x_bar
                    + c * self.sigmas[t] * (x_bar_prev - x_bar) / self.sigmas[t - 1]
                )
                mean[case2] = (
                    x_bar[case2]
                    + c * self.sigmas[t] * (y_bar[case2] - x_bar[case2]) / nsr[case2]
                )
                mean[case3] = (1.0 - self.etab) * x_bar[case3] + self.etab * y_bar[
                    case3
                ]

                std = torch.ones_like(x_bar) * self.eta * self.sigmas[t]
                std[case3] = (
                    self.sigmas[t] ** 2 - (nsr[case3] * self.etab).pow(2)
                ).sqrt()

                x_bar = mean + std * torch.randn_like(x_bar)
                x_bar_prev = x_bar.clone()
                # denoise
                x = self.denoiser(physics.V(x_bar), self.sigmas[t])

        return x


class DiffPIR(Reconstructor):
    r"""
    Diffusion PnP Image Restoration (DiffPIR).

    This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described
    in https://arxiv.org/abs/2305.08995.

    The DiffPIR algorithm is inspired on a half-quadratic splitting (HQS) plug-and-play algorithm, where the denoiser
    is a conditional diffusion denoiser, combined with a diffusion process. The algorithm writes as follows,
    for :math:`t` decreasing from :math:`T` to :math:`1`:

     .. math::
             \begin{equation*}
             \begin{aligned}
             x_{0}^{t} &= D_{\theta}(x_t, \frac{\sqrt{1-\overline{\alpha}_t}}{\sqrt{\overline{\alpha}_t}}) \\
             \widehat{x}_{0}^{t} &= \operatorname{prox}_{2 f(y, \cdot) /{\rho_t}}(x_{0}^{t}) \\
             \widehat{\varepsilon} &= \left(x_t - \sqrt{\overline{\alpha}_t} \,\,
             \widehat{x}_{0}^t\right)/\sqrt{1-\overline{\alpha}_t} \\
             \varepsilon_t &= \mathcal{N}(0, \mathbf{I}) \\
             x_{t-1} &= \sqrt{\overline{\alpha}_t} \,\, \widehat{x}_{0}^t + \sqrt{1-\overline{\alpha}_t}
             \left(\sqrt{1-\zeta} \,\, \widehat{\varepsilon} + \sqrt{\zeta} \,\, \varepsilon_t\right),
             \end{aligned}
             \end{equation*}

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

        >>> import deepinv as dinv
        >>> device = dinv.utils.get_freer_gpu(verbose=False) if torch.cuda.is_available() else 'cpu' 
        >>> x = 0.5 * torch.ones(1, 3, 32, 32, device=device) # Define a plain gray 32x32 image
        >>> physics = dinv.physics.Inpainting(
        ...   mask=0.5, tensor_size=(3, 32, 32),
        ...   noise_model=dinv.physics.GaussianNoise(0.1),
        ...   device=device
        ... )
        >>> y = physics(x) # Measurements
        >>> denoiser = dinv.models.DRUNet(pretrained="download").to(device)
        >>> model = DiffPIR(
        ...   model=denoiser,
        ...   data_fidelity=dinv.optim.data_fidelity.L2()
        ... ) # Define the DiffPIR model
        >>> xhat = model(y, physics) # Run the DiffPIR algorithm
        >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x) # Should be closer to the original
        tensor([True])
        
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
        super(DiffPIR, self).__init__()
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity
        self.max_iter = max_iter
        self.zeta = zeta
        self.verbose = verbose
        self.device = device
        self.beta_start, self.beta_end = 0.1 / 1000, 20 / 1000
        self.num_train_timesteps = 1000

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
        betas = np.linspace(
            self.beta_start, self.beta_end, self.num_train_timesteps, dtype=np.float32
        )
        betas = torch.from_numpy(betas).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t

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
        rhos, sigmas = torch.tensor(rhos).to(self.device), torch.tensor(sigmas).to(
            self.device
        )

        seq = np.sqrt(np.linspace(0, self.num_train_timesteps**2, self.max_iter))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1

        return rhos, sigmas, seq

    def find_nearest(self, array, value):
        """
        Find the argmin of the nearest value in an array.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return torch.tensor([idx])

    def compute_alpha(self, betas, t):
        """
        Compute the alpha sequence from the beta sequence.
        """
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
        at = alphas_cumprod[t]
        return at

    def get_alpha_prod(
        self, beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000
    ):
        """
        Get the alpha sequences; this is necessary for mapping noise levels to timesteps when performing pure denoising.
        """
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas = torch.from_numpy(
            betas
        )  # .to(self.device) Removing this for now, can be done outside
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t

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
        physics: deepinv.physics.LinearPhysics,
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
                t_i = self.find_nearest(
                    self.reduced_alpha_cumprod, curr_sigma.cpu().numpy()
                )
                at = 1 / sqrt_recip_alphas_cumprod[t_i] ** 2

                if (
                    i == 0
                ):  # Initialization (simpler than the original code, may be suboptimal)
                    x = (
                        x + curr_sigma * torch.randn_like(x)
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
                        self.sigmas[self.seq[i + 1]].cpu().numpy(),
                    )  # time step associated with the next noise level

                    eps = (
                        x - self.sqrt_alphas_cumprod[t_i] * x0
                    ) / self.sqrt_1m_alphas_cumprod[
                        t_i
                    ]  # effective noise

                    x = (
                        self.sqrt_alphas_cumprod[t_im1] * x0
                        + self.sqrt_1m_alphas_cumprod[t_im1]
                        * np.sqrt(1 - self.zeta)
                        * eps
                        + self.sqrt_1m_alphas_cumprod[t_im1]
                        * np.sqrt(self.zeta)
                        * torch.randn_like(x)
                    )  # sampling

        out = x / 2 + 0.5  # back to [0, 1] range

        return out


class DPS(Reconstructor):
    r"""
    Diffusion Posterior Sampling (DPS).

    This class implements the Diffusion Posterior Sampling algorithm (DPS) described in
    https://arxiv.org/abs/2209.14687.

    DPS is an approximation of a gradient-based posterior sampling algorithm,
    which has minimal assumptions on the forward model. The only restriction is that
    the measurement model has to be differentiable, which is generally the case.

    The algorithm writes as follows, for :math:`t` decreasing from :math:`T` to :math:`1`:

    .. math::

            \begin{equation*}
            \begin{aligned}
            \widehat{\mathbf{x}}_{t} &= D_{\theta}(\mathbf{x}_t, \sqrt{1-\overline{\alpha}_t}/\sqrt{\overline{\alpha}_t})
            \\
            \mathbf{g}_t &= \nabla_{\mathbf{x}_t} \log p( \widehat{\mathbf{x}}_{t}(\mathbf{x}_t) | \mathbf{y} ) \\
            \mathbf{\varepsilon}_t &= \mathcal{N}(0, \mathbf{I}) \\
            \mathbf{x}_{t-1} &= a_t \,\, \mathbf{x}_t
            + b_t \, \, \widehat{\mathbf{x}}_t
            + \tilde{\sigma}_t \, \, \mathbf{\varepsilon}_t + \mathbf{g}_t,
            \end{aligned}
            \end{equation*}

    where :math:`\denoiser{\cdot}{\sigma}` is a denoising network for noise level :math:`\sigma`,
    :math:`\eta` is a hyperparameter, and the constants :math:`\tilde{\sigma}_t, a_t, b_t` are defined as

    .. math::
            \begin{equation*}
            \begin{aligned}
              \tilde{\sigma}_t &= \eta \sqrt{ (1 - \frac{\overline{\alpha}_t}{\overline{\alpha}_{t-1}})
              \frac{1 - \overline{\alpha}_{t-1}}{1 - \overline{\alpha}_t}} \\
              a_t &= \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}/\sqrt{1-\overline{\alpha}_t} \\
              b_t &= \sqrt{\overline{\alpha}_{t-1}} - \sqrt{1 - \overline{\alpha}_{t-1} - \tilde{\sigma}_t^2}
              \frac{\sqrt{\overline{\alpha}_{t}}}{\sqrt{1 - \overline{\alpha}_{t}}}.
            \end{aligned}
            \end{equation*}

    :param torch.nn.Module model: a denoiser network that can handle different noise levels
    :param deepinv.optim.DataFidelity data_fidelity: the data fidelity operator
    :param int max_iter: the number of diffusion iterations to run the algorithm (default: 1000)
    :param float eta: DDIM hyperparameter which controls the stochasticity
    :param bool verbose: if True, print progress
    :param str device: the device to use for the computations
    """

    def __init__(
        self,
        model,
        data_fidelity,
        max_iter=1000,
        eta=1.0,
        verbose=False,
        device="cpu",
        save_iterates=False,
    ):
        super(DPS, self).__init__()
        self.model = model
        self.model.requires_grad_(True)
        self.data_fidelity = data_fidelity
        self.max_iter = max_iter
        self.eta = eta
        self.verbose = verbose
        self.device = device
        self.beta_start, self.beta_end = 0.1 / 1000, 20 / 1000
        self.num_train_timesteps = 1000
        self.save_iterates = save_iterates

        self.betas, self.alpha_cumprod = self.compute_alpha_betas()

    def compute_alpha_betas(self):
        r"""

        Get the beta and alpha sequences for the algorithm. This is necessary for mapping noise levels to timesteps.

        """
        betas = np.linspace(
            self.beta_start, self.beta_end, self.num_train_timesteps, dtype=np.float32
        )
        betas = torch.from_numpy(betas).to(self.device)

        alpha_cumprod = (
            1 - torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
        ).cumprod(dim=0)
        return betas, alpha_cumprod

    def get_alpha(self, alpha_cumprod, t):
        a = alpha_cumprod.index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def forward(self, y, physics: deepinv.physics.Physics, seed=None, x_init=None):
        if seed:
            torch.manual_seed(seed)

        skip = self.num_train_timesteps // self.max_iter
        batch_size = y.shape[0]

        seq = range(0, self.num_train_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        time_pairs = list(zip(reversed(seq), reversed(seq_next)))

        # Initial sample from x_T
        x = torch.randn_like(y) if x_init is None else (2 * x_init - 1)

        if self.save_iterates:
            xs = [x]

        xt = x.to(self.device)

        for i, j in tqdm(time_pairs, disable=(not self.verbose)):
            t = (torch.ones(batch_size) * i).to(self.device)
            next_t = (torch.ones(batch_size) * j).to(self.device)

            at = self.get_alpha(self.alpha_cumprod, t.long())
            at_next = self.get_alpha(self.alpha_cumprod, next_t.long())

            with torch.enable_grad():
                xt.requires_grad_(True)

                # 1. Denoising step
                aux_x = xt / (2 * at.sqrt()) + 0.5  # renormalize in [0, 1]
                sigma_cur = (1 - at).sqrt() / at.sqrt()  # sigma_t

                x0_t = 2 * self.model(aux_x, sigma_cur / 2) - 1
                x0_t = torch.clip(x0_t, -1.0, 1.0)

                # 2. Likelihood gradient approximation
                l2_loss = self.data_fidelity(x0_t, y, physics).sqrt().sum()

            norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=xt)[0]
            norm_grad = norm_grad.detach()

            sigma_tilde = (
                (1 - at / at_next) * (1 - at_next) / (1 - at)
            ).sqrt() * self.eta
            c2 = ((1 - at_next) - sigma_tilde**2).sqrt()

            # 3. Noise step
            epsilon = torch.randn_like(xt)

            # 4. DDPM(IM) step
            xt_next = (
                (at_next.sqrt() - c2 * at.sqrt() / (1 - at).sqrt()) * x0_t
                + sigma_tilde * epsilon
                + c2 * xt / (1 - at).sqrt()
                - norm_grad
            )

            if self.save_iterates:
                xs.append(xt_next.to("cpu"))
            xt = xt_next.clone()

        if self.save_iterates:
            return xs
        else:
            return xt


# if __name__ == "__main__":
#     import deepinv as dinv
#     from deepinv.models.denoiser import Denoiser
#     import torchvision
#     from deepinv.loss.metric import PSNR
#
#     device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
#
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(device) / 255
#
#     sigma_noise = 0.01
#     # physics = dinv.physics.Denoising()
#
#     # physics = dinv.physics.BlurFFT(img_size=x.shape[1:], filter=dinv.physics.blur.gaussian_blur(sigma=1.),
#     #                               device=device)
#     physics = dinv.physics.Decolorize()
#     # physics = dinv.physics.Inpainting(
#     #   mask=0.5, tensor_size=(3, 218, 178), device=dinv.device
#     # )
#     # physics.mask *= (torch.rand_like(physics.mask))
#     physics.noise_model = dinv.physics.GaussianNoise(sigma_noise)
#
#     y = physics(x)
#     model_spec = {
#         "name": "drunet",
#         "args": {"device": device, "pretrained": "download"},
#     }
#
#     denoiser = Denoiser(model_spec=model_spec)
#
#     f = DDRM(
#         denoiser=denoiser,
#         etab=1.0,
#         sigma_noise=sigma_noise,
#         sigmas=np.linspace(1, 0, 100),
#         verbose=True,
#     )
#
#     xhat = f(y, physics)
#     dinv.utils.plot(
#         [physics.A_adjoint(y), x, xhat], titles=["meas.", "ground-truth", "xhat"]
#     )
#
#     print(f"PSNR 1 sample: {PSNR()(x, xhat):.2f} dB")
#     # print(f'mean PSNR sample: {PSNR()(x, denoiser(y, sigma_noise)):.2f} dB')
#
#     # sampler = dinv.sampling.DiffusionSampler(f, max_iter=10, save_chain=True, verbose=True)
#     # xmean, xvar = sampler(y, physics)
#
#     # chain = sampler.get_chain()
#     # distance = np.zeros((len(chain)))
#     # for k, xhat in enumerate(chain):
#     #    dist = (xhat - xmean).pow(2).mean()
#     #    distance[k] = dist
#     # distance = np.sort(distance)
#     # thres = distance[int(len(distance) * .95)]  #
#     # err = (x - xmean).pow(2).mean()
#     # print(f'Confidence region: {thres:.2e}, error: {err:.2e}')
#
#     # xstdn = xvar.sqrt()
#     # xstdn_plot = xstdn.sum(dim=1).unsqueeze(1)
#
#     # error = (xmean - x).abs()  # per pixel average abs. error
#     # error_plot = error.sum(dim=1).unsqueeze(1)
#
#     # print(f'Correct std: {(xstdn>error).sum()/np.prod(xstdn.shape)*100:.1f}%')
#     # error = (xmean - x)
#     # dinv.utils.plot_debug(
#     #    [physics.A_adjoint(y), x, xmean, xstdn_plot, error_plot], titles=["meas.", "ground-truth", "mean", "std", "error"]
#     # )
#
#     # print(f'PSNR 1 sample: {PSNR()(x, chain[0]):.2f} dB')
#     # print(f'mean PSNR sample: {PSNR()(x, xmean):.2f} dB')
