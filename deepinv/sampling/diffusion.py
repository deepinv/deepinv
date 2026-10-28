from __future__ import annotations

from typing import Callable
import torch
import numpy as np
from deepinv.models import Denoiser
from deepinv.optim.data_fidelity import DataFidelity

from deepinv.sampling import BaseSampling
from deepinv.sampling.sampling_iterators import DiffusionIterator
from deepinv.sampling.diffusion_sde import (
    PosteriorDiffusion,
    VariancePreservingDiffusion,
    VarianceExplodingDiffusion,
)
from deepinv.sampling.noisy_datafidelity import (
    DDRMDataFidelity,
    DiffPIRDataFidelity,
    DPSDataFidelity,
)
from deepinv.sampling.sde_solver import (
    AncestralSolver,
    BaseSDESolver,
    DDIMSolver,
    DDPMSolver,
    EulerSolver,
)


def _infer_device(
    denoiser: torch.nn.Module | None, device: str | torch.device | None
) -> str | torch.device:
    if device is not None:
        return device
    if isinstance(denoiser, torch.nn.Module):
        for p in denoiser.parameters():
            return p.device
        for b in denoiser.buffers():
            return b.device
    return "cpu"


def _build_sde_and_solver(
    schedule: str,
    solver: str | BaseSDESolver,
    alpha: Callable | float,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype,
    rng: torch.Generator | None = None,
    timesteps: torch.Tensor | np.ndarray | list | None = None,
    eta: float = 0.0,
):
    if isinstance(solver, str):
        # The ancestral solvers never evaluate the denoiser at the last time step, so they can end at t = 0,
        # where the last step returns the denoised estimate.
        solvers = {
            "ancestral": (AncestralSolver, 0.0),
            "ddpm": (DDPMSolver, 0.0),
            "ddim": (DDIMSolver, 0.0),
            "euler": (EulerSolver, 0.001),
        }
        if solver.lower() not in solvers:
            raise ValueError(
                f"solver must be one of {list(solvers)} or a BaseSDESolver instance, got {solver}"
            )
        solver_class, t_end = solvers[solver.lower()]
        if timesteps is None:
            timesteps = torch.linspace(1, t_end, num_steps, device=device, dtype=dtype)
        else:
            timesteps = torch.as_tensor(timesteps, device=device, dtype=dtype).clone()
            if solver.lower() == "euler" and len(timesteps) > 0 and timesteps[-1] == 0:
                timesteps[-1] = t_end

        if solver.lower() == "ddim":
            solver = solver_class(timesteps=timesteps, eta=eta, rng=rng)
        else:
            solver = solver_class(timesteps=timesteps, rng=rng)

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
        raise ValueError(f"Only 'vp' and 've' schedules are supported, got {schedule}")

    return sde, solver


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


class DDRM(PosteriorDiffusion):
    r"""
    Denoising Diffusion Restoration Models (DDRM).

    This class implements the Denoising Diffusion Restoration Model (DDRM) described in :footcite:t:`kawar2022denoising`.

    DDRM is a posterior sampling method that uses a denoiser and the singular value decomposition (SVD) of a
    linear forward operator :math:`A = U \Sigma V^\top` (:class:`deepinv.physics.DecomposablePhysics`)
    to sample from the posterior distribution of an inverse problem.

    The algorithm solves the reverse-time SDE specified by ``schedule`` using the solver specified by ``solver``,
    approximating the likelihood score in the SVD domain via :class:`deepinv.sampling.DDRMDataFidelity`:

    .. math::

        \nabla_{x_t} \log p_t(y \vert x_t) \approx
        -\lambda V \Sigma^\top \left(\sigma_y^2 \mathrm{Id} + \sigma_t^2 \Sigma \Sigma^\top\right)^{-1}
        \left(\Sigma V^\top \denoiser{x_t}{\sigma_t} - U^\top y\right),

    where :math:`\denoiser{\cdot}{\sigma}` is a denoising network for noise level :math:`\sigma`, and
    :math:`\lambda` (exposed as ``weight`` or ``etab``) controls the strength of the spectral measurement update.

    .. note::

        This method inherits from the general posterior sampling framework :class:`deepinv.sampling.PosteriorDiffusion`,
        using :class:`deepinv.sampling.DDRMDataFidelity`. It can be coupled with any denoiser, SDE schedule and solver
        (:class:`deepinv.sampling.DDIMSolver`, :class:`deepinv.sampling.DDPMSolver`,
        :class:`deepinv.sampling.AncestralSolver`, :class:`deepinv.sampling.EulerSolver`).

    :param deepinv.models.Denoiser denoiser: a denoiser model that can handle different noise levels.
    :param str schedule: the noise schedule to use, either ``"vp"`` (default) for the variance-preserving
        schedule or ``"ve"`` for the variance-exploding schedule.
    :param str, deepinv.sampling.BaseSDESolver solver: the solver of the reverse-time SDE, either a solver instance,
        or one of:

        - ``"ancestral"`` (default) for :class:`deepinv.sampling.AncestralSolver`,
        - ``"ddim"`` for :class:`deepinv.sampling.DDIMSolver` with stochasticity parameter ``eta``,
        - ``"ddpm"`` for :class:`deepinv.sampling.DDPMSolver`,
        - ``"euler"`` for :class:`deepinv.sampling.EulerSolver`.

        A solver instance is used as is, with its own time steps and random number generator, so that
        ``num_steps``, ``sigmas`` and ``rng`` are then ignored.
    :param Callable, float alpha: the weight of the noise in the reverse-time SDE, possibly time-dependent, see
        :class:`deepinv.sampling.DiffusionSDE`. If ``None``, calibrated from ``eta`` as ``20.0 * (eta / 0.85) ** 2``.
    :param int num_steps: the number of time steps of the solver (default: ``100``). Ignored if ``sigmas`` is provided.
    :param float eta: stochasticity parameter :math:`\eta \in [0, 1]` (default: ``0.85``).
    :param float etab: alias for ``weight`` (:math:`\eta_b` in :footcite:t:`kawar2022denoising`). Default: ``1.0``.
    :param float weight: weight :math:`\lambda` of the data-fidelity term. If ``None``, defaults to ``etab``.
    :param torch.Tensor, numpy.ndarray, list sigmas: optional decreasing sequence of noise levels in :math:`[0, 1]`
        used to construct the solver time steps. If ``None``, ``num_steps`` steps from ``1`` to ``0`` are used.
    :param float eps: small numerical floor for the spectral denominator. Default: ``1e-6``.
    :param tuple[float] clip: clipping interval for the denoised output, or ``None`` to disable
        clipping. Default: ``(0.0, 1.0)``.
    :param bool verbose: if ``True``, display a progress bar during sampling. Default: ``False``.
    :param str, torch.device device: device for the computations. If ``None``, inferred from ``denoiser`` or defaults to ``"cpu"``.
    :param torch.dtype dtype: data type of the solver computations. Default: ``torch.float64``.
    :param torch.Generator rng: optional random number generator for reproducibility.

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
        denoiser: Denoiser,
        schedule: str = "vp",
        solver: str | BaseSDESolver = "ancestral",
        alpha: Callable | float | None = None,
        num_steps: int = 100,
        eta: float = 0.85,
        etab: float = 1.0,
        weight: float | None = None,
        sigmas: torch.Tensor | np.ndarray | list | None = None,
        eps: float = 1e-6,
        clip: tuple[float, float] | None = (0.0, 1.0),
        verbose: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        rng: torch.Generator | None = None,
        **kwargs,
    ):
        if not isinstance(schedule, str):
            sigmas = schedule
            schedule = "vp"
        device = _infer_device(denoiser, device)
        weight = etab if weight is None else weight
        is_ancestral = isinstance(solver, str) and solver.lower() == "ancestral"
        if alpha is None:
            alpha = 20.0 * (eta / 0.85) ** 2 if is_ancestral else 1.0
        if clip == (0.0, 1.0) and not isinstance(denoiser, Denoiser):
            clip = None
        if sigmas is not None:
            num_steps = len(sigmas)

        solver_timesteps = sigmas
        if is_ancestral and schedule.lower() == "vp":
            if sigmas is None:
                u = torch.linspace(1, 0.0, num_steps, device=device, dtype=dtype)
            else:
                u = torch.as_tensor(sigmas, device=device, dtype=dtype).clone()
            solver_timesteps = 0.6 * (u**2)

        data_fidelity = DDRMDataFidelity(
            denoiser=denoiser,
            weight=weight,
            clip=clip,
            eps=eps,
        )
        sde, solver = _build_sde_and_solver(
            schedule=schedule,
            solver=solver,
            alpha=alpha,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            rng=rng,
            timesteps=solver_timesteps,
            eta=eta,
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
        self.denoiser = denoiser
        self.sigmas = sigmas if sigmas is not None else self.solver.timesteps
        self.max_iter = len(self.solver.timesteps)
        self.eta = eta
        self.etab = weight
        self.eps = eps


class DiffPIR(PosteriorDiffusion):
    r"""
    Diffusion PnP Image Restoration (DiffPIR).

    This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described in :footcite:t:`zhu2023denoising`.

    The DiffPIR algorithm combines a diffusion reverse process with a half-quadratic splitting (HQS) plug-and-play
    proximal data-consistency step. At each diffusion step, the denoised estimate :math:`\denoiser{x_t}{\sigma_t}`
    is updated with the proximal operator of the measurement data-fidelity function :math:`f(\cdot, y)`:

    .. math::

        \hat{x}_{0, t} = \operatorname{prox}_{f(\cdot, y) / \rho_t}\left(\denoiser{x_t}{\sigma_t}\right),
        \qquad \rho_t = \lambda \frac{\sigma_y^2}{\sigma_t^2},

    which corresponds to approximating the conditional score via :class:`deepinv.sampling.DiffPIRDataFidelity`:

    .. math::

        \nabla_{x_t} \log p_t(y \vert x_t) \approx
        \frac{\operatorname{prox}_{\frac{\sigma_t^2}{\lambda \sigma_y^2} f(\cdot, y)}\left(\denoiser{x_t}{\sigma_t}\right) - \denoiser{x_t}{\sigma_t}}{\sigma_t^2}.

    .. note::

        This method inherits from the general posterior sampling framework :class:`deepinv.sampling.PosteriorDiffusion`,
        using :class:`deepinv.sampling.DiffPIRDataFidelity`. It can be coupled with any denoiser, SDE schedule and solver
        (:class:`deepinv.sampling.AncestralSolver`, :class:`deepinv.sampling.DDIMSolver`,
        :class:`deepinv.sampling.DDPMSolver`, :class:`deepinv.sampling.EulerSolver`).
        The algorithm might require careful tuning of the hyperparameters :math:`\lambda` (``weight`` / ``lambda_``)
        and :math:`\zeta` (``alpha`` / ``zeta``) to obtain optimal results.

    :param deepinv.models.Denoiser model: a denoiser network that can handle different noise levels (alias for ``denoiser``).
    :param deepinv.optim.DataFidelity data_fidelity: the measurement data-fidelity operator :math:`f(\cdot, y)`.
        If ``None``, defaults to :class:`deepinv.optim.L2`.
    :param float sigma: fallback measurement noise standard deviation :math:`\sigma_y` used when
        ``physics.noise_model`` does not define ``sigma``. Default: ``0.05``.
    :param int max_iter: alias for ``num_steps`` (default: ``100``).
    :param float zeta: hyperparameter :math:`\zeta \in [0, 1]` controlling the stochasticity in the sampling step
        (alias for ``alpha``). Default: ``1.0``.
    :param float lambda_: hyperparameter :math:`\lambda` for the data-fidelity step
        (:math:`\rho_t = \lambda \sigma_y^2 / \sigma_t^2` in the paper, where optimal values typically range
        between ``3.0`` and ``25.0`` depending on the problem; alias for ``weight``). Default: ``7.0``.
    :param bool verbose: if ``True``, display a progress bar during sampling. Default: ``False``.
    :param str, torch.device device: the device to use for the computations. If ``None``, inferred from ``denoiser`` or defaults to ``"cpu"``.
    :param deepinv.models.Denoiser denoiser: a denoiser network that can handle different noise levels.
    :param str schedule: the noise schedule to use, either ``"vp"`` (default) for the variance-preserving
        schedule or ``"ve"`` for the variance-exploding schedule.
    :param str, deepinv.sampling.BaseSDESolver solver: the solver of the reverse-time SDE, either a solver instance,
        or one of:

        - ``"ancestral"`` (default) for :class:`deepinv.sampling.AncestralSolver`, which interpolates between
          deterministic DDIM (``alpha=0``) and DDPM (``alpha=1``),
        - ``"ddpm"`` for :class:`deepinv.sampling.DDPMSolver`,
        - ``"ddim"`` for :class:`deepinv.sampling.DDIMSolver`,
        - ``"euler"`` for :class:`deepinv.sampling.EulerSolver`.

        A solver instance is used as is, with its own time steps and random number generator, so that
        ``num_steps`` and ``rng`` are then ignored.
    :param Callable, float alpha: weight of the noise in the reverse-time SDE, see :class:`deepinv.sampling.DiffusionSDE`.
        If ``None``, defaults to ``zeta``.
    :param int num_steps: the number of time steps of the solver (default: ``100``).
    :param float weight: weight :math:`\lambda` of the data-fidelity term. If ``None``, defaults to ``lambda_``.
    :param tuple[float] clip: clipping interval for the denoised output before the proximal step, or ``None`` to
        disable clipping. Default: ``(0.0, 1.0)``.
    :param torch.dtype dtype: data type of the solver computations. Default: ``torch.float64``.
    :param torch.Generator rng: optional random number generator for reproducibility.

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
        model: Denoiser | None = None,
        data_fidelity: DataFidelity | None = None,
        sigma: float = 0.05,
        max_iter: int | None = None,
        zeta: float = 1.0,
        lambda_: float = 7.0,
        verbose: bool = False,
        device: str | torch.device | None = None,
        denoiser: Denoiser | None = None,
        schedule: str = "vp",
        solver: str | BaseSDESolver = "ancestral",
        alpha: Callable | float | None = None,
        num_steps: int = 100,
        weight: float | None = None,
        clip: tuple[float, float] | None = (0.0, 1.0),
        dtype: torch.dtype = torch.float64,
        rng: torch.Generator | None = None,
        **kwargs,
    ):
        denoiser = denoiser if denoiser is not None else model
        if denoiser is None:
            raise ValueError("Either `denoiser` or `model` must be provided.")
        device = _infer_device(denoiser, device)
        num_steps = max_iter if max_iter is not None else num_steps
        weight = weight if weight is not None else lambda_
        alpha = alpha if alpha is not None else zeta
        eta = kwargs.pop("eta", 0.0)

        noisy_data_fidelity = DiffPIRDataFidelity(
            denoiser=denoiser,
            data_fidelity=data_fidelity,
            sigma=sigma,
            weight=weight,
            clip=clip,
        )
        sde, solver = _build_sde_and_solver(
            schedule=schedule,
            solver=solver,
            alpha=alpha,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            rng=rng,
            eta=eta,
        )

        super().__init__(
            sde=sde,
            denoiser=denoiser,
            data_fidelity=noisy_data_fidelity,
            solver=solver,
            verbose=verbose,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.model = denoiser
        self.denoiser = denoiser
        self.lambda_ = weight
        self.max_iter = num_steps
        self.zeta = alpha
        self.sigma = sigma


class DPS(PosteriorDiffusion):
    r"""
    Diffusion Posterior Sampling (DPS).

    This class implements the Diffusion Posterior Sampling algorithm (DPS) described in :footcite:t:`chung2022diffusion`.

    DPS is an approximation of a gradient-based posterior sampling algorithm,
    which has minimal assumptions on the forward model. The only restriction is that
    the measurement model has to be differentiable, which is generally the case.

    The algorithm solves the reverse-time SDE specified by the `schedule` argument, using the solver specified by the `solver` argument, and approximating the conditional score by the DPS data fidelity term, which is defined as follows:

    .. math::

        \nabla_{x_t} \log p_t(y|x_t) \approx -\lambda \nabla_{x_t} \|y - A D_{\sigma_t}(x_t)\|

    where :math:`\denoiser{\cdot}{\sigma}` is a denoising network for noise level :math:`\sigma`, and :math:`\lambda` is a hyperparameter that controls the weight of the data fidelity term in the approximation of the likelihood gradient.

    .. note::

        This method inherits from the general posterior sampling framework :class:`deepinv.sampling.PosteriorDiffusion`, using the DPS data fidelity :class:`deepinv.sampling.DPSDataFidelity`.
        It can be coupled with any denoiser, SDE and solver, which allows for a wide range of sampling algorithms.
        The default parameters correspond to the original DPS algorithm in :cite:t:`chung2022diffusion`.

        Please refer to the example :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for more examples on diffusion-based sampling methods.

    :param deepinv.models.Denoiser denoiser: a denoiser network that can handle different noise levels
    :param str schedule: the noise schedule to use, either `"vp"` (default, which matches the original implementation) for the variance preserving noise schedule, or `"ve"` for the variance exploding noise schedule.
    :param str, deepinv.sampling.BaseSDESolver solver: the solver of the reverse-time SDE, either a solver instance, or one of:

        - `"ancestral"` (default) for :class:`deepinv.sampling.AncestralSolver`, which gives the DDPM sampler for `alpha=1` and the DDIM sampler for `alpha=0`,
        - `"ddpm"` for :class:`deepinv.sampling.DDPMSolver`, the sampler of the original implementation,
        - `"ddim"` for the deterministic :class:`deepinv.sampling.DDIMSolver`,
        - `"euler"` for :class:`deepinv.sampling.EulerSolver`.

        The `alpha` of the SDE is ignored by `"ddpm"` and `"ddim"`. A solver instance is used as is, with its own time steps and random number generator,
        so that `num_steps` and `rng` are then ignored.
    :param int num_steps: the number of time steps of the solver (default: 1000)
    :param Callable, float alpha: the weight of the noise in the reverse-time SDE, possibly time-dependent, see :class:`deepinv.sampling.DiffusionSDE`.
        Default to 1.0, which corresponds to the original DDPM sampling scheme. Setting it to 0 corresponds to the deterministic DDIM sampling scheme.
        Intermediate values differ from the parameter :math:`\eta` of DDIM, see :class:`deepinv.sampling.AncestralSolver` for the exact relation.
    :param float weight: the weight of the data fidelity term in the approximation of the likelihood gradient. Default to 1.0.
    :param str guidance: the form of the guidance, passed to :class:`deepinv.sampling.DPSDataFidelity`.
        `"norm"` (default) differentiates the residual norm, as in the original paper; `"annealed"` differentiates
        the Gaussian negative log-likelihood with the annealed variance :math:`\sigma_y^2 + \sigma_t^2`, which puts
        `weight` on the same scale as the other noisy data-fidelity terms.
    :param bool verbose: if `True`, print the progress of the algorithm
    :param str device: the device to use for the computations


    .. tip::

        For few steps sampling (e.g. `num_steps < 50`), the ancestral solvers (`"ancestral"`, `"ddpm"` or `"ddim"`) are recommended over `"euler"`.
        For many steps sampling, all solvers give similar results.

    """

    def __init__(
        self,
        denoiser: Denoiser,
        schedule: str = "vp",
        solver: str | BaseSDESolver = "ddpm",
        alpha: Callable | float = 1.0,
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
        sde, solver = _build_sde_and_solver(
            schedule=schedule,
            solver=solver,
            alpha=alpha,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            rng=rng,
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
