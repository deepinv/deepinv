from __future__ import annotations

import warnings
from typing import Callable
import torch
import numpy as np
from deepinv.models import Denoiser
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.utils.decorators import _deprecated_alias, _deprecated_argument

from deepinv.sampling import BaseSampling
from deepinv.sampling.sampling_iterators import DiffusionIterator
from deepinv.sampling.diffusion_sde import (
    DiffusionSDE,
    PosteriorDiffusion,
    VariancePreservingDiffusion,
    VarianceExplodingDiffusion,
    FlowMatching,
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
    HeunSolver,
)


def _create_sde(
    schedule: str | DiffusionSDE,
    alpha: Callable | float | None = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> DiffusionSDE:
    r"""Create the diffusion SDE given a schedule name with default parameters, ``"vp"`` creates :class:`VariancePreservingDiffusion`,
    ``"ve"`` creates :class:`VarianceExplodingDiffusion` and `"fm"` creates :class:`FlowMatching`, with ``alpha`` (default ``1.0``).

    An SDE instance is returned as is, with its own ``alpha``, so that ``alpha`` is then ignored.
    """
    sdes = {
        "vp": VariancePreservingDiffusion,
        "ve": VarianceExplodingDiffusion,
        "fm": FlowMatching,
    }
    if isinstance(schedule, DiffusionSDE):
        # alpha is captured by the drift and diffusion of the SDE, so it cannot be overridden after construction.
        if alpha is not None:
            warnings.warn(
                "`alpha` is ignored when `schedule` is a DiffusionSDE instance, which uses its own `alpha`. "
                "Set `alpha` when creating the SDE instead.",
                stacklevel=2,
            )
        sde = schedule
    elif isinstance(schedule, str):
        if schedule.lower() not in sdes:
            raise ValueError(f"schedule must be one of {list(sdes)}, got {schedule}")
        sde = sdes[schedule.lower()](
            alpha=1.0 if alpha is None else alpha, dtype=dtype, device=device
        )
    else:
        raise ValueError(
            f"schedule must be a DiffusionSDE instance or a string, got {type(schedule)}"
        )
    return sde


def _create_solver(
    solver: str | BaseSDESolver,
    num_steps: int,
    timesteps: torch.Tensor | np.ndarray | list | None = None,
    t_start: float = 1.0,
    eta: float = 0.0,
    zeta: float | None = None,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    rng: torch.Generator | None = None,
) -> BaseSDESolver:
    r"""
    Create a solver from its name, or return the given solver instance.

    Possible solvers are ``"ancestral"``, ``"ddpm"``, ``"ddim"``, ``"euler"``, and ``"heun"``, see :ref:`the user guide <sde_ode_solvers>` for more details.

    Without ``timesteps``, the solver takes ``num_steps`` time steps evenly spaced from ``t_start``.
    ``eta`` is only used by ``"ddim"`` and ``zeta`` by ``"ancestral"``.
    """
    if not isinstance(solver, str):
        return solver
    # The ancestral and Heun solvers never evaluate the denoiser at the last time step, so they can end at t = 0.
    solvers = {
        "ancestral": (AncestralSolver, 0.0, {"zeta": zeta}),
        "ddpm": (DDPMSolver, 0.0, {}),
        "ddim": (DDIMSolver, 0.0, {"eta": eta}),
        "euler": (EulerSolver, 0.001, {}),
        "heun": (HeunSolver, 0.0, {}),
    }
    if solver.lower() not in solvers:
        raise ValueError(
            f"solver must be one of {list(solvers)} or a BaseSDESolver instance, got {solver}"
        )
    solver_class, t_end, solver_kwargs = solvers[solver.lower()]
    if timesteps is None:
        timesteps = torch.linspace(
            t_start, t_end, num_steps, device=device, dtype=dtype
        )
    else:
        timesteps = torch.as_tensor(timesteps, device=device, dtype=dtype).clone()
        if len(timesteps) > 0 and timesteps[-1] == 0:
            timesteps[-1] = t_end
    return solver_class(timesteps=timesteps, rng=rng, **solver_kwargs)


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
    It inherits from the general posterior sampling framework :class:`deepinv.sampling.PosteriorDiffusion`,
    using :class:`deepinv.sampling.DDRMDataFidelity`. It can be coupled with any denoiser, SDE schedule (see :ref:`the user guide <diffusion_sdes>`) and solver
    (see :ref:`the user guide <sde_ode_solvers>`).

    DDRM samples from the posterior distribution of a linear inverse problem :math:`y = A x + \epsilon`, with
    :math:`\epsilon \sim \mathcal{N}(0, \sigma_y^2 \mathrm{Id})`, using a denoiser and the singular value decomposition (SVD)
    :math:`A = U \Sigma V^\top` of the operator (:class:`deepinv.physics.DecomposablePhysics`).
    In the SVD domain, the measurements decouple into scalar ones, :math:`\bar{y}_i = (U^\top y)_i / s_i`, with noise level
    :math:`\nu_i = \sigma_y / s_i` for each singular value :math:`s_i`. In the VE parametrization :math:`x_t = x_0 + \sigma_t \omega`,
    let :math:`\bar{x}_t = V^\top x_t` and :math:`\bar{x}_0 = V^\top \denoiser{x_t}{\sigma_t}` be the current state and
    the denoised estimate in the SVD domain. A step from :math:`t` to :math:`t - 1` samples each component of
    :math:`\bar{x}_{t-1} = V^\top x_{t-1}` as

    .. math::

        \bar{x}_{t-1, i} \sim \begin{cases}
        \mathcal{N}\left(\bar{x}_{0, i} + \sqrt{1 - \eta^2} \, \sigma_{t-1} \frac{\bar{x}_{t, i} - \bar{x}_{0, i}}{\sigma_t},
        \eta^2 \sigma_{t-1}^2\right) & \text{if } s_i = 0, \\
        \mathcal{N}\left(\bar{x}_{0, i} + \sqrt{1 - \eta^2} \, \sigma_{t-1} \frac{\bar{y}_i - \bar{x}_{0, i}}{\nu_i},
        \eta^2 \sigma_{t-1}^2\right) & \text{if } \sigma_{t-1} < \nu_i, \\
        \mathcal{N}\left((1 - \lambda) \bar{x}_{0, i} + \lambda \bar{y}_i, \sigma_{t-1}^2 - \lambda^2 \nu_i^2\right)
        & \text{if } \sigma_{t-1} \geq \nu_i,
        \end{cases}

    where :math:`\eta` (``eta``) controls the stochasticity and :math:`\lambda` (``weight``) the weight of the measurements.

    The default schedule and solver are chosen to match the original implementaton of :footcite:t:`kawar2022denoising`.


    :param deepinv.models.Denoiser denoiser: a denoiser model that can handle different noise levels.
    :param str, deepinv.sampling.DiffusionSDE schedule: the diffusion SDE (see :ref:`the user guide <diffusion_sdes>`),
        either an SDE instance or a string, ``"vp"`` (default) for :class:`VariancePreservingDiffusion`, 
        ``"ve"`` for :class:`VarianceExplodingDiffusion` or `"fm"` for :class:`FlowMatching`. 
        An instance of :class:`EDMDiffusionSDE` with custom schedules is also possible. An SDE instance is used as is,
        with its own ``alpha``, so that ``alpha`` is then ignored.
    :param str, deepinv.sampling.BaseSDESolver solver: the solver of the reverse-time SDE (see :ref:`the user guide <sde_ode_solvers>`),
        either a solver instance or a string, e.g. ``"euler"`` for
        :class:`deepinv.sampling.EulerSolver`. The default ``"ancestral"`` uses ``zeta = eta**2``, and ``"ddim"`` uses ``eta`` (stochasticity paramter, see below).
        A solver instance is used as is, with its own time steps and random number generator, so that
        ``num_steps`` and ``rng`` are then ignored.
    :param Callable, float alpha: the weight of the noise in the reverse-time SDE, possibly time-dependent, see
        :class:`deepinv.sampling.DiffusionSDE`. Only used with a string ``schedule``, and only by ``"euler"``, ``"heun"``
        and by solver instances without ``eta`` or ``zeta``. If ``None`` (default), ``1.0`` is used.
    :param int num_steps: the number of time steps of the solver. Default: ``100``.
    :param float eta: stochasticity parameter :math:`\eta \in [0, 1]` of DDRM, used by the solver and by
        :class:`deepinv.sampling.DDRMDataFidelity` (default: ``0.85``).
    :param float weight: weight :math:`\lambda` of the data-fidelity term, denoted :math:`\eta_b` in
        :footcite:t:`kawar2022denoising`. It replaces the deprecated ``etab``. Default: ``1.0``.
    :param float eps: small numerical floor for the spectral denominator. Default: ``1e-6``.
    :param tuple[float] clip: clipping interval for the denoised output, or ``None`` to disable
        clipping. Default: ``(0.0, 1.0)``.
    :param bool verbose: if ``True``, display a progress bar during sampling. Default: ``False``.
    :param torch.dtype dtype: data type of the solver computations. Default: ``torch.float64``.
    :param str, torch.device device: device for the computations. Default: ``"cpu"``.
    :param torch.Generator rng: optional random number generator for reproducibility.

    |sep|

    :Examples:

        DDRM with a pretrained DRUNet denoiser, for the inpainting of a noisy :math:`32 \times 32` image:

        >>> import deepinv as dinv
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> x = dinv.utils.load_example("butterfly.png", img_size=32)
        >>> physics = dinv.physics.Inpainting(
        ...     img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
        ... )
        >>> y = physics(x)
        >>> denoiser = dinv.models.DRUNet(pretrained="download")
        >>> model = dinv.sampling.DDRM(denoiser=denoiser, num_steps=50)
        >>> xhat = model(y, physics)
        >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)
        tensor([True])
        >>> dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DDRM"])  # doctest: +SKIP

    .. plot::

        import deepinv as dinv
        import torch
        torch.manual_seed(0)
        x = dinv.utils.load_example("butterfly.png", img_size=32)
        physics = dinv.physics.Inpainting(
            img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
        )
        y = physics(x)
        denoiser = dinv.models.DRUNet(pretrained="download")
        model = dinv.sampling.DDRM(denoiser=denoiser, num_steps=50)
        xhat = model(y, physics)
        dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DDRM"])

    """

    @_deprecated_argument(
        "sigmas",
        message="The 'sigmas' argument is deprecated and will be removed in a future version. Use `schedule` and `num_steps` instead.",
    )
    @_deprecated_alias(etab="weight")
    def __init__(
        self,
        denoiser: Denoiser,
        schedule: str | DiffusionSDE = "vp",
        solver: str | BaseSDESolver = "ancestral",
        alpha: Callable | float | None = None,
        num_steps: int = 100,
        eta: float = 0.85,
        weight: float = 1.0,
        eps: float = 1e-6,
        clip: tuple[float, float] | None = (0.0, 1.0),
        verbose: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        rng: torch.Generator | None = None,
        **kwargs,
    ):
        if isinstance(schedule, (torch.Tensor, np.ndarray, list)):
            warnings.warn(
                "Argument 'sigmas' is deprecated and will be removed in a future version. "
                "It is ignored, use 'num_steps' and 'schedule' instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            schedule = "vp"

        is_ancestral = isinstance(solver, str) and solver.lower() == "ancestral"
        if clip == (0.0, 1.0) and not isinstance(denoiser, Denoiser):
            clip = None

        solver_timesteps = None
        if is_ancestral and isinstance(schedule, str) and schedule.lower() == "vp":
            u = torch.linspace(1, 0.0, num_steps, device=device, dtype=dtype)
            solver_timesteps = 0.6 * (u**2)

        data_fidelity = DDRMDataFidelity(
            denoiser=denoiser,
            weight=weight,
            eta=eta,
            clip=clip,
            eps=eps,
        )
        # Create the SDE and solver instances
        sde = _create_sde(schedule, alpha=alpha, dtype=dtype, device=device)

        solver = _create_solver(
            solver,
            num_steps,
            timesteps=solver_timesteps,
            t_start=sde.T,
            eta=eta,
            zeta=eta**2,
            dtype=dtype,
            device=device,
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
        self.denoiser = denoiser
        self.max_iter = len(self.solver.timesteps)
        self.eta = eta
        self.eps = eps


class DiffPIR(PosteriorDiffusion):
    r"""
    Diffusion PnP Image Restoration (DiffPIR).

    This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described in :footcite:t:`zhu2023denoising`.

    The DiffPIR algorithm combines a diffusion reverse process with a half-quadratic splitting (HQS) plug-and-play
    proximal data-consistency step. At each diffusion step, the denoised estimate :math:`\denoiser{x_t}{\sigma_t}`
    is updated with the proximal operator of the measurement data-fidelity function :math:`f`:

    .. math::

        \hat{x}_{0, t} = \operatorname{prox}_{\frac{1}{\rho_t} f}\left(\denoiser{x_t}{\sigma_t}\right),
        \qquad \rho_t = \lambda \frac{\sigma_y^2}{\sigma_t^2},

    which corresponds to approximating the conditional score via :class:`deepinv.sampling.DiffPIRDataFidelity`:

    .. math::

        \nabla_{x_t} \log p_t(y \vert x_t) \approx
        \frac{\hat{x}_{0, t} - \denoiser{x_t}{\sigma_t}}{\sigma_t^2}.

    With the default ``"ancestral"`` solver and ``"vp"`` schedule, each step is the update of :footcite:t:`zhu2023denoising`.
    Let :math:`s_t` and :math:`\sigma_t` be the scale and noise level of the forward SDE, i.e.
    :math:`x_t = s_t \cdot x_0 + s_t \sigma_t \cdot \omega` with :math:`\omega \sim \mathcal{N}(0, \mathrm{Id})`
    (see :class:`deepinv.sampling.EDMDiffusionSDE`). A step of :class:`deepinv.sampling.AncestralSolver` from :math:`t`
    to :math:`t - 1` is then:

    .. math::

        x_{t - 1} = \frac{s_{t-1}}{s_t} x_t + s_t s_{t-1} \sigma_t^2 \left(1 - r^{1 + \alpha}\right) \nabla_{x_t} \log p_t(x_t \vert y)
        + s_{t-1} \sigma_{t-1} \sqrt{1 - r^{2 \alpha}} z,
        
    with :math:`r = \frac{\sigma_{t-1}}{\sigma_t}` and :math:`z \sim \mathcal{N}(0, \mathrm{Id})`.

    The parameter :math:`\zeta` of DiffPIR sets :math:`r^{2 \alpha} = 1 - \zeta`, so that a fraction :math:`\zeta` of the noise variance is renewed on each step.

    The conditional score is the sum of the prior score, given by the denoised estimate :math:`\denoiser{x_t / s_t}{\sigma_t}`
    through Tweedie's formula, and of the likelihood term above, divided by :math:`s_t` by the chain rule since the denoiser
    is applied to :math:`x_t / s_t`:

    .. math::

        \nabla_{x_t} \log p_t(x_t \vert y)
        &= \frac{s_t \denoiser{x_t / s_t}{\sigma_t} - x_t}{s_t^2 \sigma_t^2}
        + \frac{\hat{x}_{0, t} - \denoiser{x_t / s_t}{\sigma_t}}{s_t \sigma_t^2} \\
        &= \frac{s_t \hat{x}_{0, t} - x_t}{s_t^2 \sigma_t^2} \\
        &= -\frac{\hat{\varepsilon}}{s_t \sigma_t}, \qquad \hat{\varepsilon} = \frac{x_t - s_t \hat{x}_{0, t}}{s_t \sigma_t}.

    The data-fidelity term thus replaces the denoised estimate by :math:`\hat{x}_{0, t}` in Tweedie's formula, and
    :math:`\hat{\varepsilon}` is the effective noise, such that :math:`x_t = s_t \hat{x}_{0, t} + s_t \sigma_t \hat{\varepsilon}`.
    Substituting :math:`x_t` and the score in the ancestral step gives:

    .. math::

        x_{t - 1} &= s_{t-1} \left(\hat{x}_{0, t} + \sigma_t \hat{\varepsilon}\right)
        - s_{t-1} \sigma_t \left(1 - \frac{\sigma_{t-1}}{\sigma_t} \sqrt{1 - \zeta}\right) \hat{\varepsilon}
        + s_{t-1} \sigma_{t-1} \sqrt{\zeta} \, z \\
        &= s_{t-1} \hat{x}_{0, t} + s_{t-1} \sigma_{t-1} \left(\sqrt{1 - \zeta} \, \hat{\varepsilon} + \sqrt{\zeta} \, z\right).

    For :class:`deepinv.sampling.VariancePreservingDiffusion`, :math:`s_t = \sqrt{\bar{\alpha}_t}` and
    :math:`s_t \sigma_t = \sqrt{1 - \bar{\alpha}_t}`, which gives the sampling step of DiffPIR in :footcite:t:`zhu2023denoising`:

    .. math::

        \hat{\varepsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t} \hat{x}_{0, t}}{\sqrt{1 - \bar{\alpha}_t}}, \qquad
        x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_{0, t} + \sqrt{1 - \bar{\alpha}_{t-1}} \left(\sqrt{1 - \zeta} \, \hat{\varepsilon} + \sqrt{\zeta} \, z\right).

    The VP schedule is the continuous-time limit of the discrete :math:`\bar{\alpha}_t` of DDPM, whose noise levels differ by a few percent.
    When a different SDE is used, the sampling step uses the same estimate of the conditional score, but :math:`s_t` and :math:`\sigma_t` are given by the SDE, 
    making the sampler more flexible than the original DiffPIR algorithm.

    Note that :math:`\sigma_t` and :math:`\sigma_y` are both noise levels in :math:`[0, 1]`. The original code of :cite:`zhu2023denoising` uses the noise level in :math:`[-1, 1]`, 
    therefore the values of :math:`\lambda` is 4 times smaller than in the original code.

    .. note::

        This method inherits from the general posterior sampling framework :class:`deepinv.sampling.PosteriorDiffusion`,
        using :class:`deepinv.sampling.DiffPIRDataFidelity`. It can be coupled with any denoiser, SDE schedule and solver
        (see :ref:`the user guide <sde_ode_solvers>`).
        The algorithm might require careful tuning of the hyperparameters :math:`\lambda` (``weight``)
        and :math:`\zeta` (``zeta``) to obtain optimal results.

    :param deepinv.models.Denoiser denoiser: a denoiser network that can handle different noise levels.
    :param deepinv.optim.DataFidelity data_fidelity: the measurement data-fidelity operator :math:`f(\cdot, y)`.
        If ``None``, defaults to :class:`deepinv.optim.L2`.
    :param float sigma: fallback measurement noise standard deviation :math:`\sigma_y` used when
        ``physics.noise_model`` does not define ``sigma``. Default: ``0.05``.
    :param int max_iter: alias for ``num_steps`` (default: ``100``).
    :param float zeta: fraction :math:`\zeta \in [0, 1]` of the noise variance renewed on each step of the ``"ancestral"``
        solver, controlling the stochasticity of the sampling step. Default: ``1.0``.
    :param float weight: weight :math:`\lambda` of the data-fidelity step, in :math:`\rho_t = \lambda \sigma_y^2 / \sigma_t^2`
        with both noise levels in the :math:`[0, 1]` range of the images. The typical values of :footcite:t:`zhu2023denoising`,
        between ``3.0`` and ``25.0`` depending on the problem.
        It replaces the deprecated ``lambda_``. Default: ``7.0``.
    :param bool verbose: if ``True``, display a progress bar during sampling. Default: ``False``.
    :param str, deepinv.sampling.DiffusionSDE schedule: the diffusion SDE (see :ref:`the user guide <diffusion_sdes>`),
        either an SDE instance or a string, ``"vp"`` (default) for :class:`deepinv.sampling.VariancePreservingDiffusion`
        or ``"ve"`` for :class:`deepinv.sampling.VarianceExplodingDiffusion`. An SDE instance, e.g. with a custom noise
        level ``sigma_t``, is used as is, with its own ``alpha``, so that ``alpha`` is then ignored.
    :param str, deepinv.sampling.BaseSDESolver solver: the solver of the reverse-time SDE (see :ref:`the user guide <sde_ode_solvers>`),
        either a solver instance or the name of a solver class in lowercase without ``Solver``, e.g. ``"euler"`` for
        :class:`deepinv.sampling.EulerSolver`. The default ``"ancestral"`` uses ``zeta``.
        A solver instance is used as is, with its own time steps and random number generator, so that
        ``num_steps`` and ``rng`` are then ignored.
    :param Callable, float alpha: weight of the noise in the reverse-time SDE, see :class:`deepinv.sampling.DiffusionSDE`.
        Only used with a string ``schedule``, and only by ``"euler"``, ``"heun"`` and by solver instances without ``eta`` or ``zeta``.
        If ``None`` (default), ``1.0`` is used.
    :param int num_steps: the number of time steps of the solver (default: ``100``).
    :param tuple[float] clip: clipping interval for the denoised output before the proximal step, or ``None`` to
        disable clipping. Default: ``(0.0, 1.0)``.
    :param torch.dtype dtype: data type of the solver computations. Default: ``torch.float64``.
    :param str, torch.device device: the device to use for the computations. Default: ``"cpu"``.
    :param torch.Generator rng: optional random number generator for reproducibility.

    |sep|

    :Examples:

        DiffPIR with a pretrained DRUNet denoiser, for the inpainting of a noisy :math:`32 \times 32` image:

        >>> import deepinv as dinv
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> x = dinv.utils.load_example("butterfly.png", img_size=32)
        >>> physics = dinv.physics.Inpainting(
        ...     img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
        ... )
        >>> y = physics(x)
        >>> denoiser = dinv.models.DRUNet(pretrained="download")
        >>> model = dinv.sampling.DiffPIR(denoiser=denoiser, num_steps=50)
        >>> xhat = model(y, physics)
        >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)
        tensor([True])
        >>> dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DiffPIR"])  # doctest: +SKIP

    .. plot::

        import deepinv as dinv
        import torch
        torch.manual_seed(0)
        x = dinv.utils.load_example("butterfly.png", img_size=32)
        physics = dinv.physics.Inpainting(
            img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
        )
        y = physics(x)
        denoiser = dinv.models.DRUNet(pretrained="download")
        model = dinv.sampling.DiffPIR(denoiser=denoiser, num_steps=50)
        xhat = model(y, physics)
        dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DiffPIR"])

    """

    @_deprecated_alias(lambda_="weight")
    def __init__(
        self,
        denoiser: Denoiser | None = None,
        data_fidelity: DataFidelity | None = None,
        sigma: float = 0.05,
        max_iter: int | None = None,
        zeta: float = 1.0,
        weight: float = 7.0,
        verbose: bool = False,
        schedule: str | DiffusionSDE = "vp",
        solver: str | BaseSDESolver = "ancestral",
        alpha: Callable | float | None = None,
        num_steps: int = 100,
        clip: tuple[float, float] | None = (0.0, 1.0),
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        rng: torch.Generator | None = None,
        **kwargs,
    ):
        if denoiser is None:
            raise ValueError("A `denoiser` must be provided.")
        num_steps = max_iter if max_iter is not None else num_steps
        eta = kwargs.pop("eta", 0.0)

        noisy_data_fidelity = DiffPIRDataFidelity(
            denoiser=denoiser,
            data_fidelity=data_fidelity,
            sigma=sigma,
            weight=weight,
            clip=clip,
        )
        sde = _create_sde(schedule, alpha=alpha, dtype=dtype, device=device)
        solver = _create_solver(
            solver,
            num_steps,
            t_start=sde.T,
            eta=eta,
            zeta=zeta,
            dtype=dtype,
            device=device,
            rng=rng,
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
        self.max_iter = num_steps
        self.zeta = zeta
        self.sigma = sigma


class DPS(PosteriorDiffusion):
    r"""
    Diffusion Posterior Sampling (DPS).

    This class implements the Diffusion Posterior Sampling algorithm (DPS) described in :footcite:t:`chung2022diffusion`.

    DPS samples from the posterior distribution of an inverse problem :math:`y = A(x) + \epsilon`, with
    :math:`\epsilon \sim \mathcal{N}(0, \sigma_y^2 \mathrm{Id})` and :math:`A` can be nonlinear. It solves the reverse-time SDE given by ``schedule``
    with ``solver``, where the score of the prior is replaced by the conditional score, given by Bayes' rule:

    .. math::

        \nabla_{x_t} \log p_t(x_t \vert y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y \vert x_t).

    The prior score is given by the denoiser :math:`\denoiser{\cdot}{\sigma}` through Tweedie's formula.

    The likelihood :math:`p_t(y \vert x_t)` is intractable.
    In the VE parametrerization, :footcite:t:`chung2022diffusion` estimates its score by:

    .. math::

        \nabla_{x_t} \log p_t(y \vert x_t) \approx -\lambda \nabla_{x_t} \|y - A(\denoiser{x_t}{\sigma_t})\|,

    where :math:`\lambda` is the ``weight`` of the data-fidelity term. With ``guidance="annealed"``, it is the score of a
    Gaussian likelihood whose variance is increased by :math:`\sigma_t^2` to account for the error of the denoised estimate:

    .. math::

        \nabla_{x_t} \log p_t(y \vert x_t) \approx
        -\lambda \nabla_{x_t} \frac{\|y - A(\denoiser{x_t}{\sigma_t})\|^2}{2 (\sigma_y^2 + \sigma_t^2)},

    which puts :math:`\lambda` on the same scale as the other noisy data-fidelity terms, see :class:`deepinv.sampling.DPSDataFidelity`.
    Both gradients are computed by backpropagation through the denoiser, so that each step of DPS costs a backward pass
    on top of the forward pass of the denoiser, unlike :class:`deepinv.sampling.DiffPIR` and :class:`deepinv.sampling.DDRM`.

    .. note::

        This method inherits from the general posterior sampling framework :class:`deepinv.sampling.PosteriorDiffusion`, using the DPS data fidelity :class:`deepinv.sampling.DPSDataFidelity`.
        It can be coupled with any denoiser, SDE and solver, which allows for a wide range of sampling algorithms.
        The default schedule and solver correspond to the original DPS algorithm in :cite:t:`chung2022diffusion`.

        See :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_samplers.py` for a comparison with DiffPIR and DDRM, and
        :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for more examples on diffusion-based sampling methods.

    :param deepinv.models.Denoiser denoiser: a denoiser network that can handle different noise levels
    :param str, deepinv.sampling.DiffusionSDE schedule: the diffusion SDE (see :ref:`the user guide <diffusion_sdes>`), either an SDE instance or a string,
        `"vp"` (default, which matches the original implementation) for :class:`deepinv.sampling.VariancePreservingDiffusion`
        or `"ve"` for :class:`deepinv.sampling.VarianceExplodingDiffusion`.
        An SDE instance, e.g. with a custom noise level `sigma_t`, is used as is, with its own `alpha`, so that `alpha` is then ignored.
    :param str, deepinv.sampling.BaseSDESolver solver: the solver of the reverse-time SDE, either a solver instance, or one of:

        - `"ancestral"` (default) for :class:`deepinv.sampling.AncestralSolver`, which gives the DDPM sampler for `alpha=1` and the DDIM sampler for `alpha=0`,
        - `"ddpm"` for :class:`deepinv.sampling.DDPMSolver`, the sampler of the original implementation,
        - `"ddim"` for the deterministic :class:`deepinv.sampling.DDIMSolver`,
        - `"euler"` for :class:`deepinv.sampling.EulerSolver`,
        - `"heun"` for :class:`deepinv.sampling.HeunSolver`.

        The `alpha` of the SDE is ignored by `"ddpm"` and `"ddim"`. A solver instance is used as is, with its own time steps and random number generator,
        so that `num_steps` and `rng` are then ignored.
    :param int num_steps: the number of time steps of the solver (default: 1000)
    :param Callable, float alpha: the weight of the noise in the reverse-time SDE, possibly time-dependent, see :class:`deepinv.sampling.DiffusionSDE`.
        Only used with a string `schedule`. If `None` (default), `1.0` is used, which corresponds to the original DDPM sampling scheme.
        Setting it to 0 corresponds to the deterministic DDIM sampling scheme.
        Intermediate values differ from the parameter :math:`\eta` of DDIM, see :class:`deepinv.sampling.AncestralSolver` for the exact relation.
    :param float weight: the weight of the data fidelity term in the approximation of the likelihood gradient. Default to 1.0.
    :param str guidance: the form of the guidance, passed to :class:`deepinv.sampling.DPSDataFidelity`.
        `"norm"` (default) differentiates the residual norm, as in the original paper; `"annealed"` differentiates
        the Gaussian negative log-likelihood with the annealed variance :math:`\sigma_y^2 + \sigma_t^2`, which puts
        `weight` on the same scale as the other noisy data-fidelity terms.
    :param bool verbose: if `True`, print the progress of the algorithm
    :param str device: the device to use for the computations


    .. tip::

        For few steps sampling (e.g. `num_steps < 50`), the ancestral solvers (`"ancestral"`, `"ddpm"` or `"ddim"`) are recommended over `"euler"` and `"heun"`.
        For many steps sampling, all solvers give similar results.

    |sep|

    :Examples:

        DPS with a pretrained DRUNet denoiser, for the inpainting of a noisy :math:`32 \times 32` image:

        >>> import deepinv as dinv
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> x = dinv.utils.load_example("butterfly.png", img_size=32)
        >>> physics = dinv.physics.Inpainting(
        ...     img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
        ... )
        >>> y = physics(x)
        >>> denoiser = dinv.models.DRUNet(pretrained="download")
        >>> model = dinv.sampling.DPS(denoiser=denoiser, num_steps=50, weight=5.0, guidance="annealed")
        >>> xhat = model(y, physics)
        >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x)
        tensor([True])
        >>> dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DPS"])  # doctest: +SKIP

    .. plot::

        import deepinv as dinv
        import torch
        torch.manual_seed(0)
        x = dinv.utils.load_example("butterfly.png", img_size=32)
        physics = dinv.physics.Inpainting(
            img_size=(3, 32, 32), mask=0.7, noise_model=dinv.physics.GaussianNoise(0.05)
        )
        y = physics(x)
        denoiser = dinv.models.DRUNet(pretrained="download")
        model = dinv.sampling.DPS(denoiser=denoiser, num_steps=50, weight=5.0, guidance="annealed")
        xhat = model(y, physics)
        dinv.utils.plot([x, y, xhat], titles=["Ground truth", "Measurement", "DPS"])

    """

    def __init__(
        self,
        denoiser: Denoiser,
        schedule: str | DiffusionSDE = "vp",
        solver: str | BaseSDESolver = "ddpm",
        alpha: Callable | float | None = None,
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
        sde = _create_sde(schedule, alpha=alpha, dtype=dtype, device=device)
        solver = _create_solver(
            solver, num_steps, t_start=sde.T, dtype=dtype, device=device, rng=rng
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
