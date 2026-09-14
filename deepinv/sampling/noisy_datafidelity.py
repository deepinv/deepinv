from __future__ import annotations
from typing import Callable
import torch
from deepinv.optim import DataFidelity
from deepinv.optim.linear import conjugate_gradient
import deepinv as dinv
from deepinv.physics import Physics
from deepinv.models import Denoiser


class NoisyDataFidelity(DataFidelity):
    r"""
    Data fidelity term for noisy input data :math:`- \log p(y|x + \sigma(t) \omega)` with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This is a base class used for approximating :math:`- \log p_t(y|x_t)` used in diffusion
    algorithms for inverse problems, in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method computing the negative log-likelihood gradient :math:` - \nabla_{x_t} \log p_t(y|x_t)`.

    By default we have

    .. math::

         - \nabla_{x_t} \log p(y|x + \sigma(t) \omega) = A^\top(\forw{x_t}-y),

    which subclasses override with their own approximation.

    .. note::

        Unlike :class:`deepinv.optim.DataFidelity`, these terms are defined through their
        gradient only: the approximations of :math:`- \log p_t(y|x_t)` used by diffusion
        samplers generally do not admit a closed-form value. The `.forward` method
        therefore raises a `NotImplementedError` unless a subclass defines it.

    :param float weight: Weighting factor for the data fidelity term. Default to 1.
    """

    def __init__(self, weight=1.0, *args, **kwargs):
        super().__init__()
        # these terms are defined through `.grad`, not through a distance, see `forward`.
        del self.d
        self.weight = weight

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term :math:`\lambda A^\top(A(x) - y)`.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :return: (torch.Tensor) data-fidelity term.
        """
        difference = physics.A(x) - y
        return self.weight * (
            physics.A_adjoint(difference)
            if isinstance(physics, dinv.physics.LinearPhysics)
            else physics.A_dagger(difference)
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        Not implemented: noisy data-fidelity terms are defined through their gradient
        :math:`-\nabla_{x_t} \log p_t(y|x_t)`, see :meth:`grad`. Subclasses whose
        approximation of :math:`- \log p_t(y|x_t)` has a closed form override this method.
        """
        raise NotImplementedError


class ALDDataFidelity(NoisyDataFidelity):
    r"""
    Score-based annealed Langevin dynamics (Score-ALD) data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in
    :cite:`jalal2021robust`, and reviewed in :cite:`daras2024survey`, given by

    .. math::

        p_t(y|x_t) \approx \mathcal{N} \left( y; A x_t,
        \left(\sigma_y^2 + \gamma_t^2\right)\mathrm{Id} \right).

    The resulting negative log-likelihood gradient is

    .. math::

        -\nabla_{x_t} \log p_t(y|x_t) \approx
        \lambda \frac{A^\top \left(A x_t - y\right)}{\sigma_y^2 + \gamma_t^2},

    where :math:`\sigma_y` is the measurement noise level and :math:`\lambda`,
    exposed as ``weight``, controls the scale of the data-fidelity term.

    .. note::

        :math:`\gamma_t` should decrease along the diffusion, so that the guidance
        strengthens as :math:`x_t` gets closer to the data manifold. The default
        ``gamma=None`` follows :cite:`jalal2021robust` and uses the current diffusion
        noise level, :math:`\gamma_t=\sigma_t`.

    :param Callable, float gamma: annealing parameter :math:`\gamma_t`. If `None`
        (default), :math:`\gamma_t = \sigma_t`, the current diffusion noise level. A
        `float` uses a constant value, and a `Callable` is evaluated as
        :math:`\gamma_t = \text{gamma}(\sigma_t)`.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    """

    def __init__(
        self,
        gamma: Callable | float | None = None,
        weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(weight=weight)
        self.gamma = gamma

    def _guidance_strength(
        self, physics: Physics, sigma: torch.Tensor | float, reference: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Computes the denominator :math:`\sigma_y^2 + \gamma_t^2` in the approximation of the gradient of the data fidelity term.

        The measurement noise level :math:`\sigma_y` is read from ``physics.noise_model.sigma`` when the noise is Gaussian, and is taken to be zero otherwise.

        :param deepinv.physics.Physics physics: physics model.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.
        :param torch.Tensor reference: tensor to broadcast against.
        :return: (:class:`torch.Tensor`) guidance strength :math:`\sigma_y^2 + \gamma_t^2`.
        """
        if self.gamma is None:
            gamma_t = sigma
        elif callable(self.gamma):
            gamma_t = self.gamma(sigma)
        else:
            gamma_t = self.gamma
        gamma_t = Denoiser._handle_sigma(gamma_t, reference_tensor=reference)

        noise_model = getattr(physics, "noise_model", None)
        if isinstance(noise_model, dinv.physics.GaussianNoise):
            sigma_y = Denoiser._handle_sigma(
                noise_model.sigma, reference_tensor=reference
            )
        else:
            sigma_y = torch.zeros_like(gamma_t)
        return sigma_y**2 + gamma_t**2

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the Score-ALD data-fidelity gradient.

        .. math::

            -\nabla_{x_t} \log p_t(y|x_t) \approx
            \lambda \frac{A^\top \left(A x_t - y\right)}{\sigma_y^2 + \gamma_t^2}.

        The measurement noise level :math:`\sigma_y` is read from ``physics.noise_model.sigma`` when the noise is Gaussian, and is taken to be zero otherwise.

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: physics model.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.

        :return: Score-ALD gradient, with the same shape as ``x``.
        """
        difference = physics.A(x) - y
        strength = self._guidance_strength(physics, sigma, difference)
        return self.weight * physics.A_adjoint(difference / strength)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Returns the loss term
        :math:`\lambda \| A x_t - y \|^2 / \left(2(\sigma_y^2 + \gamma_t^2)\right)`,
        whose gradient is given by :meth:`grad`.

        :param torch.Tensor x: input image.
        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward operator.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.

        :return: (:class:`torch.Tensor`) loss term, of size `B` the batch size.
        """
        difference = physics.A(x) - y
        strength = self._guidance_strength(physics, sigma, difference)
        squared_norm = (
            torch.linalg.vector_norm(
                difference, ord=2, dim=tuple(range(1, difference.ndim))
            )
            ** 2
        )
        return self.weight * squared_norm / (2 * strength.flatten())


class ScoreSDEDataFidelity(ALDDataFidelity):
    r"""
    Score-SDE data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in
    :cite:`song2020score`, and reviewed in :cite:`daras2024survey`. The difference with
    :class:`deepinv.sampling.ALDDataFidelity` is that the measurements are noised to the
    current diffusion noise level before the mismatch is computed,

    .. math::

        y_t = y + \sigma_t\epsilon, \qquad \epsilon\sim\mathcal{N}(0,\mathrm{Id}),

    so that :math:`y_t` and :math:`A x_t` live at the same noise level. The resulting
    negative log-likelihood gradient is

    .. math::

        -\nabla_{x_t} \log p_t(y|x_t) \approx
        \lambda \frac{A^\top \left(A x_t - y_t\right)}{\sigma_y^2 + \gamma_t^2},

    where :math:`\lambda`, exposed as ``weight``, controls the scale of the
    data-fidelity term.

    .. note::

        :cite:`daras2024survey` writes this approximation without a guidance strength,
        noting that it then differs from :class:`deepinv.sampling.ALDDataFidelity` only
        by the noising of the measurements. We keep the annealed guidance strength
        :math:`\sigma_y^2 + \gamma_t^2` here, so that the term stays balanced against
        the unconditional score across noise levels.

    :param Callable, float gamma: annealing parameter :math:`\gamma_t`. If `None`
        (default), :math:`\gamma_t = \sigma_t`, the current diffusion noise level.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    :param torch.Generator rng: Random number generator used to noise the measurements,
        for reproducibility. Default: `None`.
    """

    def __init__(
        self,
        gamma: Callable | float | None = None,
        weight: float = 1.0,
        rng: torch.Generator = None,
        *args,
        **kwargs,
    ):
        super().__init__(gamma=gamma, weight=weight)
        self.rng = rng

    def _noised_measurements(
        self, y: torch.Tensor, sigma: torch.Tensor | float
    ) -> torch.Tensor:
        r"""
        Draw the noised measurements :math:`y_t = y + \sigma_t\epsilon`.

        :param torch.Tensor y: Input data.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.
        :return: (:class:`torch.Tensor`) noised measurements :math:`y_t`.
        """
        noise = torch.randn(y.shape, generator=self.rng, device=y.device, dtype=y.dtype)
        sigma_t = Denoiser._handle_sigma(sigma, reference_tensor=y)
        return y + sigma_t * noise

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the Score-SDE data-fidelity gradient
        :math:`\lambda A^\top \left(A x_t - y_t\right) / (\sigma_y^2 + \gamma_t^2)`.

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: physics model.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.

        :return: Score-SDE gradient, with the same shape as ``x``.
        """
        difference = physics.A(x) - self._noised_measurements(y, sigma)
        strength = self._guidance_strength(physics, sigma, difference)
        return self.weight * physics.A_adjoint(difference / strength)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        Not implemented: the measurements are re-noised at every call, so this term has
        no deterministic value, see :meth:`grad`.
        """
        raise NotImplementedError


class ILVRDataFidelity(ScoreSDEDataFidelity):
    r"""
    Iterative Latent Variable Refinement (ILVR) data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in
    :cite:`choi2021ilvr`, and reviewed in :cite:`daras2024survey`. ILVR is a
    preconditioned version of :class:`deepinv.sampling.ScoreSDEDataFidelity`: the
    mismatch against the noised measurements :math:`y_t = y + \sigma_t\epsilon` is
    lifted back to the image space with the pseudo-inverse
    :math:`A^\dagger` instead of the adjoint :math:`A^\top`,

    .. math::

        -\nabla_{x_t} \log p_t(y|x_t) \approx
        \lambda \frac{A^\dagger \left(A x_t - y_t\right)}{\sigma_y^2 + \gamma_t^2},
        \qquad A^\dagger = \left(A^\top A\right)^{-1} A^\top,

    where :math:`\lambda`, exposed as ``weight``, controls the scale of the
    data-fidelity term.

    :param Callable, float gamma: annealing parameter :math:`\gamma_t`. If `None`
        (default), :math:`\gamma_t = \sigma_t`, the current diffusion noise level.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    :param torch.Generator rng: Random number generator used to noise the measurements,
        for reproducibility. Default: `None`.
    """

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the ILVR data-fidelity gradient
        :math:`\lambda A^\dagger \left(A x_t - y_t\right) / (\sigma_y^2 + \gamma_t^2)`.

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: physics model.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.

        :return: ILVR gradient, with the same shape as ``x``.
        """
        difference = physics.A(x) - self._noised_measurements(y, sigma)
        strength = self._guidance_strength(physics, sigma, difference)
        return self.weight * physics.A_dagger(difference / strength)


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    Diffusion posterior sampling data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in :cite:`chung2022diffusion`.
    For the VE parametrization :math:`x_t=x_0+\sigma_t\omega`, DPS replaces
    :math:`p(x_0|x_t)` by a Dirac mass at the denoised posterior mean:

    .. math::

        p(x_0|x_t)
        \approx \delta\!\left(x_0-D(x_t,\sigma_t)\right).

    Two guidance strengths are available, selected with ``guidance``.
    ``guidance="norm"`` (the default) follows :cite:`chung2022diffusion` and normalizes
    the residual by its own norm,

    .. math::

            -\nabla_x \log p_t(y|x) \approx \lambda \nabla_x \| \forw{\denoiser{x}{\sigma}} - y \|,

    which is the step size :math:`\zeta/\|y - A D(x_t,\sigma_t)\|` of the original paper.
    ``guidance="annealed"`` instead uses the Gaussian negative log-likelihood with the
    annealed variance :math:`\sigma_y^2+\sigma_t^2`,

    .. math::

            -\nabla_x \log p_t(y|x) \approx \lambda \nabla_x
            \frac{\| \forw{\denoiser{x}{\sigma}} - y \|^2}{2\left(\sigma_y^2+\sigma_t^2\right)},

    where :math:`\sigma = \sigma(t)` is the noise level and :math:`\lambda`
    controls the strength of the approximation.

    .. note::

        The two options put ``weight`` on very different scales. ``"norm"`` carries no
        noise variance, so :math:`\lambda` has to absorb a factor of order
        :math:`\|y - A D(x_t,\sigma_t)\|/\sigma_y^2`, which is typically in the hundreds.
        ``"annealed"`` shares the guidance strength of
        :class:`deepinv.sampling.ALDDataFidelity`, so :math:`\lambda\approx 1` is the
        natural choice, consistent with the other noisy data-fidelity terms.

    .. seealso::
        This class can be used for building custom DPS-based diffusion models.
        A self-contained implementation of the original DPS algorithm can be
        found in :class:`deepinv.sampling.DPS`.

    :param deepinv.models.Denoiser denoiser: Denoiser network
    :param float weight: Weighting factor for the data fidelity term. Default to 1.0 .
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    :param str guidance: Either ``"norm"`` (default), the residual norm of
        :cite:`chung2022diffusion`, or ``"annealed"``, the Gaussian negative
        log-likelihood with variance :math:`\sigma_y^2+\sigma_t^2`, for which
        ``weight`` is on the same scale as the other noisy data-fidelity terms.
    """

    def __init__(
        self,
        denoiser: Denoiser | None = None,
        weight: float = 1.0,
        clip: tuple = None,
        guidance: str = "norm",
        *args,
        **kwargs,
    ):
        super().__init__()
        if guidance not in ("norm", "annealed"):
            raise ValueError(
                f"guidance must be 'norm' or 'annealed', but got {guidance}."
            )
        self.denoiser = denoiser
        self.clip = sorted(clip) if clip is not None else None
        self.weight = weight
        self.guidance = guidance

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the gradient :math:`-\nabla_{x_t} \log p_t(y|x_t) \approx \lambda \nabla_{x_t} \| \forw{\denoiser{x}{\sigma}} - y \|`.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :param float, torch.Tensor sigma: Standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term (and denoised output if `get_model_outputs` is `True`).
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            out = self.forward(
                x,
                y,
                physics,
                sigma,
                *args,
                get_model_outputs=get_model_outputs,
                **kwargs,
            )
            # In case we also want the denoised output
            if get_model_outputs:
                l2_loss = out[0]
            else:
                l2_loss = out

            grad_outputs = torch.ones_like(l2_loss)
        norm_grad = torch.autograd.grad(
            outputs=l2_loss, inputs=x, grad_outputs=grad_outputs
        )[0]
        if get_model_outputs:
            return norm_grad, out[1].detach()
        else:
            return norm_grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the loss term
        :math:`\lambda \| \forw{\denoiser{x}{\sigma}} - y \|`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float, torch.Tensor sigma: standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the loss. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) loss term (and denoised output if `get_model_outputs` is `True`).
        """

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)
        x0_t = self.denoiser(x.to(torch.float32), sigma, *args, **kwargs)

        if self.clip is not None:
            x0_t = torch.clip(x0_t, self.clip[0], self.clip[1])  # optional

        difference = physics.A(x0_t) - y
        residual_norm = torch.linalg.vector_norm(
            difference, ord=2, dim=tuple(range(1, difference.ndim))
        )
        if self.guidance == "norm":
            out = self.weight * residual_norm
        else:
            noise_model = getattr(physics, "noise_model", None)
            sigma_y = (
                noise_model.sigma
                if isinstance(noise_model, dinv.physics.GaussianNoise)
                else 0.0
            )
            sigma_y = Denoiser._handle_sigma(sigma_y, reference_tensor=difference)
            sigma_t = Denoiser._handle_sigma(sigma, reference_tensor=difference)
            strength = (sigma_y**2 + sigma_t**2).flatten()
            out = self.weight * residual_norm**2 / (2 * strength)

        if get_model_outputs:
            return out, x0_t
        else:
            return out


class PiGDMDataFidelity(NoisyDataFidelity):
    r"""
    Pseudoinverse-guided diffusion model (PiGDM) data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in :cite:`song2023pseudoinverse`.
    For the VE parametrization :math:`x_t=x_0+\sigma_t\omega`, PiGDM uses the
    isotropic Gaussian approximation

    .. math::

        p(x_0|x_t) \approx \mathcal{N} \left( x_0;D(x_t,\sigma_t),\Sigma_t(x_t) \right),
        \qquad \Sigma_t(x_t)=r_t^2\mathrm{Id},
        \qquad r_t^2=\frac{\sigma_t^2}{1+\sigma_t^2}.

    For a linear forward operator and Gaussian measurement noise with standard
    deviation :math:`\sigma_y`,
    integrating this approximation gives a Gaussian approximation of
    :math:`p_t(y|x_t)`. Its negative log-likelihood gradient is

    .. math::

       -\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
        \left(r_t^2 A A^\top + \sigma_y^2\mathrm{Id}\right)^{-1}
        \left(A D(x_t, \sigma_t) - y\right).

    Here :math:`D` is a denoiser and :math:`J_D` is its Jacobian. The parameter
    :math:`\lambda`, exposed as ``weight``, controls the scale of the
    data-fidelity term. The inverse is evaluated
    exactly for :class:`deepinv.physics.DecomposablePhysics` operators and
    approximated with conjugate gradient for other linear operators.
    The measurement noise level :math:`\sigma_y` is read from
    ``physics.noise_model.sigma``.

    :param deepinv.models.Denoiser denoiser: Denoiser network. It may be left as
        ``None`` when the data fidelity is passed to
        :class:`deepinv.sampling.PosteriorDiffusion`, which supplies its denoiser.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    :param int cg_max_iter: Maximum number of conjugate-gradient iterations.
        Default: ``3``.
    :param float cg_tol: Relative conjugate-gradient tolerance. Default: ``1e-4``.
    :param bool verbose: If ``True``, print conjugate-gradient convergence
        information. Default: ``False``.
    """

    def __init__(
        self,
        denoiser: Denoiser | None = None,
        weight: float = 1.0,
        clip: tuple = None,
        cg_max_iter: int = 3,
        cg_tol: float = 1e-4,
        verbose: bool = False,
    ):
        super().__init__(weight=weight)
        self.denoiser = denoiser
        self.clip = sorted(clip) if clip is not None else None
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.verbose = verbose

    def solve_inverse(
        self,
        physics: Physics,
        u: torch.Tensor,
        r_t2: torch.Tensor | float,
        sigma_y: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""
        Apply
        :math:`(r_t^2 A A^\top + \sigma_y^2\mathrm{Id})^{-1}` to ``u``.

        :param deepinv.physics.Physics physics: Linear physics operator.
        :param torch.Tensor u: Tensor in the measurement space.
        :param torch.Tensor, float r_t2: PiGDM covariance parameter
            :math:`r_t^2`.
        :param torch.Tensor, float sigma_y: Measurement noise standard deviation
            :math:`\sigma_y`.
        :return: Solution in the measurement space.
        """
        if isinstance(physics, dinv.physics.DecomposablePhysics):
            transformed_u = physics.U_adjoint(u)
            r_t2 = Denoiser._handle_sigma(r_t2, reference_tensor=transformed_u)
            sigma_y = Denoiser._handle_sigma(sigma_y, reference_tensor=transformed_u)
            singular_values = physics.mask.to(
                device=transformed_u.device, dtype=transformed_u.dtype
            )
            denominator = sigma_y**2 + r_t2 * singular_values.conj() * singular_values
            return physics.U(transformed_u / denominator)

        r_t2 = Denoiser._handle_sigma(r_t2, reference_tensor=u)
        sigma_y = Denoiser._handle_sigma(sigma_y, reference_tensor=u)

        def operator(v):
            return sigma_y**2 * v + r_t2 * physics.A_A_adjoint(v)

        return conjugate_gradient(
            operator,
            u,
            max_iter=self.cg_max_iter,
            tol=self.cg_tol,
            verbose=self.verbose,
        )

    @torch.no_grad()
    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        get_model_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the PiGDM data-fidelity gradient.

        .. math::

               -\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
                \left(r_t^2 A A^\top + \sigma_y^2\mathrm{Id}\right)^{-1}
                \left(A D(x_t, \sigma_t) - y\right).

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Linear physics operator.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: PiGDM gradient, with the same shape and dtype as ``x``.
        """
        if not isinstance(physics, dinv.physics.LinearPhysics):
            raise ValueError("PiGDMDataFidelity only supports linear physics.")
        if self.denoiser is None:
            raise ValueError("PiGDMDataFidelity requires a denoiser.")
        input_dtype = x.dtype
        x_denoiser = x.detach().to(torch.float32)
        sigma_denoiser = Denoiser._handle_sigma(sigma, reference_tensor=x_denoiser)
        if not isinstance(physics.noise_model, dinv.physics.GaussianNoise):
            raise ValueError("This data fidelity requires Gaussian measurement noise.")
        sigma_y = physics.noise_model.sigma
        denoised, denoiser_vjp = torch.func.vjp(
            lambda z: self.denoiser(z, sigma_denoiser, *args, **kwargs),
            x_denoiser,
        )
        if self.clip is not None:
            denoised = torch.clip(denoised, self.clip[0], self.clip[1])
        measurement = physics.A(denoised)
        difference = measurement - y.to(
            device=measurement.device, dtype=measurement.dtype
        )
        r_t2 = sigma**2 / (1.0 + sigma**2)
        inverse_difference = self.solve_inverse(physics, difference, r_t2, sigma_y)
        adjoint = physics.A_adjoint(inverse_difference).to(denoised.dtype)
        gradient = denoiser_vjp(adjoint)[0]

        if get_model_outputs:
            return (self.weight * gradient).to(input_dtype), denoised.detach()
        else:
            return (self.weight * gradient).to(input_dtype)


class MomentMatchingDataFidelity(NoisyDataFidelity):
    r"""
    Moment-matching data-fidelity term for diffusion posterior sampling.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in :cite:`rozet2024learning`.
    For the VE parametrization, Moment Matching approximates the full conditional distribution with the
    Gaussian with mean and covariance given by the denoiser and its Jacobian:

    .. math::

        p(x_0|x_t) \approx \mathcal{N} \left(
        x_0;D(x_t,\sigma_t),\Sigma_t(x_t) \right),
        \qquad
        \Sigma_t(x_t)=\sigma_t^2J_D(x_t,\sigma_t).

    The resulting negative log-likelihood gradient is

    .. math::

        -\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
        \left(\sigma_t^2 A J_D(x_t, \sigma_t) A^\top
        + \sigma_y^2\mathrm{Id}\right)^{-1}
        \left(A D(x_t, \sigma_t) - y\right).

    The parameter :math:`\lambda`, exposed as ``weight``, controls the scale of
    the data-fidelity term. The Jacobian products are evaluated with
    vector-Jacobian products, without
    materializing the denoiser Jacobian, and the measurement-space system is
    approximated with conjugate gradient.
    The measurement noise level :math:`\sigma_y` is read from
    ``physics.noise_model.sigma``.

    .. note::

        Conjugate gradient assumes that the effective moment-matching operator
        is symmetric positive definite, as is expected for an exact MMSE
        denoiser covariance.

    :param deepinv.models.Denoiser denoiser: Denoiser network. It may be left as
        ``None`` when the data fidelity is passed to
        :class:`deepinv.sampling.PosteriorDiffusion`, which supplies its denoiser.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    :param int cg_max_iter: Maximum number of conjugate-gradient iterations.
        Default: ``3``.
    :param float cg_tol: Relative conjugate-gradient tolerance. Default: ``1e-4``.
    :param bool verbose: If ``True``, print conjugate-gradient convergence
        information. Default: ``False``.
    """

    def __init__(
        self,
        denoiser: Denoiser | None = None,
        weight: float = 1.0,
        clip: tuple = None,
        cg_max_iter: int = 3,
        cg_tol: float = 1e-4,
        verbose: bool = False,
    ):
        super().__init__(weight=weight)
        self.denoiser = denoiser
        self.clip = sorted(clip) if clip is not None else None
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.verbose = verbose

    @torch.no_grad()
    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        get_model_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the moment-matching data-fidelity gradient.

         .. math::

                -\nabla_{x_t} \log p_t(y|x_t) \approx \lambda J_D(x_t, \sigma_t)^\top A^\top
                \left(\sigma_t^2 A J_D(x_t, \sigma_t) A^\top
                + \sigma_y^2\mathrm{Id}\right)^{-1}
                \left(A D(x_t, \sigma_t) - y\right).

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Linear physics operator.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: Moment-matching gradient, with the same shape and dtype as
            ``x``.
        """
        if not isinstance(physics, dinv.physics.LinearPhysics):
            raise ValueError("MomentMatchingDataFidelity only supports linear physics.")
        if self.denoiser is None:
            raise ValueError("MomentMatchingDataFidelity requires a denoiser.")

        input_dtype = x.dtype
        x_denoiser = x.detach().to(torch.float32)
        if not isinstance(physics.noise_model, dinv.physics.GaussianNoise):
            raise ValueError("This data fidelity requires Gaussian measurement noise.")
        sigma_y = physics.noise_model.sigma
        sigma_denoiser = Denoiser._handle_sigma(sigma, reference_tensor=x_denoiser)
        denoised, denoiser_vjp = torch.func.vjp(
            lambda z: self.denoiser(z, sigma_denoiser, *args, **kwargs),
            x_denoiser,
        )
        if self.clip is not None:
            denoised = torch.clip(denoised, self.clip[0], self.clip[1])
        measurement = physics.A(denoised)
        difference = measurement - y.to(
            device=measurement.device, dtype=measurement.dtype
        )
        sigma_y = Denoiser._handle_sigma(sigma_y, reference_tensor=difference)
        sigma_t = Denoiser._handle_sigma(sigma, reference_tensor=difference)

        def operator(v):
            adjoint = physics.A_adjoint(v).to(denoised.dtype)
            covariance_product = denoiser_vjp(adjoint)[0]
            return sigma_y**2 * v + sigma_t**2 * physics.A(covariance_product)

        inverse_difference = conjugate_gradient(
            operator,
            difference,
            max_iter=self.cg_max_iter,
            tol=self.cg_tol,
            verbose=self.verbose,
        )
        adjoint = physics.A_adjoint(inverse_difference).to(denoised.dtype)
        gradient = denoiser_vjp(adjoint)[0]

        if get_model_outputs:
            return (self.weight * gradient).to(input_dtype), denoised.detach()
        else:
            return (self.weight * gradient).to(input_dtype)
