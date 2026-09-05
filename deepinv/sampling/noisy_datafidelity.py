from __future__ import annotations
import torch
from deepinv.optim import DataFidelity, Distance
import deepinv as dinv
from deepinv.physics import Physics
from deepinv.models import Denoiser


class NoisyDataFidelity(DataFidelity):
    r"""
    Preconditioned data fidelity term for noisy data :math:`- \log p(y|x + \sigma(t) \omega)`
    with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This is a base class for the conditional classes for approximating :math:`\log p_t(y|x_t)` used in diffusion
    algorithms for inverse problems, in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method computing the score :math:`\nabla_{x_t} \log p_t(y|x_t)`.

    By default we have

    .. math::

         \nabla_{x_t} \log p(y|x + \sigma(t) \omega) = P(\forw{x_t'}-y),


    where :math:`P` is a preconditioner and :math:`x_t'` is an estimation of the image :math:`x`.
    By default, :math:`P` is defined as :math:`A^\top`, :math:`x_t' = x_t` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.

    :param deepinv.optim.Distance d: Distance metric to use for the data fidelity term. Default to :class:`deepinv.optim.L2Distance`.
    :param float weight: Weighting factor for the data fidelity term. Default to 1.
    """

    def __init__(self, d: Distance = None, weight=1.0, *args, **kwargs):
        super().__init__()
        if d is not None:
            self.d = Distance(d)
        else:
            self.d = dinv.optim.L2Distance()
        self.weight = weight

    def precond(
        self, u: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        The preconditioner :math:`P` for the data fidelity term. Default to :math:`A^{\top}`.

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.

        :return: (torch.Tensor) preconditioned tensor :math:`P(u)`.
        """
        return (
            physics.A_adjoint(u)
            if isinstance(physics, dinv.physics.LinearPhysics)
            else physics.A_dagger(u)
        )

    def diff(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the difference :math:`A(x) - y` between the forward operator applied to the current iterate and the input data.


        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics), physics=physics)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y) * self.weight


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    Diffusion posterior sampling data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in
    `Diffusion Posterior Sampling for General Noisy Inverse Problems
    <https://arxiv.org/abs/2209.14687>`_.

    .. math::
            \nabla_x \log p_t(y|x)
            \approx
            -\lambda \nabla_x
            \|\forw{\denoiser{x}{\sigma}} - y\|.

    This class also supports latent diffusion models, corresponding to LDPS.
    By setting original_algo to True we can match the original discrete DPS algorithm
    ignoring the additional SDE and Euler weighting introduced during sampling:

    .. math::
            \text{compensation} = \frac{1}{dt * (1 + \alpha(t))/2 * g(t)^2}

    :param deepinv.models.Denoiser denoiser: Denoiser network.
    :param float weight: Weight of the data-fidelity term. Default to 1.0.
    :param bool original_algo: Whether to use the original DPS algorithm. Default to False.
    :param tuple[float] clip: Optional clipping interval for the denoised output.
    :param sde: SDE used for sampling. If provided together with ``timesteps``,
        the gradient is compensated to reproduce the discrete DPS update.
    :param torch.Tensor timesteps: Solver timesteps.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight: float = 1.0,
        original_algo: bool = False,
        clip: tuple = None,
        sde=None,
        timesteps=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        self.weight = weight
        self.original_algo = original_algo

        if clip is not None:
            if len(clip) != 2:
                raise ValueError(f"clip must be None or length 2, but got {clip}")
            clip = sorted(clip)

        self.clip = clip

        # Optional compensation of the SDE/Euler weighting.

        self._sigmas = None
        self._sde_step_weights = None

        if self.original_algo and sde is not None and timesteps is not None:
            with torch.no_grad():

                timesteps = timesteps.to(
                    device=sde.device,
                    dtype=sde.dtype,
                )

                t_cur = timesteps[:-1]
                dt = torch.abs(timesteps[1:] - timesteps[:-1])

                self._sigmas = torch.stack([sde.sigma_t(t) for t in t_cur]).flatten()

                self._sde_step_weights = torch.stack(
                    [
                        dti * (1 + sde.alpha(t)) / 2 * sde.forward_diffusion(t) ** 2
                        for t, dti in zip(t_cur, dt, strict=True)
                    ]
                ).flatten()

    def _sde_step_weight(self, sigma):
        """Return the Euler/SDE weight associated with ``sigma``."""

        if self._sde_step_weights is None:
            return None

        sigma = torch.as_tensor(
            sigma,
            device=self._sigmas.device,
            dtype=self._sigmas.dtype,
        ).flatten()[0]

        idx = torch.argmin(torch.abs(self._sigmas - sigma))

        return self._sde_step_weights[idx]

    def precond(
        self,
        x: torch.Tensor,
        physics: Physics,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        with torch.enable_grad():

            x = x.detach().requires_grad_(True)

            out = self.forward(
                x,
                y,
                physics,
                sigma,
                *args,
                get_model_outputs=get_model_outputs,
                **kwargs,
            )

            loss = out[0] if get_model_outputs else out

            grad = torch.autograd.grad(
                outputs=loss,
                inputs=x,
                grad_outputs=torch.ones_like(loss),
            )[0]

        # Cancel the additional SDE + Euler weighting to match the paper algorithm.
        if self.original_algo:
            step_weight = self._sde_step_weight(sigma)

        if self.original_algo and step_weight is not None:
            step_weight = step_weight.to(
                device=grad.device,
                dtype=grad.dtype,
            )

            grad = grad / step_weight.clamp_min(torch.finfo(grad.dtype).eps)

        if get_model_outputs:
            return grad, out[1].detach()

        return grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ):

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)

        x0_t = self.denoiser(
            x.to(torch.float32),
            sigma,
            *args,
            **kwargs,
        )

        # LDPS
        if getattr(self.denoiser, "vae", None) is not None:

            x0_t_dec = self.denoiser._decode(x0_t).clamp(0, 1)

            residual = physics.A(x0_t_dec) - y

            loss = residual.flatten(1).norm(dim=1) * self.weight

        # DPS in pixel space
        else:

            if self.clip is not None:
                x0_t = torch.clip(
                    x0_t,
                    self.clip[0],
                    self.clip[1],
                )

            residual = physics.A(x0_t) - y

            loss = residual.flatten(1).norm(dim=1) * self.weight

        if get_model_outputs:
            return loss, x0_t

        return loss


class PSLDDataFidelity(DPSDataFidelity):
    r"""
    Posterior Sampling with Latent Diffusion (PSLD) data-fidelity term.

    .. math::
            \omega \|A\mathcal{D}(\hat z_0)-y\|
            +
            \gamma \|\hat z_0-\mathcal{E}(A^\top y +
            (I-A^\top A)\mathcal{D}(\hat z_0))\|.

    By setting original_algo to True to match the original discrete PSLD algorithm,
    this class compensates for the additional SDE and Euler weighting introduced during sampling:

    .. math::
            \text{compensation} = \frac{1}{dt * (1 + \alpha(t))/2 * g(t)^2}

    :param deepinv.models.Denoiser denoiser: Latent diffusion denoiser with VAE.
    :param sde: Diffusion SDE used for sampling.
    :param torch.Tensor timesteps: Solver timesteps.
    :param bool original_algo: Whether to use the original DPS algorithm. Default to True.
    :param float omega: Weight of the measurement term. Default to 1.0.
    :param float gamma: Weight of the gluing term. Default to 0.1.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        sde,
        timesteps,
        original_algo: bool = True,
        omega: float = 1.0,
        gamma: float = 0.1,
    ):
        super().__init__(
            denoiser=denoiser,
            weight=1.0,
            clip=None,
        )

        self.original_algo = original_algo
        self.omega = omega
        self.gamma = gamma

        # Precompute the extra factor introduced by one Euler SDE step:
        if self.original_algo:
            with torch.no_grad():
                t_cur = timesteps[:-1].to(sde.device, sde.dtype)
                dt = torch.abs(timesteps[1:] - timesteps[:-1]).to(sde.device, sde.dtype)

                self._sigmas = torch.stack([sde.sigma_t(t) for t in t_cur]).flatten()

                self._sde_step_weights = torch.stack(
                    [
                        dti * (1 + sde.alpha(t)) / 2 * sde.forward_diffusion(t) ** 2
                        for t, dti in zip(t_cur, dt, strict=True)
                    ]
                ).flatten()

    def forward(
        self,
        x,
        y,
        physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ):
        if not isinstance(physics, dinv.physics.LinearPhysics):
            raise ValueError("PSLD requires a linear physics operator.")

        if getattr(self.denoiser, "vae", None) is None:
            raise ValueError("PSLD requires a latent model with a VAE.")

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.float()

        kwargs.setdefault("input_in_minus_one_one", True)

        # Denoised latent
        z0_hat = self.denoiser(
            x.float(),
            sigma,
            *args,
            **kwargs,
        )

        # Decode into image space
        x0_hat = self.denoiser._decode(z0_hat)

        # Measurement term
        residual = physics.A(x0_hat) - y

        measurement_loss = residual.flatten(1).norm(dim=1)

        # x* = A^T y + (I - A^T A) x
        x_glued = x0_hat + physics.A_adjoint(y - physics.A(x0_hat))

        # Re-encode projected image
        z_glued = self.denoiser._encode(
            x_glued,
            dtype=z0_hat.dtype,
        )

        glue_loss = (z0_hat - z_glued).flatten(1).norm(dim=1)

        loss = self.omega * measurement_loss + self.gamma * glue_loss

        if get_model_outputs:
            return loss, z0_hat

        return loss

    def grad(
        self,
        x,
        y,
        physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            out = self.forward(
                x,
                y,
                physics,
                sigma,
                *args,
                get_model_outputs=get_model_outputs,
                **kwargs,
            )

            loss = out[0] if get_model_outputs else out

            grad = torch.autograd.grad(
                loss,
                x,
                grad_outputs=torch.ones_like(loss),
            )[0]

        # Cancel the additional SDE + Euler weighting to match the paper algorithm.
        if self.original_algo:
            step_weight = self._sde_step_weight(sigma).to(
                device=grad.device,
                dtype=grad.dtype,
            )

            grad = grad / step_weight

        if get_model_outputs:
            return grad, out[1].detach()

        return grad
