from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from deepinv.physics import Physics
from deepinv.models.latentden import LatentDiffusion


class DDIMDiffusion(nn.Module):
    r"""
    DDIM sampler (:math:`\eta \ge 0`).
    
    Implements the DDIM update of :footcite:t:`song2020denoising` for a latent
    trajectory :math:`z_T \rightarrow \cdots \rightarrow z_0`. For each step
    :math:`t \rightarrow t-1`, with cumulative schedule :math:`\bar{\alpha}_t`,
    we define the proxy clean latent
    
    .. math::
    
       z_0(z_t)
       \;=\;
       \frac{z_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(z_t,t)}
            {\sqrt{\bar{\alpha}_t}}.
    
    The **DDIM** update is
    
    .. math::
    
       z_{t-1}
       \;=\;
       \sqrt{\bar{\alpha}_{t-1}}\, z_0
       \;+\;
       \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\epsilon_\theta
       \;+\;
       \sigma_t\,\xi,\qquad \xi\sim\mathcal{N}(0,I),
    
    where the noise scale :math:`\sigma_t` is controlled by :math:`\eta`:
    
    .. math::
    
       \sigma_t
       \;=\;
       \eta
       \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}
       \sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}\, .
    
    **Special cases.**
    - :math:`\eta=0`: deterministic DDIM (no stochastic term).
    - :math:`\eta>0`: stochastic sampling with variance :math:`\sigma_t^2`.
    
    .. note::
       This sampler expects the model to predict the noise
       :math:`\epsilon_\theta(z_t, t)` at each step. Latent shapes in SD-style
       models are typically ``(B, 4, H/8, W/8)``.
    
    :param float beta_min: Minimum value of the linear :math:`\beta_t` schedule.
    :param float beta_max: Maximum value of the linear :math:`\beta_t` schedule.
    :param int num_train_timesteps: Training horizon :math:`T` used to build schedules.
    :param int num_inference_steps: Number of sampling steps.
    :param deepinv.models.LatentDiffusion | None model:
        Latent model exposing ``forward(x, t, prompt=...)`` and VAE ``encode``/``decode``;
        its ``forward`` must predict :math:`\epsilon_\theta`.
    :param str prompt: Optional text prompt passed through to the model.
    :param torch.dtype dtype: Computation dtype used inside the sampler.
    :param torch.device | None device: Target device. Defaults to CUDA if available, else CPU.
    
    :returns: The final clean latent :math:`z_0` with the same shape as the input latent.
    :rtype: torch.Tensor
    """


    def __init__(
        self,
        beta_min: float = 0.00085,
        beta_max: float = 0.012,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 200,
        model: LatentDiffusion | None = None,
        prompt: str = "",
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_d = self.beta_max - self.beta_min
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.model = model
        self.prompt = prompt
        self.dtype = dtype
        self.device = device

        self.alphas_cumprod = self.get_alpha_prod(
            beta_start=beta_min,
            beta_end=beta_max,
            num_train_timesteps=num_train_timesteps,
        )[4].to(self.device, self.dtype)

    def get_alpha_prod(
        self,
        beta_start: float = 0.1 / 1000,
        beta_end: float = 20 / 1000,
        num_train_timesteps: int = 1000,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build sequences derived from the cumulative schedule :math:`\bar{\alpha}_t`.
        
        Given a linear noise schedule :math:`\beta_t \in [\text{beta\_start}, \text{beta\_end}]`
        over ``num_train_timesteps`` steps, we form :math:`\bar{\alpha}_t=\prod_{i\le t}(1-\beta_i)`
        and return the following vectors (all of shape ``(num_train_timesteps,)``):
        
        - ``reduced_alpha_cumprod``:
          :math:`\sqrt{\dfrac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}`
        - ``sqrt_recip_alphas_cumprod``:
          :math:`\sqrt{\dfrac{1}{\bar{\alpha}_t}}`
        - ``sqrt_recipm1_alphas_cumprod``:
          :math:`\sqrt{\dfrac{1}{\bar{\alpha}_t}-1}`
        - ``sqrt_1m_alphas_cumprod``:
          :math:`\sqrt{1-\bar{\alpha}_t}`
        - ``sqrt_alphas_cumprod``:
          :math:`\sqrt{\bar{\alpha}_t}`
        
        :param float beta_start:
            Start of the linear :math:`\beta_t` schedule (inclusive).
        :param float beta_end:
            End of the linear :math:`\beta_t` schedule (inclusive).
        :param int num_train_timesteps:
            Number of timesteps :math:`T` used to build :math:`\bar{\alpha}_t`.
        
        :returns:
            Tuple ``(reduced_alpha_cumprod, sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod, sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)``.
        :rtype:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """

        betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, dtype=torch.float16
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(
            alphas.cpu(), axis=0
        )  # \bar{\alpha}_t (NumPy by design)

        torch_ab = torch.as_tensor(alphas_cumprod)
        sqrt_alphas_cumprod = torch.sqrt(torch_ab)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - torch_ab)
        reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / torch_ab)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / torch_ab - 1.0)

        return (
            reduced_alpha_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            sqrt_1m_alphas_cumprod,
            sqrt_alphas_cumprod,
        )

    def forward(
        self,
        sample: Tensor,
        eta: float = 0.0,
        noise: Tensor | None = None,
    ) -> Tensor:
        r"""
        Run **DDIM** sampling (generic :math:`\eta \ge 0`).
        
        At each step, the sampler updates the latent according to the DDIM rule with
        stochasticity controlled by :math:`\eta` (see class docstring for equations).
        
        :param torch.Tensor sample:
            Initial latent :math:`z_T` of shape ``(B, C, H, W)``.
        :param float eta:
            DDIM stochasticity parameter. ``0.0`` yields deterministic DDIM; values
            ``> 0`` inject Gaussian noise with scale :math:`\sigma_t`. Default: ``0.0``.
        :param torch.Tensor | None noise:
            Optional precomputed noise tensor (same shape as ``sample``). If provided
            and ``eta > 0``, it is used in place of freshly sampled noise (useful for
            reproducibility).
        
        :returns:
            The last predicted clean latent :math:`z_0` (same shape as ``sample``).
        :rtype: torch.Tensor
        """
        if self.model is None:
            raise RuntimeError("DDIMDiffusion.model is None.")

        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # Creates integer timesteps by multiplying by ratio.
        timesteps = (
            (np.arange(0, self.num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps += 1

        last_pred_z0: Tensor | None = None

        for timestep in tqdm(timesteps, desc="DDIM Sampling", total=len(timesteps)):
            # 1) previous step (= t-1)
            prev_timestep = (
                timestep - self.num_train_timesteps // self.num_inference_steps
            )

            # 2) schedule terms
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.alphas_cumprod[0]
            )
            beta_prod_t = 1.0 - alpha_prod_t  # == (1 - \bar{α}_t)

            # 3) predict ε_θ and z0(z_t)
            eps = self.model(
                x=sample.to(torch.float16),
                t=torch.tensor([timestep], device=self.device, dtype=torch.float16),
                prompt=self.prompt,
            ).to(self.dtype)

            pred_z0 = (sample - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
            last_pred_z0 = pred_z0

            # 4) compute σ_t for DDIM
            #    σ_t = η * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1}})
            #    (safe if prev_timestep < 0: ᾱ_{t-1} term falls back to ᾱ_0)
            sigma_t = (
                float(eta)
                * torch.sqrt((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t))
                * torch.sqrt(1.0 - (alpha_prod_t / alpha_prod_t_prev))
            )

            # 5) direction term (sqrt(1-ᾱ_{t-1}-σ_t^2) * ε̂)
            dir_coeff = torch.sqrt(
                torch.clamp(1.0 - alpha_prod_t_prev - sigma_t**2, min=0.0)
            )
            pred_dir = dir_coeff * eps

            # 6) DDIM update with optional noise term
            if sigma_t > 0.0:
                if noise is None:
                    noise_t = torch.randn_like(
                        sample, device=self.device, dtype=self.dtype
                    )
                else:
                    noise_t = noise.to(device=self.device, dtype=self.dtype)
                sample = (
                    alpha_prod_t_prev.sqrt() * pred_z0 + pred_dir + sigma_t * noise_t
                )
            else:
                sample = alpha_prod_t_prev.sqrt() * pred_z0 + pred_dir

            sample = sample.detach()

        assert last_pred_z0 is not None
        return last_pred_z0


class PSLDDiffusionPosterior(nn.Module):
    r"""
    DDIM sampler with **PSLD** latent correction (generic :math:`\eta \ge 0`).
    
    At each step :math:`t \rightarrow t-1`, with cumulative schedule
    :math:`\bar{\alpha}_t`, we
    
    1. predict the proxy clean latent :math:`\hat z_0 = z_0(z_t)`,
    2. take a **DDIM** update to :math:`z'_{t-1}` (with noise scale :math:`\sigma_t`),
    3. apply a **PSLD** correction by subtracting a gradient step w.r.t. the current
       latent :math:`z_t`.
    
    **DDIM update** (:footcite:t:`song2020denoising`)
    
    .. math::
    
       \hat z_0
       \;=\;
       \frac{z_t - \sqrt{1-\bar{\alpha}_t}\,\hat\epsilon}{\sqrt{\bar{\alpha}_t}},
       \qquad
       z_{t-1}
       \;=\;
       \sqrt{\bar{\alpha}_{t-1}}\,\hat z_0
       +
       \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\,\hat\epsilon
       +
       \sigma_t\,\xi,\quad \xi\sim\mathcal{N}(0,I),
    
    with
    
    .. math::
    
       \sigma_t
       \;=\;
       \eta
       \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}}
       \sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}\, .
    
    Thus :math:`\eta=0` yields deterministic DDIM, while :math:`\eta>0`
    injects stochasticity via :math:`\sigma_t`.
    
    **PSLD loss** (evaluated via VAE decode/encode; see :footcite:t:`Rout2023SolvingLI`)
    
    .. math::
    
       \mathcal{L}_\text{PSLD}(z_t)
       \;=\;
       \omega\,\|A(x(\hat z_0)) - y\|
       \;+\;
       \gamma\,\|\mathrm{Enc}(\Pi(x(\hat z_0))) - \hat z_0\|,
    
    where :math:`x(\hat z_0)=\mathrm{Dec}(\hat z_0)` and
    :math:`\Pi(x)=A^\*y + (I-A^\*A)\,x`. The correction step is
    
    .. math::
    
       z_{t-1} \;\leftarrow\; z'_{t-1} \;-\; \eta_t\,\nabla_{z_t}\mathcal{L}_\text{PSLD} \, .
    
    :param float beta_min: Minimum value of the linear :math:`\beta_t` schedule.
    :param float beta_max: Maximum value of the linear :math:`\beta_t` schedule.
    :param float alpha: Unused (kept for API compatibility).
    :param int num_train_timesteps: Training horizon :math:`T` used to build schedules.
    :param int num_inference_steps: Number of sampling steps.
    :param deepinv.models.LatentDiffusion | None model:
        Latent model exposing ``forward(x, t, prompt=...)`` (predicts
        :math:`\epsilon_\theta`), and VAE ``encode(x)`` / ``decode(z)``.
    :param torch.dtype dtype: Computation dtype.
    :param torch.device device: Target device (CUDA if available, else CPU).
    
    :notes:
        The per-step DDIM stochasticity is controlled by the ``eta`` argument passed
        to the forward method; ``eta=0`` recovers the deterministic case. The weights
        :math:`\omega` (data term) and :math:`\gamma` (gluing term) are configurable
        in the forward method.
    """


    def __init__(
        self,
        beta_min: float = 0.00085,
        beta_max: float = 0.012,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 200,
        model: LatentDiffusion | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_d = self.beta_max - self.beta_min
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.model = model
        self.dtype = dtype
        self.device = device

        self.alphas_cumprod = self.get_alpha_prod(
            beta_start=beta_min,
            beta_end=beta_max,
            num_train_timesteps=num_train_timesteps,
        )[4].to(self.device, self.dtype)

    def get_alpha_prod(
        self,
        beta_start: float = 0.1 / 1000,
        beta_end: float = 20 / 1000,
        num_train_timesteps: int = 1000,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Build sequences derived from the cumulative schedule :math:`\bar{\alpha}_t`.
        
        Given a linear noise schedule :math:`\beta_t \in [\text{beta\_start}, \text{beta\_end}]`
        over ``num_train_timesteps`` steps, we form :math:`\bar{\alpha}_t=\prod_{i\le t}(1-\beta_i)`
        and return the following vectors (all of shape ``(num_train_timesteps,)``):
        
        - ``reduced_alpha_cumprod``:
          :math:`\sqrt{\dfrac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}`
        - ``sqrt_recip_alphas_cumprod``:
          :math:`\sqrt{\dfrac{1}{\bar{\alpha}_t}}`
        - ``sqrt_recipm1_alphas_cumprod``:
          :math:`\sqrt{\dfrac{1}{\bar{\alpha}_t}-1}`
        - ``sqrt_1m_alphas_cumprod``:
          :math:`\sqrt{1-\bar{\alpha}_t}`
        - ``sqrt_alphas_cumprod``:
          :math:`\sqrt{\bar{\alpha}_t}`
        
        :param float beta_start:
            Start of the linear :math:`\beta_t` schedule (inclusive).
        :param float beta_end:
            End of the linear :math:`\beta_t` schedule (inclusive).
        :param int num_train_timesteps:
            Number of timesteps :math:`T` used to build :math:`\bar{\alpha}_t`.
        
        :returns:
            Tuple ``(reduced_alpha_cumprod, sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod, sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)``.
        :rtype:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, dtype=torch.float16
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # \bar{α}_t (NumPy by design)

        ab = torch.as_tensor(alphas_cumprod)
        sqrt_alphas_cumprod = torch.sqrt(ab)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - ab)
        reduced_alpha_cumprod = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / ab)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / ab - 1.0)

        return (
            reduced_alpha_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            sqrt_1m_alphas_cumprod,
            sqrt_alphas_cumprod,
        )

    def forward(
        self,
        sample: Tensor,
        y: Tensor,
        physics: Physics,
        dps_eta: float = 1.0,
        gamma: float = 1e-1,
        omega: float = 1.0,
        DDIM_eta: float = 0.0,
        noise: Tensor | None = None,
    ) -> Tensor:
        r"""
        Run **DDIM** with **PSLD** correction (generic :math:`\eta \ge 0`).
        
        For each step :math:`t \to t-1`, compute :math:`\hat z_0(z_t)`, take the DDIM
        update (noise scale :math:`\sigma_t` controlled by ``DDIM_eta``), then subtract
        a PSLD gradient step (size ``dps_eta``) computed w.r.t. the current noisy latent
        :math:`z_t`.
        
        :param torch.Tensor sample: Current noisy latent :math:`z_t`, shape ``(B, C, H, W)``.
        :param torch.Tensor y: Measurement in the range/shape expected by ``physics.A``.
        :param Physics physics: Linear measurement operator exposing ``A`` and ``A_adjoint``.
        :param float dps_eta: PSLD gradient step size :math:`\eta_t` (kept constant here).
        :param float gamma: Weight of the latent “gluing” term.
        :param float omega: Weight of the data-consistency term.
        :param float DDIM_eta:
            **DDIM** stochasticity parameter. ``0.0`` → deterministic DDIM; values ``> 0``
            inject Gaussian noise with per-step scale :math:`\sigma_t`.
        :param torch.Tensor | None noise:
            Optional precomputed noise (same shape as ``sample``) used when ``DDIM_eta > 0``
            for reproducibility.
        
        :returns: Last predicted clean latent :math:`\hat z_0`.
        :rtype: torch.Tensor
        """

        if self.model is None:
            raise RuntimeError("PSLDDiffusionPosterior.model is None.")

        step_ratio = self.num_train_timesteps // self.num_inference_steps
        if step_ratio <= 0:
            raise ValueError(
                "num_train_timesteps must be >= num_inference_steps and both > 0."
            )

        timesteps = (
            (np.arange(0, self.num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps += 1

        loop = tqdm(timesteps, desc="PSLD", total=len(timesteps))
        last_pred_x0: Tensor | None = None

        for timestep in loop:
            prev_timestep = (
                timestep - self.num_train_timesteps // self.num_inference_steps
            )

            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.alphas_cumprod[0]
            )
            beta_prod_t = 1.0 - alpha_prod_t

            # --- predict ε and z0 from current z_t
            sample = sample.detach().requires_grad_(True)
            eps = self.model(
                x=sample.to(torch.float16),
                t=torch.tensor([timestep], device=self.device, dtype=torch.float16),
                prompt="",  # unconditional for PSLD by default
            ).to(self.dtype)

            z0 = (sample - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
            last_pred_x0 = z0

            # --- DDIM step to z'_{t-1} (η may be > 0)
            # σ_t = η * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1}})
            sigma_t = (
                float(DDIM_eta)
                * torch.sqrt((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t))
                * torch.sqrt(1.0 - (alpha_prod_t / alpha_prod_t_prev))
            )

            dir_coeff = torch.sqrt(
                torch.clamp(1.0 - alpha_prod_t_prev - sigma_t**2, min=0.0)
            )
            pred_dir = dir_coeff * eps

            if sigma_t > 0.0:
                noise_t = (
                    torch.randn_like(sample, device=self.device, dtype=self.dtype)
                    if noise is None
                    else noise.to(device=self.device, dtype=self.dtype)
                )
                zt_prime = alpha_prod_t_prev.sqrt() * z0 + pred_dir + sigma_t * noise_t
            else:
                zt_prime = alpha_prod_t_prev.sqrt() * z0 + pred_dir  # η=0 branch

            # --- PSLD loss built from z0(z_t); gradient w.r.t. z_t
            x = self.model.decode(z0.half())
            meas_pred = physics.A(x.float())
            residual = meas_pred - y
            data_loss = torch.norm(residual)  # no σ_y scaling (by design)

            # Projection-based "gluing"
            ortho = x.float() - physics.A_adjoint(meas_pred)  # (I - A^*A) x
            para = physics.A_adjoint(y)  # A^* y
            projected = (para + ortho).clamp_(-1, 1)

            recon_z = self.model.encode(projected.half())
            glue_loss = torch.linalg.norm(recon_z.float() - z0.float())

            total = omega * data_loss + gamma * glue_loss

            grad = torch.autograd.grad(
                total, sample, create_graph=False, retain_graph=False
            )[0]

            # --- PSLD correction on z'_{t-1}
            sample = (zt_prime.detach() - float(dps_eta) * grad.detach()).to(self.dtype)

            loop.set_postfix(
                loss=float(data_loss.detach().cpu()),
                gluing=float(glue_loss.detach().cpu()),
            )

        assert last_pred_x0 is not None
        return last_pred_x0.detach()
