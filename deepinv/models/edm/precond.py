import torch
import numpy as np

# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


class VPPrecond(torch.nn.Module):
    r"""
    Pre-conditioning of a denoiser, as proposed in the paper:
    Score-Based Generative Modeling through Stochastic Differential Equations (https://arxiv.org/abs/2011.13456).

    Given a neural network :math:`F_{\theta}` of weights :math:`\theta`. The denoiser :math:`D_{\theta}` is defined for
    any noisy image :math:`x` and noise level :math:`\sigma` as follows:

    ...math::
        D_{\theta}(x, \sigma) = c_{\mathrm{skip}}(\sigma) x + c_{\mathrm{out}}(\sigma) F_{\theta}(c_{\mathrm{in}} x, c_{\mathrm{noise}}(\sigma)).

    The pre-conditioning parameters are defined as follows:

    ..math::
        \begin{align}
        c_{\mathrm{skip}}(\sigma) &= 1   \\
        c_{\mathrm{out}}(\sigma) &= -\sigma                \\
        c_{\mathrm{in}}(\sigma) &= \frac{1}{\sqrt{\sigma^2 + 1}}                     \\
        c_{\mathrm{noise}}(\sigma) &= (M - 1)\sigma^{-1}(\sigma)
        \end{align}

    where, :math:`M` is the number of steps (1000) in the original paper, :math:`\sigma^{-1}` is the inverse of the noise schedule function:

    ..math::
        \sigma^{-1}(\sigma) = \frac{\sqrt{\log \left(\beta_{\mathrm{min}}^2 + 2 \beta_{\mathrm{d}} (1 + \sigma^2) \right) - \beta_{\mathrm{min}}}{\beta_{\mathrm{d}}}

        (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d
    """

    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = model

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = handle_sigma(sigma, torch.float32, x.device, x.size(0))

        if class_labels is not None:
            class_labels = class_labels.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


class VEPrecond(torch.nn.Module):
    r"""
    Pre-conditioning of a denoiser, as proposed in the paper:
    Score-Based Generative Modeling through Stochastic Differential Equations (https://arxiv.org/abs/2011.13456).

    Given a neural network :math:`F_{\theta}` of weights :math:`\theta`. The denoiser :math:`D_{\theta}` is defined for
    any noisy image :math:`x` and noise level :math:`\sigma` as follows:

    ..math::
        D_{\theta}(x, \sigma) = x + \sigma F_{\theta}(x, \log (\sigma / 2)).

    """

    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.model = model

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = handle_sigma(sigma, torch.float32, x.device, x.size(0))
        if class_labels is not None:
            class_labels = class_labels.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            x.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = x + sigma * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".


class iDDPMPrecond(torch.nn.Module):
    r"""
    Pre-conditioning of a denoiser, as proposed in the DDIM paper:
    Denoising diffusion implicit models: https://arxiv.org/abs/2010.02502

    Given a neural network :math:`F_{\theta}` of weights :math:`\theta`. The denoiser :math:`D_{\theta}` is defined for
    any noisy image :math:`x` and noise level :math:`\sigma` as follows:

    ..math::
        D_{\theta}(x, \sigma) = c_{\mathrm{skip}}(\sigma) x + c_{\mathrm{out}}(\sigma) F_{\theta}(c_{\mathrm{in}} x, c_{\mathrm{noise}}(\sigma)).

    The pre-conditioning parameters are defined as follows:

    ..math::
        \begin{align}
        c_{\mathrm{skip}}(\sigma) &= 1   \\
        c_{\mathrm{out}}(\sigma) &= -\sigma \\
        c_{\mathrm{in}}(\sigma) &= \frac{1}{\sqrt{\sigma^2 + 1}}                     \\
        c_{\mathrm{noise}}(\sigma) &= M - 1 - \mathrm{argmin}_j |u_j - \sigma | 
        \end{align}

    where :math:`u_{j - 1} = \sqrt{\frac{u_j^2 + 1}{\max (\bar \alpha_{j-1}  / \bar \alpha_{j}, C_1)} - 1}` and :math:`u_M = 0`.
    """

    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        C_1=0.001,  # Timestep adjustment at low noise levels.
        C_2=0.008,  # Timestep adjustment at high noise levels.
        M=1000,  # Original number of timesteps in the DDPM formulation.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = model

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1)
                / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
                - 1
            ).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = handle_sigma(sigma, torch.float32, x.device, x.size(0))

        if class_labels is not None:
            class_labels = class_labels.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        )

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
            self.u.reshape(1, -1, 1),
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPrecond(torch.nn.Module):
    r"""
    Pre-conditioning of a denoiser, as proposed in the paper:
    Elucidating the Design Space of Diffusion-Based Generative Models: https://arxiv.org/pdf/2206.00364

    Given a neural network :math:`F_{\theta}` of weights :math:`\theta`. The denoiser :math:`D_{\theta}` is defined for
    any noisy image :math:`x` and noise level :math:`\sigma` as follows:

    ..math::
        D_{\theta}(x, \sigma) = c_{\mathrm{skip}}(\sigma) x + c_{\mathrm{out}}(\sigma) F_{\theta}(c_{\mathrm{in}} x, c_{\mathrm{noise}}(\sigma)).

    The pre-conditioning parameters are defined as follows:

    ..math::
        \begin{align}
        c_{\mathrm{skip}}(\sigma) &= \frac{\sigma_{\mathrm{data}}^2}{\sigma^2 + \sigma_{\mathrm{data}}^2}   \\
        c_{\mathrm{out}}(\sigma) &= \sigma \frac{\sigma_{\mathrm{data}}{\sqrt{\sigma^2 + \sigma_{\mathrm{data}}^2}}               \\
        c_{\mathrm{in}}(\sigma) &= \frac{1}{\sqrt{\sigma^2 + \sigma_{\mathrm{data}}^2}}                     \\
        c_{\mathrm{noise}}(\sigma) &= \log(\sigma) / 4
        \end{align}

    """

    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = handle_sigma(sigma, torch.float32, x.device, x.size(0))
        if class_labels is not None:
            class_labels = class_labels.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x


# ----------------------------------------------------------------------------
# Some additional utilities
def handle_sigma(sigma, dtype, device, batch_size):
    if isinstance(sigma, torch.Tensor):
        if sigma.ndim == 0:
            sigma = sigma[None]
        assert (
            sigma.size(0) == batch_size or sigma.size(1) == 1
        ), "sigma must be a Tensor with batch_size equal to 1 or the batch_size of input images"
        return (
            sigma.to(device, dtype).reshape(-1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        )

    elif isinstance(sigma, (float, int)):
        return (
            torch.tensor([sigma])
            .to(device, dtype)
            .reshape(-1, 1, 1, 1)
            .expand(batch_size, -1, -1, -1)
        )
    else:
        raise ValueError("Unsupported sigma type.")
