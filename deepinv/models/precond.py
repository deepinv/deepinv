import torch
from deepinv.models import Denoiser

# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPrecond(Denoiser):
    r"""
    Pre-conditioning of a denoiser, as proposed in the paper:
    `Elucidating the Design Space of Diffusion-Based Generative Models <https://arxiv.org/pdf/2206.00364>`_.

    Given a neural network :math:`\tilde{\mathrm{F}}`, the denoiser :math:`\denoiser{x}{\sigma}` is defined for
    any noisy image :math:`x` and noise level :math:`\sigma` as follows:

    ..math::
        \denoiser{x}{\sigma} = c_{\mathrm{skip}}(\sigma) x + c_{\mathrm{out}}(\sigma) \tilde{\mathrm{F}}_{c_{\mathrm{noise}}\sigma}(c_{\mathrm{in}} x).

    The pre-conditioning parameters are defined as follows:

    ..math::
        \begin{align}
        c_{\mathrm{skip}}(\sigma) &= \frac{\sigma_{\mathrm{pixel}}^2}{\sigma^2 + \sigma_{\mathrm{pixel}}^2}   \\
        c_{\mathrm{out}}(\sigma) &= \sigma \frac{\sigma_{\mathrm{pixel}}{\sqrt{\sigma^2 + \sigma_{\mathrm{pixel}}^2}}               \\
        c_{\mathrm{in}}(\sigma) &= \frac{1}{\sqrt{\sigma^2 + \sigma_{\mathrm{pixel}}^2}}                     \\
        c_{\mathrm{noise}}(\sigma) &= \log(\sigma) / 4
        \end{align}

    """

    def __init__(
        self,
        model,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        pixel_std=0.75,  # Expected standard deviation of the training data in pixel.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.pixel_std = pixel_std
        self.model = model

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = self._handle_sigma(sigma, torch.float32, x.device, x.size(0))
        if class_labels is not None:
            class_labels = class_labels.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.pixel_std**2 / (sigma**2 + self.pixel_std**2)
        c_out = sigma * self.pixel_std / (sigma**2 + self.pixel_std**2).sqrt()
        c_in = 1 / (self.pixel_std**2 + sigma**2).sqrt()
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

    @staticmethod
    def _handle_sigma(sigma, dtype, device, batch_size):
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim == 0:
                return sigma[None].to(device, dtype).view(-1, 1, 1, 1)
            elif sigma.ndim == 1:
                assert (
                    sigma.size(0) == batch_size or sigma.size(0) == 1
                ), "sigma must be a Tensor with batch_size equal to 1 or the batch_size of input images"
                return sigma.to(device, dtype).view(-1, 1, 1, 1)

            else:
                raise ValueError(f"Unsupported sigma shape {sigma.shape}.")

        elif isinstance(sigma, (float, int)):
            return torch.tensor([sigma]).to(device, dtype).reshape(-1, 1, 1, 1)
        else:
            raise ValueError("Unsupported sigma type.")
