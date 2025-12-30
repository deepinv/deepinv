from __future__ import annotations
import torch
from .base import Denoiser


class DiffusersDenoiserWrapper(Denoiser):
    """
    Wraps a `HuggingFace diffusers <https://huggingface.co/docs/diffusers/index>`_ model as a DeepInv Denoiser.

    :param str mode_id: Diffusers model id or HuggingFace hub repository id. For example, 'google/ddpm-cat-256'.
        The id must work with `DiffusionPipeline`.
        See `Diffusers Documentation <https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/overview#diffusers.DiffusionPipeline>`_.
    :param bool clip_output: Whether to clip the output to the model range. Default is `True`.
    :param device: Device to load the model on. Default is 'cpu'.

    .. note::
        Currently, only models trained with `DDPMScheduler` are supported.

    .. warning::
        This wrapper requires the `diffusers` and `transformers` packages.
        You can install them via `pip install diffusers transformers`.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> from deepinv.models import DiffusersDenoiserWrapper
        >>> import torch
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> denoiser = DiffusersDenoiserWrapper(mode_id='google/ddpm-cat-256', device=device)
        >>> x = dinv.utils.load_example(
        ...         "cat.jpg",
        ...         img_size=256,
        ...         resize_mode="resize",
        ...     ).to(device)


        >>> sigma = 0.1
        >>> x_noisy = x + sigma * torch.randn_like(x)
        >>> with torch.no_grad():
        ...     x_denoised = denoiser(x_noisy, sigma=sigma)

    """

    def __init__(
        self,
        mode_id: str = None,
        clip_output: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        assert (
            mode_id is not None
        ), "Provide a diffusers model id. E.g., 'google/ddpm-cat-256'"

        try:
            from diffusers import DiffusionPipeline, DDPMScheduler
        except ImportError:
            raise ImportError(
                "diffusers is not installed. Please install it via 'pip install diffusers'."
            )

        pipeline = DiffusionPipeline.from_pretrained(mode_id, torch_dtype=torch.float32)
        self.model = pipeline.unet
        self.scheduler = pipeline.scheduler

        assert isinstance(
            self.scheduler, DDPMScheduler
        ), "Currently, only DDPMScheduler is supported."

        self.clip_output = clip_output

        # Precompute sigma(t) over training timeline
        ac = self.scheduler.alphas_cumprod

        self.register_buffer(
            "scale_schedule", ac.sqrt()
        )  # the scaling factor (EDM) -- sqrt_alphas_cumprod
        self.register_buffer(
            "sigma_schedule", (1 / ac - 1.0).clamp(min=0).sqrt()
        )  # the noise level (EDM) -- sqrt_one_minus_alphas_cumprod / sqrt_alphas_cumprod

        # prediction_type: "epsilon", "v_prediction", or "sample"
        self.prediction_type = getattr(
            self.scheduler.config, "prediction_type", "epsilon"
        )

        self.to(device)

    def _nearest_t_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Maps a sigma to the nearest training time-step index.
        Supports scalar or per-batch tensor sigma.
        """
        sigmas = self.sigma_schedule  # [T]
        # If batch, do per-element argmin.
        if sigma.dim() == 0:
            timestep = torch.argmin((sigmas - sigma).abs())
        else:
            # sigma shape [B]; produce [B] of indices
            diffs = (sigmas[None, :] - sigma[:, None]).abs()  # [B, T]
            timestep = torch.argmin(diffs, dim=1)
        return timestep

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Applies denoiser :math:`\denoiser{x}{\sigma}`.
        The input `x` is expected to be in `[0, 1]` range (up to random noise) and the output is also in `[0, 1]` range.

        :param torch.Tensor x: noisy input, of shape `[B, C, H, W]`.
        :param torch.Tensor, float sigma: noise level. Can be a `float` or a :class:`torch.Tensor` of shape `[B]`.
            If a single `float` is provided, the same noise level is used for all samples in the batch.
            Otherwise, batch-wise noise levels are used.
        :param args: additional positional arguments to be passed to the model.
        :param kwarg: additional keyword arguments to be passed to the model. For example, a `prompt` for text-conditioned or `class_label` for class-conditioned models.

        :returns: (:class:`torch.Tensor`) the denoised output.
        """

        device = x.device
        dtype = x.dtype

        assert sigma is not None, "Please provide a noise level sigma."

        # Handle sigma
        sigma = self._handle_sigma(
            sigma,
            batch_size=x.shape[0],
            ndim=x.ndim,
            device=device,
            dtype=dtype,
        )

        sigma = sigma * 2  # since image is in [-1, 1] range in the model
        timestep = self._nearest_t_from_sigma(sigma.squeeze())

        scale = self.scale_schedule[timestep].view(-1, *(1,) * (x.ndim - 1))

        # Rescale input x from [0, 1] to model scale [-1, 1] and apply scaling following DDPM
        x = (x * 2 - 1) * scale
        # UNet forward
        pred = self.model(x, timestep, *args, return_dict=False, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # take the first output if multiple outputs are returned
        pred = pred.to(dtype)

        # Convert model output to x0 depending on prediction type
        pt = self.prediction_type
        if pt == "epsilon":
            x0 = x / scale - sigma * pred
        elif pt == "v_prediction":
            x0 = scale * (x - sigma * pred)
        elif pt == "sample":
            x0 = pred
        else:
            raise ValueError(f"Unsupported prediction_type: {pt}")

        # Optional: clamp to model range [-1, 1]
        if self.clip_output:
            x0 = x0.clamp(-1, 1)

        # Rescale to [0, 1]
        x0 = (x0 + 1) / 2
        return x0


class ComplexDenoiserWrapper(Denoiser):
    r"""
    Complex-valued wrapper for a real-valued denoiser :math:`\denoisername(\cdot, \sigma)`.

    This class lifts any real-valued denoiser to the complex domain by applying it separately to a chosen *pair* of real representations of the complex input and recombining the outputs.

    Let the input be :math:`x \in \mathbb{C}^{B\times C\times H\times W}` and a noise level :math:`\sigma > 0` (scalar or batch of size :math:`B`).
    The underlying denoiser (given by `denoiser`) :math:`\denoisername` acts on real tensors only. Two processing modes are supported:

    |sep|

    1. `'real_imag'` mode

    We decompose

    .. math::

        x = x_{\mathrm{real}} + i x_{\mathrm{imag}}.

    The denoiser is applied on the real and imaginary parts (same :math:`\sigma` broadcast across both halves).
    The complex reconstruction is

    .. math::

        \hat x = \denoisername(x_{\mathrm{real}}, \sigma) + i \, \denoisername(x_{\mathrm{imag}}, \sigma).

    If the provided input tensor is real (i.e. `torch.is_complex(x)` is ``False``), it is interpreted as :math:`x_{\mathrm{real}}` with :math:`x_{\mathrm{imag}}=0` and the output is returned as
    :math:`\denoisername(x_{\mathrm{real}},\sigma) + i 0` (complex dtype ensured).

    |sep|

    2. `'abs_angle'` mode

    We use the polar decomposition

    .. math::

        x = m \exp(i\phi), \qquad m = |x|,\; \phi = \mathrm{arg}(x) \in (-\pi,\pi].

    The denoiser is applied on the magnitude and phase parts (same :math:`\sigma` broadcast across both halves).
    The reconstructed complex output is

    .. math::

        \hat x = \denoisername(m, \sigma) \exp \big(i\, \denoisername(\phi, \sigma)\big).

    Note that the phase estimate :math:`\denoisername(\phi,\sigma)` is **clipped** back to :math:`(-\pi,\pi]`.

    .. note::

        This wrapper can only process complex inputs that are compatible with the underlying real-valued denoiser.
        For example, if the wrapped ``denoiser`` supports only single-channel (grayscale) real images, then the
        corresponding complex input must also be single-channel.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> import torch
        >>> from deepinv.models import ComplexDenoiserWrapper, DRUNet
        >>> denoiser = DRUNet() # doctest: +IGNORE_OUTPUT
        >>> complex_denoiser = ComplexDenoiserWrapper(denoiser, mode="real_imag")
        >>> y = torch.randn(2, 3, 32, 32, dtype=torch.complex64)  # complex input
        >>> sigma = 0.1
        >>> with torch.no_grad():
        ...     denoised = complex_denoiser(y, sigma)
        >>> print(denoised.dtype)  # should be complex dtype
        torch.complex64

    :param deepinv.models.Denoiser denoiser: Real-valued denoiser :math:`\denoisername` to wrap.
    :param str mode: Either ``'real_imag'`` or ``'abs_angle'``. Default ``'real_imag'``.
    :raises ValueError: If an unsupported mode string is provided.
    :returns: Complex denoised output :math:`\hat x` with same spatial shape as the input.
    """

    def __init__(self, denoiser: Denoiser, mode: str = "real_imag", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.denoiser = denoiser

        if mode.lower() not in ["real_imag", "abs_angle"]:
            raise ValueError(
                f"'mode' must be 'real_imag' or 'abs_angle'. Got {mode} instead."
            )

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        r"""
        Applies the complex-valued denoiser. If a real tensor is provided, it is treated as a complex tensor with zero imaginary part.

        :param torch.Tensor x: complex-valued input images.
        :param float or torch.Tensor sigma: noise level.

        :return: Denoised images, with the same shape as the input and will always be in complex dtype.
        """
        # Duplicate sigma in the batch dimension for real and imaginary parts
        sigma = self._handle_sigma(
            sigma,
            batch_size=x.size(0) * 2,
            ndim=x.ndim,
            device=x.device,
            dtype=x.real.dtype,
        )

        if self.mode == "real_imag":
            x_real = x.real

            if torch.is_complex(x):
                noisy_batch = torch.cat((x_real, x.imag), 0)
                denoised_batch = self.denoiser(noisy_batch, sigma)
                return (
                    denoised_batch[: x_real.shape[0], ...]
                    + 1j * denoised_batch[x_real.shape[0] :, ...]
                )
            else:
                return self.denoiser(x_real, sigma) + 0j

        else:  # abs_angle
            x_mag = torch.abs(x)
            x_phase = torch.angle(x)
            noisy_batch = torch.cat((x_mag, x_phase), 0)
            denoised_batch = self.denoiser(noisy_batch, sigma)
            return denoised_batch[: x_mag.shape[0], ...] * torch.exp(
                1j * denoised_batch[x_mag.shape[0] :, ...].clamp(-torch.pi, torch.pi)
            )
