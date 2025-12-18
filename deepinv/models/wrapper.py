from __future__ import annotations
import torch
from torch import nn
from deepinv.models import Denoiser
from typing import Callable
import numpy as np


class ScoreModelWrapper(Denoiser):
    r"""
    Wraps a score model as a DeepInv Denoiser.

    Given a noisy sample :math:`x_t = s_t(x_0 + \sigma_t \varepsilon)`, where :math:`\varepsilon \sim \mathcal{N}(0, I)`,
    depending on the `prediction_type`, the input `score_model` is trained to predict, either:
        - the noise :math:`\varepsilon` (`prediction_type = noise`)
        - the denoised sample :math:`x_0` (`prediction_type = denoised`)
        - the `v-prediction` :math:`s_t (\varepsilon - sigma_t * x_0)` as proposed by :footcite:`salimans2022progressive` (`prediction_type = v_prediction`)
        - the velocity (or drift) of the corresponding ODE/SDE :math:`s_t (\varepsilon - sigma_t * x_0)` as typically the case for flow-matching models (`prediction_type = velocity`)

    :param nn.Module | Callable score_model: score model to be wrapped.
    :param str prediction_type: type of prediction made by the score model.
    :param bool clip_output: whether to clip the output to the model range. Default is `True`.
    :param Callable | torch.Tensor sigma_schedule: continuous function or tensor (of shape `[N]` with `N` the number of time steps) defining the noise schedule :math:`\sigma_t`.
    :param Callable | torch.Tensor scale_schedule: function or tensor (of shape `[N]` with `N` the number of time steps) defining the scaling schedule :math:`s_t`.
    :param Callable sigma_inverse: analytic inverse of the `sigma_schedule`. If not provided, a numeric inversion is used.
    :param bool variance_preserving: whether the schedule is variance-preserving. If `True`, the `scale_schedule` is computed from the `sigma_schedule`.
    :param bool variance_exploding: whether the schedule is variance-exploding. If `True`, the `scale_schedule` is set to `1`.
    :param float T: maximum time value for continuous schedules. Default is `1.0`.
    :param str: device to load the model on. Default is `'cpu'`.
    """

    def __init__(
        self,
        score_model: nn.Module | Callable = None,
        prediction_type: str = "epsilon",  # prediction_type: "epsilon", "v_prediction", or "sample"
        clip_output: bool = True,
        sigma_schedule: Callable | torch.Tensor = None,
        scale_schedule: Callable | torch.Tensor = None,
        sigma_inverse: Callable = None,
        variance_preserving: bool = False,
        variance_exploding: bool = False,
        T: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.model = score_model
        self.clip_output = clip_output
        self.prediction_type = prediction_type

        if scale_schedule is None:
            if variance_preserving:
                if isinstance(sigma_schedule, Callable):

                    def scale_schedule(t):
                        t = self._handle_time_step(t)
                        return (1 / (1 + self.sigma_t(t) ** 2)) ** 0.5

                else:
                    scale_schedule = 1.0 / (1 + sigma_schedule**2) ** 0.5
            elif variance_exploding:
                if isinstance(sigma_schedule, Callable):
                    scale_schedule = lambda t: 1.0
                else:
                    scale_schedule = torch.ones_like(sigma_schedule)

        if isinstance(sigma_schedule, torch.Tensor):
            self.register_buffer("sigma_schedule", sigma_schedule)
        else:
            self.sigma_schedule = sigma_schedule
        if isinstance(scale_schedule, torch.Tensor):
            self.register_buffer("scale_schedule", scale_schedule)
        else:
            self.scale_schedule = scale_schedule

        self.sigma_inverse = sigma_inverse
        self.T = T
        self.to(device)

    def get_schedule_value(
        self,
        schedule: Callable | torch.Tensor,
        t: torch.Tensor,
        target_size: torch.Size = None,
    ) -> torch.Tensor:
        r"""
        Get the value of a schedule (function or tensor) at given time steps.
        :param Callable | torch.Tensor schedule: schedule function or tensor.
        :param torch.Tensor t: time steps, of shape `[B]` or `[]`.
        :param torch.Size target_size: target size to broadcast the output to. Default is `None`.
        :returns: (:class:`torch.Tensor`) schedule values at time steps `t`, of shape that is broadcastable to `target_size` if `target_size` is provided.
        """

        if isinstance(schedule, torch.Tensor):
            val = schedule[t.long()]
        else:
            val = schedule(t)

        if target_size is not None:
            val = val.view(-1, *[1] * (len(target_size) - 1))
        return val

    def _pred_to_score(self, pred, x, sigma, scale):
        pt = self.prediction_type
        if pt == "epsilon":  # predicts white noise
            score = -self.stable_division(pred, sigma)
        elif (
            pt == "v_prediction"
        ):  # predicts s_t*(eps - sigma_t * x). See https://arxiv.org/pdf/2202.00512.
            score = -self.stable_division(pred / scale + sigma * x, sigma)
        elif pt == "sample":  # predicts the denoised image (Tweedie formula)
            score = self.stable_division(x + (scale * sigma) ** 2 * pred, scale)
        else:
            raise ValueError(f"Unsupported prediction_type: {pt}")
        return score

    def _pred_to_x0(self, pred, x, sigma, scale):
        pt = self.prediction_type
        if pt == "epsilon":  # predics white noise
            x0 = x / scale - sigma * pred
        elif (
            pt == "v_prediction"
        ):  # predics s_t*eps - sigma_t * x. See https://arxiv.org/pdf/2202.00512.
            x0 = scale * (x - sigma * pred)
        elif pt == "sample":  # predics the denoised image
            x0 = pred
        else:
            raise ValueError(f"Unsupported prediction_type: {pt}")
        return x0

    def time_from_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:
        r"""
        Computes the time step `t` corresponding to a given noise level `sigma`.

        If an analytic inverse of the `sigma_schedule` is provided, it is used.
        Otherwise, a numeric inversion is performed (nearest neighbor for discrete schedules, binary search for continuous schedules).

        :param torch.Tensor | float sigma: noise level(s), either a scalar or a tensor of shape `[B]`.

        """

        sigma = torch.as_tensor(sigma)

        # 1) If user provided an analytic / predefined inverse, use it.
        if self.sigma_inverse is not None:
            return self.sigma_inverse(sigma)

        # 2) If we have a discrete table, use nearest index.
        if isinstance(self.sigma_schedule, torch.Tensor):
            sigmas = self.sigma_schedule  # [T]
            sigma = sigma.to(device=sigmas.device, dtype=sigmas.dtype)
            if sigma.dim() == 0:
                return torch.argmin((sigmas - sigma).abs())
            else:
                diffs = (sigmas[None, :] - sigma[:, None]).abs()  # [B, T]
                return torch.argmin(diffs, dim=1)
        else:
            # 3) Fallback: numeric inversion for continuous schedules (binary search).
            t_low = torch.zeros_like(sigma)
            t_high = torch.full_like(sigma, self.T)
            for _ in range(32):
                t_mid = (t_low + t_high) / 2
                sigma_mid = self.sigma_schedule(t_mid)
                go_right = sigma_mid < sigma
                t_low = torch.where(go_right, t_mid, t_low)
                t_high = torch.where(go_right, t_high, t_mid)
            return (t_low + t_high) / 2

    @staticmethod
    def stable_division(a, b, epsilon: float = 1e-7):
        if isinstance(b, torch.Tensor):
            b = torch.where(
                b.abs().detach() > epsilon,
                b,
                torch.full_like(b, fill_value=epsilon) * b.sign(),
            )
        elif isinstance(b, (float, int)):
            b = max(epsilon, abs(b)) * np.sign(b)
        return a / b

    def score(
        self, x: torch.Tensor, t: float | torch.Tensor = None, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the score function :math:`\nabla_x \log p_t(x)`.

        :param torch.Tensor x: input tensor of shape `[B, C, H, W]`.
        :param torch.Tensor | float t: single time or tensor of shape `[B]` or `[]`.
        :param args: additional positional arguments of the model.
        :param kwargs: additional keyword arguments of the model.

        :returns: (:class:`torch.Tensor`) the score function of shape `[B, C, H, W]`.
        """
        device = x.device
        dtype = x.dtype

        assert t is not None, "Please provide a time step t."

        # Handle time step
        t = self._handle_sigma(
            t, batch_size=x.size(), ndim=1, device=device, dtype=dtype
        )

        # UNet forward
        pred = self.model(x, t, *args, return_dict=False, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = pred.to(dtype)

        sigma = self.get_schedule_value(self.sigma_schedule, t, x.shape)
        scale = self.get_schedule_value(self.scale_schedule, t, x.shape)

        return self._pred_to_score(pred, x, sigma, scale)

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
        timestep = self.time_from_sigma(sigma.squeeze())
        scale = self.get_schedule_value(self.scale_schedule, timestep, x.shape)

        # Rescale input x from [0, 1] to model scale [-1, 1] and apply scaling following DDPM
        x = (x * 2 - 1) * scale
        # UNet forward
        pred = self.model(x, timestep, *args, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # take the first output if multiple outputs are returned
        pred = pred.to(dtype)

        # Convert model output to x0 depending on prediction type
        x0 = self._pred_to_x0(pred, x, sigma, scale)

        # Optional: clamp to model range [-1, 1]
        if self.clip_output:
            x0 = x0.clamp(-1, 1)

        # Rescale to [0, 1]
        x0 = (x0 + 1) / 2
        return x0


class DiffusersDenoiserWrapper(ScoreModelWrapper):
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

        model = pipeline.unet
        scheduler = pipeline.scheduler
        prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")

        assert isinstance(
            scheduler, DDPMScheduler
        ), "Currently, only DDPMScheduler is supported."
        ac = scheduler.alphas_cumprod
        scale_schedule = ac.sqrt()
        sigma_schedule = (1 / ac - 1.0).clamp(min=0).sqrt()

        super().__init__(
            score_model=model,
            prediction_type=prediction_type,
            clip_output=clip_output,
            scale_schedule=scale_schedule,
            sigma_schedule=sigma_schedule,
            device=device,
        )

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

        return super().forward(x, sigma, *args, return_dict=False, **kwargs)


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
