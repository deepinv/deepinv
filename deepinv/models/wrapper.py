from __future__ import annotations
import torch
from torch import Tensor, nn
from deepinv.models import Denoiser
from typing import Callable
import numpy as np


class ScoreModelWrapper(Denoiser):
    r"""
    Wraps a score model as a DeepInv Denoiser.

    Given a noisy sample :math:`x_t = s_t(x_0 + \sigma_t \varepsilon)`, where :math:`\varepsilon \sim \mathcal{N}(0, I)`,
    depending on the `prediction_type`, the input `score_model` is trained to predict, either:

        * the noise :math:`\varepsilon` (`prediction_type = 'epsilon'`) as typically the case for DDPM models, or
        * the denoised sample :math:`x_0` (`prediction_type = 'sample'`) or
        * the `v-prediction` :math:`s_t (\varepsilon - \sigma_t \cdot x_0)` as proposed by :footcite:`salimans2022progressive` (`prediction_type = 'v_prediction'`)

    :param torch.nn.Module | Callable score_model: score model to be wrapped.
    :param str prediction_type: type of prediction made by the score model.
    :param bool clip_output: whether to clip the output to the model range. Default is `True`.
    :param Callable | torch.Tensor sigma_t: continuous function or tensor (of shape `[N]` with `N` the number of time steps) defining the noise schedule :math:`\sigma_t`.
    :param Callable | torch.Tensor scale_t: function or tensor (of shape `[N]` with `N` the number of time steps) defining the scaling schedule :math:`s_t`.
    :param Callable sigma_inverse: analytic inverse of the `sigma_t`. If not provided, a numeric inversion is used.
    :param bool variance_preserving: whether the schedule is variance-preserving. If `True`, `scale_t` is computed from the `sigma_t`.
    :param bool variance_exploding: whether the schedule is variance-exploding. If `True`, `scale_t` is set to `1`.
    :param float T: maximum time value for continuous schedules. Default is `1.0`.
    :param bool takes_integer_time: whether the model takes integer time steps (in `[0, n_timesteps-1]`) as input. Default is `False`.
    :param int n_timesteps: number of time steps for discrete schedules. Default is `1000`.
    :param bool _was_trained_on_minus_one_one: whether the model was trained on images in `[-1, 1]` range (`True`) or `[0, 1]` range (`False`). Default is `True`.
    :param str: device to load the model on. Default is `'cpu'`.
    """

    def __init__(
        self,
        score_model: nn.Module | Callable = None,
        prediction_type: str = "epsilon",  # prediction_type: "epsilon", "v_prediction", or "sample"
        clip_output: bool = True,
        sigma_t: Callable | torch.Tensor = None,
        scale_t: Callable | torch.Tensor = None,
        sigma_inverse: Callable = None,
        variance_preserving: bool = False,
        variance_exploding: bool = False,
        T: float = 1.0,
        takes_integer_time: bool = False,
        n_timesteps: int = 1000,
        _was_trained_on_minus_one_one: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.model = score_model
        self.clip_output = clip_output
        if prediction_type not in ["epsilon", "v_prediction", "sample"]:
            raise ValueError(
                f"Unsupported prediction_type: {prediction_type}. Supported types are 'epsilon', 'v_prediction', and 'sample'."
            )

        if variance_preserving and variance_exploding:
            raise ValueError(
                "variance_preserving and variance_exploding cannot both be True."
            )

        self.prediction_type = prediction_type
        self.takes_integer_time = takes_integer_time
        self.n_timesteps = n_timesteps
        self._was_trained_on_minus_one_one = _was_trained_on_minus_one_one
        self.variance_preserving = variance_preserving
        self.variance_exploding = variance_exploding

        self._initialize_schedules(sigma_t, scale_t)
        self.sigma_inverse = sigma_inverse
        self.T = T
        self.to(device)

    def _map_schedule(
        self, source_schedule: Tensor | Callable, transform_fn: Callable
    ) -> Tensor | Callable:
        """
        Applies a transform function to a schedule (Tensor or Callable).

        Guarantees:
            1. If Callable: input 't' is converted to a Tensor if it is a float/int.
            2. The output is always a Tensor (casting Python floats/ints if needed).
        """
        if isinstance(source_schedule, Callable):

            def wrapped_schedule(t):
                t = torch.as_tensor(t)
                val = source_schedule(t)
                res = transform_fn(val)
                # Ensure output is a Tensor
                if not torch.is_tensor(res):
                    res = torch.tensor(res, device=t.device, dtype=t.dtype)
                return res

            return wrapped_schedule

        else:
            # source_schedule is already a Tensor
            res = transform_fn(source_schedule)
            # Ensure output is a Tensor (e.g., if transform returned plain 1.0)
            if not torch.is_tensor(res):
                res = torch.tensor(
                    res, device=source_schedule.device, dtype=source_schedule.dtype
                )
        return res

    def _initialize_schedules(self, sigma_t, scale_t):
        """
        A helper function to initialize the schedules based on the provided arguments and configuration.
        """
        ops = {
            "vp_sigma_to_scale": lambda s: (1 / (1 + s**2)).sqrt(),
            "vp_scale_to_sigma": lambda s: (1 / s**2 - 1).clamp(min=0).sqrt(),
            "ve_sigma_to_scale": lambda s: torch.ones_like(s),
        }

        # Determine which operation to use based on configuration
        transform_op = None
        source = None
        target_name = None

        # If scale_t is None, but sigma_t is provided, we can try to compute scale_t from sigma_t if variance_preserving or variance_exploding is True. If neither is True, we skip the transformation since we don't know how to compute the missing schedule.
        if scale_t is None and sigma_t is not None:
            target_name = "scale"
            source = sigma_t
            if self.variance_preserving:
                transform_op = ops["vp_sigma_to_scale"]
            elif self.variance_exploding:
                transform_op = ops["ve_sigma_to_scale"]

        # scale_t is not None, but sigma_t is None, we can try to compute sigma_t from scale_t if variance_preserving is True. If variance_exploding is True, sigma_t is not defined since scale is always 1, so we skip the transformation in that case.
        elif sigma_t is None and scale_t is not None:
            target_name = "sigma"
            source = scale_t
            if self.variance_preserving:
                transform_op = ops["vp_scale_to_sigma"]

        else:
            # Either both schedules are provided, or both are None. In both cases, we do nothing.
            pass

        #  Apply the transformation
        if transform_op and source is not None:
            result = self._map_schedule(source, transform_op)
            # Assign back to the correct variable
            if target_name == "scale":
                scale_t = result
            else:
                sigma_t = result

        if isinstance(sigma_t, torch.Tensor):
            self.register_buffer("sigma_t", sigma_t)
        else:
            self.sigma_t = sigma_t

        if isinstance(scale_t, torch.Tensor):
            self.register_buffer("scale_t", scale_t)
        else:
            self.scale_t = scale_t

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
            time_idx = (t * (self.n_timesteps - 1) / self.T).long()
            val = schedule[time_idx]
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
        Computes the time step `t \in [0,T]` corresponding to a given noise level `sigma`.

        If an analytic inverse of the `sigma_t` is provided, it is used.
        Otherwise, a numeric inversion is performed (nearest neighbor for discrete schedules, binary search for continuous schedules).

        :param torch.Tensor | float sigma: noise level(s), either a scalar or a tensor of shape `[B]`.
        """

        sigma = torch.as_tensor(sigma)

        # 1) If user provided an analytic / predefined inverse, use it.
        if self.sigma_inverse is not None:
            return self.sigma_inverse(sigma)

        # 2) If we have a discrete table, use nearest index.
        if isinstance(self.sigma_t, torch.Tensor):
            sigmas = self.sigma_t  # [T]
            sigma = sigma.to(device=sigmas.device, dtype=sigmas.dtype)
            if sigma.dim() == 0:
                time_idx = torch.argmin((sigmas - sigma).abs())
                return time_idx.float() * self.T / (self.n_timesteps - 1)
            else:
                diffs = (sigmas[None, :] - sigma[:, None]).abs()  # [B, T]
                time_idx = torch.argmin(diffs, dim=1)
                return time_idx.float() * self.T / (self.n_timesteps - 1)
        else:
            # 3) Fallback: numeric inversion for continuous schedules (binary search).
            t_low = torch.zeros_like(sigma)
            t_high = torch.full_like(sigma, self.T)
            for _ in range(32):
                t_mid = (t_low + t_high) / 2
                sigma_mid = self.sigma_t(t_mid)
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
        :param torch.Tensor | float t: single timestep or tensor of shape `[B]` or `[]`.
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
        if self.takes_integer_time:
            t_model = (t * (self.n_timesteps - 1)).long()
        else:
            t_model = t
        # UNet forward
        pred = self.model(x, t_model, *args, return_dict=False, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred = pred.to(dtype)

        sigma = self.get_schedule_value(self.sigma_t, t, x.shape)
        scale = self.get_schedule_value(self.scale_t, t, x.shape)

        return self._pred_to_score(pred, x, sigma, scale)

    def forward(
        self,
        x: torch.Tensor,
        sigma: float | torch.Tensor = None,
        input_in_minus_one_one: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Applies denoiser :math:`\denoiser{x}{\sigma}`.
        If `input_in_minus_one_one` is `False` (default value), the input `x` is expected to be in `[0, 1]` range (up to random noise) and the output is also in `[0, 1]` range.
        Otherwise, both input and output are expected in `[-1, 1]` range.

        :param torch.Tensor x: noisy input, of shape `[B, C, H, W]`.
        :param torch.Tensor, float sigma: noise level. Can be a `float` or a :class:`torch.Tensor` of shape `[B]`.
            If a single `float` is provided, the same noise level is used for all samples in the batch.
            Otherwise, batch-wise noise levels are used.
        :param bool input_in_minus_one_one: whether the input `x` is in `[-1, 1]` range. Default is `False`.
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
        if not input_in_minus_one_one and self._was_trained_on_minus_one_one:
            sigma = sigma * 2  # since image is in [-1, 1] range in the model

        timestep = self.time_from_sigma(sigma.squeeze())
        scale = self.get_schedule_value(self.scale_t, timestep, x.shape)

        if not input_in_minus_one_one and self._was_trained_on_minus_one_one:
            # Rescale input x from [0, 1] to model scale [-1, 1] and apply scaling following DDPM
            x = (x * 2 - 1) * scale
        else:
            x = x * scale
        if self.takes_integer_time:
            t_model = (timestep * (self.n_timesteps - 1)).long()
        else:
            t_model = timestep
        # UNet forward
        pred = self.model(x, t_model, *args, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # take the first output if multiple outputs are returned
        pred = pred.to(dtype)

        # Convert model output to x0 depending on prediction type
        x0 = self._pred_to_x0(pred, x, sigma, scale)

        # Optional: clamp to model range [-1, 1]
        if self.clip_output:
            x0 = x0.clamp(-1, 1)

        if not input_in_minus_one_one:
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
        Currently, only models trained with `DDPMScheduler`, `DDIMScheduler` or `PNDMScheduler` are supported.

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
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        assert (
            mode_id is not None
        ), "Provide a diffusers model id. E.g., 'google/ddpm-cat-256'"

        try:
            from diffusers import (
                DiffusionPipeline,
                DDPMScheduler,
                PNDMScheduler,
                DDIMScheduler,
            )
        except ImportError:
            raise ImportError(
                "diffusers is not installed. Please install it via 'pip install diffusers'."
            )

        pipeline = DiffusionPipeline.from_pretrained(mode_id, torch_dtype=dtype).to(
            device
        )

        model = pipeline.unet
        scheduler = getattr(pipeline, "scheduler", None)
        prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")

        if isinstance(scheduler, (PNDMScheduler, DDPMScheduler, DDIMScheduler)):
            if hasattr(scheduler, "alphas_cumprod"):
                alphas_cumprod = scheduler.alphas_cumprod
                scale_t = torch.sqrt(alphas_cumprod)
            else:
                if scheduler.beta_schedule == "scaled_linear":
                    N = scheduler.config.num_train_timesteps
                    beta_start = 0.5 * N * scheduler.config.beta_start
                    beta_end = 0.5 * N * scheduler.config.beta_end
                    a = np.sqrt(beta_start)
                    c = np.sqrt(beta_end) - a
                    B_t = lambda t: (a**2) * t + a * c * t**2 + (c**2 / 3.0) * t**3
                    scale_t = lambda t: torch.exp(-B_t(t))
                elif scheduler.beta_schedule == "linear":
                    N = scheduler.config.num_train_timesteps
                    beta_start = 0.5 * scheduler.config.beta_start * N
                    beta_end = 0.5 * scheduler.config.beta_end * N
                    delta = beta_end - beta_start
                    scale_t = lambda t: torch.exp(
                        -(beta_start * t + 0.5 * delta * t**2)
                    )
                else:
                    raise ValueError(
                        "only 'scaled_linear' and 'linear' schedule are supported for beta"
                    )

            sigma_t = None
            variance_preserving = True
            variance_exploding = False

        else:
            raise ValueError(
                f"Scheduler of type {type(scheduler)} is not supported yet."
            )

        super().__init__(
            score_model=model,
            prediction_type=prediction_type,
            clip_output=clip_output,
            scale_t=scale_t,
            sigma_t=sigma_t,
            variance_preserving=variance_preserving,
            variance_exploding=variance_exploding,
            takes_integer_time=True,
            n_timesteps=scheduler.config.num_train_timesteps,
            device=device,
            *args,
            **kwargs,
        )

        self.scheduler = scheduler

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
        >>> denoiser = DRUNet()
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


class MinusOneOneDenoiserWrapper(nn.Module):
    r"""
    A wrapper for denoisers trained on :math:`[x_{\mathrm{min}}, x_{\mathrm{max}}]` images to be used with math:`[-1, 1]` images, i.e. on diffusion sampling iterates.

    :param deepinv.models.Denoiser denoiser: the denoiser to be wrapped.
    :param float xmin: minimum value of the denoiser training range. Default to `0.0`.
    :param float xmax: maximum value of the denoiser training range. Default to `1.0`.
    """

    def __init__(self, model: nn.Module, xmin: float = 0.0, xmax: float = 1.0):
        super().__init__()
        self.model = model
        self.xmin = xmin
        self.xmax = xmax

    def forward(self, x: Tensor, sigma: Tensor, *args, **kwargs) -> Tensor:
        # Scale from [-1, 1] to [xmin, xmax], except if specified otherwise with the 'input_in_minus_one_one' argument in kwargs
        if not kwargs.get("input_in_minus_one_one", True):
            x = (x + 1) / 2 * (self.xmax - self.xmin) + self.xmin
            sigma = sigma * (self.xmax - self.xmin) / 2
        denoised = self.model(x, sigma, *args, **kwargs)
        # Scale back to [-1, 1], except if specified otherwise with the 'input_in_minus_one_one' argument in kwargs
        if not kwargs.get("input_in_minus_one_one", True):
            denoised = 2 * (denoised - self.xmin) / (self.xmax - self.xmin) - 1
        return denoised
