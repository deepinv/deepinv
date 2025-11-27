from __future__ import annotations
import torch
from deepinv.models import Denoiser


class ScoreModelWrapper(Denoiser):
    """
    Wraps a score model as a DeepInv Denoiser.
    """

    def __init__(
        self,
        score_model: nn.Module = None,
        prediction_type: str = "epsilon", # prediction_type: "epsilon", "v_prediction", or "sample"
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
                    def scale_t(t):
                        t = self._handle_time_step(t)
                        return (1 / (1 + self.sigma_t(t) ** 2)) ** 0.5
                else:
                    scale_schedule = 1.0 / (1 + sigma_schedule ** 2) ** 0.5
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
        self.device = device
    
        self.to(device)

    def _handle_time_step(self, t: Tensor | float) -> Tensor:
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t

    def t_from_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:

        sigma = torch.as_tensor(sigma, device=self.device, dtype=self.dtype)

        # 1) If user provided an analytic / predefined inverse, use it.
        if self.sigma_inverse is not None:
            t = self.sigma_inverse(sigma)
            return t

        # 2) If we have a discrete table, use nearest index.
        if isinstance(self.sigma_schedule, torch.Tensor):
            sigmas = self.sigma_schedule  # [T]

            if sigma.dim() == 0:
                return torch.argmin((sigmas - sigma).abs())
            else:
                diffs = (sigmas[None, :] - sigma[:, None]).abs()  # [B, T]
                return torch.argmin(diffs, dim=1)
        else:
            # 3) Fallback: numeric inversion for continuous schedules (binary search, as before).
            t_min = torch.tensor(0.0, device=self.device)
            t_max = torch.tensor(self.T, device=self.device)
            t_low = t_min.expand_as(sigma).clone()
            t_high = t_max.expand_as(sigma).clone()

            for _ in range(32):
                t_mid = 0.5 * (t_low + t_high)
                sigma_mid = self.sigma_schedule(t_mid)
                go_right = sigma_mid < sigma  # flip sign if schedule is decreasing
                t_low = torch.where(go_right, t_mid, t_low)
                t_high = torch.where(go_right, t_high, t_mid)
            t_est = 0.5 * (t_low + t_high)
            return t_est[0] if t_est.numel() == 1 else t_est


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
        if isinstance(self.scale_schedule, torch.Tensor):
            scale = self.scale_schedule[timestep].view(-1, *(1,) * (x.ndim - 1))
        else:
            scale = self.scale_schedule(timestep).view(-1, *(1,) * (x.ndim - 1))

        # Rescale input x from [0, 1] to model scale [-1, 1] and apply scaling following DDPM
        x = (x * 2 - 1) * scale
        # UNet forward
        pred = self.model(x, timestep, *args, return_dict=False, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # take the first output if multiple outputs are returned
        pred = pred.to(dtype)

        # Convert model output to x0 depending on prediction type
        pt = self.prediction_type
        if pt == "epsilon": # predics white noise
            x0 = x / scale - sigma * pred
        elif pt == "v_prediction": # predics s_t*eps - sigma_t * x. See https://arxiv.org/pdf/2202.00512. 
            x0 = scale * (x - sigma * pred)
        elif pt == "sample": # predics the denoised image
            x0 = pred
        else:
            raise ValueError(f"Unsupported prediction_type: {pt}")

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
        prediction_type = getattr(
            self.scheduler.config, "prediction_type", "epsilon"
        )

        assert isinstance(
            scheduler, DDPMScheduler
        ), "Currently, only DDPMScheduler is supported."
        scale_schedule = ac.sqrt() 
        sigma_schedule = (1 / ac - 1.0).clamp(min=0).sqrt()
        
        super().__init__(model = model, prediction_type = prediction_type, clip_output = clip_output, scale_schedule = scale_schedule, sigma_schedule = sigma_schedule, device = device)