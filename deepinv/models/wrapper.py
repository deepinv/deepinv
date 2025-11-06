from __future__ import annotations
import torch
from deepinv.models import Denoiser
import warnings


class DiffusersDenoiserWrapper(Denoiser):
    """
    Wraps a diffusers model as a DeepInv Denoiser.

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

        self.register_buffer("sqrt_alpha_cumprod", ac.sqrt())  # alpha_t
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", (1.0 - ac).clamp(min=0).sqrt()
        )  # sigma_t
        self.register_buffer("beta_cumprod", 1 - ac)

        # prediction_type: "epsilon", "v_prediction", or "sample"
        self.prediction_type = getattr(
            self.scheduler.config, "prediction_type", "epsilon"
        )

        self.to(device)

    def _nearest_t_from_sigma(self, sigma: float | torch.Tensor) -> torch.Tensor:
        """
        Map a sigma to the nearest training time-step index.
        Supports scalar or per-batch tensor sigma.
        """
        sigmas = self.sqrt_one_minus_alpha_cumprod  # [T]
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
        timestep: int | torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Applies denoiser :math:`\denoiser{x}{\sigma}`.
        The input `x` is expected to be in `[0, 1]` range and the output is also in `[0, 1]` range.

        :param torch.Tensor x: noisy input, of shape `[B, C, H, W]`.
        :param torch.Tensor, float sigma: noise level. Can be a `float` or a :class:`torch.Tensor` of shape `[B]`.
            If a single `float` is provided, the same noise level is used for all samples in the batch.
            Otherwise, batch-wise noise levels are used.
        :param int, torch.Tensor timestep: an optional timestep index. Can be an `int` or a :class:`torch.Tensor` of shape `[B]`. If a single `int` is provided, the same timestep is used for all samples in the batch.
            Otherwise, batch-wise timesteps are used. This parameter is ignored if `sigma` is provided.
        :param args: additional positional arguments to be passed to the model.
        :param kwarg: additional keyword arguments to be passed to the model. For example, a `prompt` for text-conditioned or `class_label` for class-conditioned models.

        :returns: (:class:`torch.Tensor`) the denoised output.
        """

        device = x.device
        dtype = x.dtype

        assert (sigma is not None) or (
            timestep is not None
        ), "Provide either sigma or timestep."

        # Handle sigma
        sigma = self._handle_sigma(
            sigma,
            batch_size=x.shape[0],
            ndim=x.ndim,
            device=device,
            dtype=dtype,
        )

        sigma = sigma * 2  # since image is in [-1, 1] range in the model
        sqrt_alpha = 1 / (1 + sigma**2).sqrt()
        sigma = sigma * sqrt_alpha

        # Resolve timestep
        if sigma is not None:
            if timestep is not None:
                warnings.warn(
                    "Both sigma and timestep are provided. Ignoring timestep and using sigma."
                )
            timestep = self._nearest_t_from_sigma(sigma.squeeze())
        else:
            timestep = self._handle_sigma(
                timestep, batch_size=x.shape[0], ndim=1, device=device, dtype=torch.long
            )

        sqrt_alpha_bar_t = self._handle_sigma(
            self.sqrt_alpha_cumprod[timestep],
            batch_size=x.shape[0],
            ndim=x.ndim,
            device=device,
            dtype=dtype,
        )

        # Rescale input x from [0, 1] to model scale [-1, 1] and apply sqrt_alpha scaling following DDPM
        x = (x * 2 - 1) * sqrt_alpha
        # UNet forward
        pred = self.model(x, timestep, *args, return_dict=False, **kwargs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # take the first output if multiple outputs are returned
        pred = pred.to(dtype)

        # Convert model output to x0 depending on prediction type
        pt = self.prediction_type
        if pt == "epsilon":
            x0 = (x - sigma * pred) / sqrt_alpha_bar_t
        elif pt == "v_prediction":
            x0 = sqrt_alpha_bar_t * x - sigma * pred
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
