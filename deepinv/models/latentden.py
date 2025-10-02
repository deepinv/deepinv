import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel


class LatentDiffusion(nn.Module):
    r"""
    Stable-Diffusion v1.5 latent model wrapper (UNet + VAE + CLIP).

    This module loads SD-1.5 components directly from the Hugging Face repo
    (default: ``runwayml/stable-diffusion-v1-5``) and exposes:

    * :py:meth:`forward` — predicts the noise :math:`\epsilon_\theta(x_t, t)` for a latent
      :math:`x_t` and timestep ``t`` using Classifier-Free Guidance (CFG).
    * :py:meth:`encode` — VAE encode image :math:`\to` latent (applies SD scaling factor).
    * :py:meth:`decode` — VAE decode latent :math:`\to` image (applies SD scaling factor).

    .. note::
       No scheduler/pipeline is created here; your sampler (e.g., DDIM/PSLD) must supply timesteps.
       The VAE interface here works in the range ``[-1, 1]`` for images.

    :param int num_inference_steps: kept for API symmetry; not used internally.
    :param float guidance_scale: CFG scale (``1.0`` effectively disables CFG).
    :param str prompt: kept for API symmetry; not used internally.
    :param int height: kept for API symmetry; not used internally.
    :param int width: kept for API symmetry; not used internally.
    :param str | torch.device device: device to load models on (e.g., ``"cuda"``).
    :param torch.dtype dtype: weights/activations dtype for UNet/VAE/TextEncoder (default: ``torch.float16``).
    :param str model_id: Hugging Face repo id for SD-1.5 components.
    """

    def __init__(
        self,
        guidance_scale=7.5,  # CFG scale
        device="cuda",
        dtype=torch.float16,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ):
        super().__init__()

        self.device = device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.guidance_scale = float(guidance_scale)
        self.dtype = dtype

        # --- Load components directly from the SD-1.5 repo ---
        # UNet (noise predictor)
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", variant="fp16", torch_dtype=dtype
        ).to(self.device)

        # VAE (latent encoder/decoder)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", variant="fp16", torch_dtype=dtype
        ).to(self.device)

        # Tokenizer and text encoder (CLIP)
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)

        # SD scaling factor (stored in the VAE config by Diffusers checkpoints)
        self._scaling: float = float(self.vae.config.scaling_factor)

        self.guidance_scale = guidance_scale

    def forward(self, x, t, prompt=None):
        r"""
        Predict the noise :math:`\epsilon_\theta(x_t, t)` with Classifier-Free Guidance.

        Steps:
          1. Tokenize & encode the conditional prompt(s) and an empty prompt (unconditional).
          2. Duplicate the latent batch along the batch dimension.
          3. Run the UNet once with concatenated unconditional/conditional embeddings.
          4. Combine the two outputs with ``guidance_scale``.

        :param torch.Tensor x: latent tensor :math:`x_t` of shape ``(B, C, H, W)``.
        :param torch.Tensor t: diffusion timestep(s); usually a 1-D tensor broadcast over the batch.
        :param str | Sequence[str] | None prompt: text prompt(s) for conditional guidance.
        :return: predicted noise :math:`\epsilon_\theta(x_t, t)` with the same shape as ``x``.
        :rtype: torch.Tensor
        """

        # Define the prompt
        self.prompt = [prompt]

        # Encode the prompt to conditioning embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        self.text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[
            0
        ]
        # Create unconditional (empty) prompt embeddings for CFG
        uncond_inputs = self.tokenizer(
            [""] * len(self.prompt),  # Empty prompt for unconditional guidance
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Encode unconditional prompt
        self.uncond_embeddings = self.text_encoder(
            uncond_inputs.input_ids.to(self.device)
        )[0]
        # Concatenate unconditional and conditional embeddings for CFG
        self.text_embeddings_cfg = torch.cat(
            [self.uncond_embeddings, self.text_embeddings], dim=0
        )

        # Expand latents for unconditional/conditional input for CFG
        latent_model_input = torch.cat([x.to(dtype=torch.float16)] * 2, dim=0)

        # Format timestep correctly
        t_tensor = t.to(dtype=torch.float16, device=self.device)
        # Forward pass through UNet
        noise_pred = self.unet(
            latent_model_input, t_tensor, encoder_hidden_states=self.text_embeddings_cfg
        ).sample

        # Split the outputs for CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        return noise_pred

    def encode(self, x):
        r"""
        Encode an image into the latent space using the VAE encoder.

        The input is clipped to ``[-1, 1]`` and mapped to latent space with the
        Stable-Diffusion scaling factor.

        :param torch.Tensor x: image tensor of shape ``(B, C, H, W)``.
        :return: latent tensor of shape ``(B, C_latent, H/8, W/8)``.
        :rtype: torch.Tensor
        """
        # Scale input to [-1, 1]
        # x = 2 * x - 1
        latents = (
            self.vae.encode(x.type(torch.float16).clip(-1, 1)).latent_dist.mean
            * self.vae.config.scaling_factor
        )

        return latents

    def decode(self, z):
        r"""
        Decode a latent representation into an image using the VAE decoder.

        Output values are clamped to ``[-1, 1]`` (no re-scaling to ``[0, 1]`` in this wrapper).

        :param torch.Tensor z: latent tensor of shape ``(B, C_latent, H/8, W/8)``.
        :return: image tensor of shape ``(B, C, H, W)`` in ``[-1, 1]``.
        :rtype: torch.Tensor
        """
        x = self.vae.decode(z / self.vae.config.scaling_factor).sample.clip(-1, 1)
        # x = (x + 1) / 2
        # x = x.clamp(0, 1)
        return x
