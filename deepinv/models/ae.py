from __future__ import annotations
import torch
from torch import nn
from typing import Optional, Union, Tuple
from pathlib import Path
from .base import Denoiser


class AutoEncoder(Denoiser):
    r"""
    Simple fully connected autoencoder network.

    Simple architecture that can be used for debugging or fast prototyping.

    :param int dim_input: total number of elements (pixels) of the input.
    :param int dim_hid: number of features in intermediate layer.
    :param int dim_hid: latent space dimension.
    :param int residual: use a residual connection between input and output.

    """

    def __init__(self, dim_input, dim_mid=1000, dim_hid=32, residual=True):
        super().__init__()
        self.residual = residual

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_input, dim_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_mid, dim_hid),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(dim_hid, dim_mid),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_mid, dim_input),
        )

    def forward(self, x, sigma=None, **kwargs):
        B, *S = x.shape

        x = x.reshape(B, -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if self.residual:
            decoded = decoded + x

        decoded = decoded.reshape(B, *S)
        return decoded


class VAE(Denoiser):
    """
    Unified VAE wrapper for:
      • Diffusers AutoencoderKL  (pixel ↔ latent, [0,1]↔[-1,1], scaling_factor)
      • Generic autoencoders with .encoder/.decoder (e.g., the AutoEncoder class)

    API:
      - encode(x) -> z      # Diffusers: scaled latents; Generic: encoder(x_flat)
      - decode(z) -> x_hat  # Diffusers: to [0,1];    Generic: decoder(z) reshaped
      - forward(x, sigma) = decode(encode(x))

    Notes (generic backend):
      - Inputs are flattened to (B, -1) for .encoder.
      - Outputs are reshaped back using `data_shape` (C,H,W,...) or last seen input.
      - No range mapping or scaling is applied (you own any normalization).
    """

    def __init__(
        self,
        vae: Optional[nn.Module] = None,
        model_id_or_path: Optional[Union[str, Path]] = None,
        *,
        # common
        device: Union[str, torch.device] = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        backend: str = "auto",                 # "auto" | "diffusers" | "generic"
        scaling_factor: Optional[float] = "auto",
        sample_default: bool = False,
        # diffusers-only conveniences
        subfolder: Optional[str] = None,
        use_slicing: bool = False,
        use_tiling: bool = False,
        # generic-only convenience
        data_shape: Optional[Tuple[int, ...]] = None,  # expected output shape (non-batch)
        **from_pretrained_kwargs,
    ):
        super().__init__(device=device)

        # --------------- load / accept a VAE ---------------
        if vae is None and model_id_or_path is None:
            raise ValueError("Provide either `vae` or `model_id_or_path`.")

        if vae is None:
            # lazy-load diffusers VAE
            AutoencoderKL = self._lazy_import_autoencoderkl()
            vae = AutoencoderKL.from_pretrained(
                str(model_id_or_path),
                subfolder=subfolder,
                torch_dtype=torch_dtype,
                **from_pretrained_kwargs,
            )

        # place + dtype
        if torch_dtype is not None:
            vae = vae.to(dtype=torch_dtype)
        vae = vae.to(device)

        self.vae = vae
        self.sample_default = bool(sample_default)
        self.backend = self._resolve_backend(vae, model_id_or_path, backend)

        # perf knobs (diffusers only)
        if self.backend == "diffusers":
            if hasattr(vae, "set_use_slicing"):
                vae.set_use_slicing(use_slicing)
            if hasattr(vae, "set_use_tiling"):
                vae.set_use_tiling(use_tiling)

        # scaling factor
        if self.backend == "diffusers":
            self.scaling_factor = self._resolve_scaling_factor(vae, scaling_factor)
        else:
            # generic AEs typically don't use a scaling factor
            self.scaling_factor = 1.0 if scaling_factor in ("auto", None) else float(scaling_factor)

        # for device/dtype access and .to()
        first_param = next(self.vae.parameters(), None)
        p_dtype = first_param.dtype if first_param is not None else (torch_dtype or torch.float32)
        self._dummy = nn.Parameter(torch.zeros((), device=device, dtype=p_dtype))

        # generic reshape handling
        self._data_shape = data_shape  # (C,H,W,...) or any non-batch shape
        self._last_seen_shape: Optional[Tuple[int, ...]] = None

        self._generic_residual = getattr(vae, "residual", False)
        self._last_input_flat = None

    # ------------------------ backend resolution ------------------------

    @staticmethod
    def _lazy_import_autoencoderkl():
        try:
            from diffusers.models import AutoencoderKL
        except ImportError:
            try:
                from diffusers import AutoencoderKL
            except ImportError as e:
                raise ImportError(
                    "The `diffusers` package is required to load a Diffusers VAE.\n"
                    "Install: pip install diffusers"
                ) from e
        return AutoencoderKL

    @staticmethod
    def _looks_like_diffusers_vae(vae: nn.Module) -> bool:
        # Heuristic: has .encode/.decode and a config with scaling_factor
        cfg = getattr(vae, "config", None)
        return hasattr(vae, "encode") and hasattr(vae, "decode") and hasattr(cfg, "scaling_factor")

    @staticmethod
    def _looks_like_generic_ae(vae: nn.Module) -> bool:
        # Heuristic: has .encoder and .decoder submodules (like provided AutoEncoder)
        return hasattr(vae, "encoder") and hasattr(vae, "decoder")

    def _resolve_backend(self, vae: nn.Module, model_id_or_path, backend: str) -> str:
        if backend in ("diffusers", "generic"):
            return backend
        # auto-detect
        if model_id_or_path is not None:
            return "diffusers"  # loading from repo/path implies diffusers
        if self._looks_like_diffusers_vae(vae):
            return "diffusers"
        if self._looks_like_generic_ae(vae):
            return "generic"
        # fallback: if it has encode/decode methods, assume diffusers-like
        return "diffusers" if (hasattr(vae, "encode") and hasattr(vae, "decode")) else "generic"

    # ------------------------ scaling factor (diffusers) ------------------------

    @staticmethod
    def _resolve_scaling_factor(vae, scaling_factor):
        if scaling_factor == "auto" or scaling_factor is None:
            cfg = getattr(vae, "config", None)
            if cfg is not None and hasattr(cfg, "scaling_factor"):
                return float(cfg.scaling_factor)
            if hasattr(vae, "scaling_factor"):
                try:
                    return float(getattr(vae, "scaling_factor"))
                except Exception:
                    pass
            return 0.18215  # SD/SDXL default latent scaling
        return float(scaling_factor)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    # ----------------------------- encode -----------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, sample: Optional[bool] = None) -> torch.Tensor:
        """
        Diffusers:
            expects x in [0,1], maps to [-1,1], returns scaled latents.
        Generic:
            flattens to (B,-1), runs .encoder, returns latent vector (no scaling).
        """
        if sample is None:
            sample = self.sample_default

        if self.backend == "diffusers":
            x = x.to(device=self.device, dtype=next(self.vae.parameters()).dtype)
            x = (x - 0.5) * 2.0  # [0,1] -> [-1,1]
            out = self.vae.encode(x)
            # Diffusers returns AutoencoderKLOutput with .latent_dist
            dist = getattr(out, "latent_dist", None)
            if dist is None:
                # Some VAEs might return distribution directly
                dist = out
            z = dist.sample() if sample else dist.mean
            return z * self.scaling_factor

        # generic AE
        B, *S = x.shape
        self._last_seen_shape = tuple(S)

        x_flat = x.reshape(B, -1).to(device=self.device, dtype=next(self.vae.parameters()).dtype)
        if self._generic_residual:
            self._last_input_flat = x_flat  # cache for decode()
        if not hasattr(self.vae, "encoder"):
            raise AttributeError("Generic backend expects `vae.encoder` module.")
        z = self.vae.encoder(x_flat)
        return z

    # ----------------------------- decode -----------------------------

    @torch.no_grad()
    def decode(self, z: torch.Tensor, *, output_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """
        Diffusers:
            expects scaled latents; returns [0,1] image.
        Generic:
            expects latent vector; returns tensor reshaped to `output_shape` (or last seen / data_shape).
        """
        if self.backend == "diffusers":
            z = z.to(device=self.device, dtype=next(self.vae.parameters()).dtype)
            z_unscaled = z / self.scaling_factor
            x_hat = self.vae.decode(z_unscaled).sample
            x_hat = (x_hat / 2.0 + 0.5).clamp(0.0, 1.0)
            return x_hat

        # generic AE
        if not hasattr(self.vae, "decoder"):
            raise AttributeError("Generic backend expects `vae.decoder` module.")

        B = z.shape[0]
        y_flat = self.vae.decoder(z.to(device=self.device, dtype=next(self.vae.parameters()).dtype))

        # choose target shape (non-batch)
        target = (
            output_shape
            or self._data_shape
            or self._last_seen_shape
        )
        if target is None:
            # No shape info—return as flat (B, *)
            return y_flat
        
        if self._generic_residual and self._last_input_flat is not None:
            if self._last_input_flat.shape[0] == y_flat.shape[0]:
                y_flat = y_flat + self._last_input_flat

        return y_flat.view(B, *target)

    # ----------------------------- Denoiser API -----------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sigma=None, **kwargs) -> torch.Tensor:
        sample = kwargs.pop("sample", self.sample_default)
        # For generic AE, this will remember the shape for decode()
        return self.decode(self.encode(x, sample=sample))

    # ----------------------------- constructors -----------------------------

    @classmethod
    def from_pretrained(cls, model_id_or_path: Union[str, Path], **kwargs) -> "VAE":
        return cls(vae=None, model_id_or_path=model_id_or_path, **kwargs)

    @classmethod
    def from_generic(cls, vae: nn.Module, *, data_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> "VAE":
        """
        Convenience constructor when passing a generic AE (with .encoder/.decoder).
        """
        return cls(vae=vae, backend="generic", data_shape=data_shape, **kwargs)
