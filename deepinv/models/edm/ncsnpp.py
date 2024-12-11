import torch
from torch.nn.functional import silu
from typing import List
import numpy as np
from .utils import (
    PositionalEmbedding,
    FourierEmbedding,
    Linear,
    UNetBlock,
    Conv2d,
    GroupNorm,
)
from ..base import Denoiser


class NCSNpp(Denoiser):
    r"""Re-implementation of the DDPM++ and NCSN++ architectures from the paper: `Score-Based Generative Modeling through Stochastic Differential Equations <https://arxiv.org/abs/2011.13456>`_.
    Equivalent to the original implementation by Song et al., available at `<https://github.com/yang-song/score_sde_pytorch`_.

    The architecture consists of a series of convolution layer, down-sampling residual blocks and up-sampling residual blocks with skip-connections of scale :math:`\sqrt{0.5}`.
    The model also supports an additional class condition model.
    Each residual block has a self-attention mechanism with multiple channels per attention head.
    The noise level can be embedded using either Positional Embedding  or Fourier Embedding with optional augmentation linear layer.

    :param int img_resolution: Image spatial resolution at input/output.
    :param int in_channels: Number of color channels at input.
    :param int out_channels: Number of color channels at output.
    :param int label_dim: Number of class labels, 0 = unconditional.
    :param int augment_dim: Augmentation label dimensionality, 0 = no augmentation.
    :param int model_channels: Base multiplier for the number of channels.
    :param list channel_mult: Per-resolution multipliers for the number of channels.
    :param int channel_mult_emb: Multiplier for the dimensionality of the embedding vector.
    :param int num_blocks: Number of residual blocks per resolution.
    :param list attn_resolutions: List of resolutions with self-attention.
    :param float dropout: Dropout probability of intermediate activations.
    :param float label_dropout: Dropout probability of class labels for classifier-free guidance.
    :param str embedding_type: Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
    :param int channel_mult_noise: Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
    :param str encoder_type: Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
    :param str decoder_type: Decoder architecture: 'standard' for both DDPM++ and NCSN++.
    :param list resample_filter: Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.

    """

    def __init__(
        self,
        img_resolution: int,  # Image spatial resolution at input/output.
        in_channels: int = 3,  # Number of color channels at input.
        out_channels: int = 3,  # Number of color channels at output.
        label_dim: int = 0,  # Number of class labels, 0 = unconditional.
        augment_dim: int = 9,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels: int = 128,  # Base multiplier for the number of channels.
        channel_mult: List = [
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb: int = 4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks: int = 4,  # Number of residual blocks per resolution.
        attn_resolutions: List = [16],  # List of resolutions with self-attention.
        dropout: float = 0.10,  # Dropout probability of intermediate activations.
        label_dropout: float = 0.0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type: str = "fourier",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise: int = 2,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type: str = "residual",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type: str = "standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter: List = [
            1,
            3,
            3,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels, **init)
            if label_dim
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim, out_features=noise_channels, bias=False, **init
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_level, class_labels=None, augment_labels=None):
        r"""
        Run the denoiser on noisy image.

        :param torch.Tensor x: noisy image
        :param torch.Tensor noise_level: noise level
        :param torch.Tensor class_labels: class labels
        :param torch.Tensor augment_labels: augmentation labels

        :return torch.Tensor: denoised image.
        """
        # Mapping.
        noise_level = self._handle_sigma(noise_level, x.dtype, x.device, x.size(0))
        emb = self.map_noise(noise_level)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

    @classmethod
    def from_pretrained(cls, model_name: str = "edm-ffhq64-uncond-ve"):
        r"""
        Load a pretrained model from the Hugging Face Hub.

        :param str model_name: Name of the model to load.

        :return NCSNpp: The loaded model.
        """
        if "ffhq64" in model_name or "afhq64" in model_name:
            default_64x64_config = dict(
                img_resolution=64,
                in_channels=3,
                out_channels=3,
                augment_dim=9,
                model_channels=128,
                channel_mult=[1, 2, 2, 2],
                channel_mult_noise=2,
                embedding_type="fourier",
                encoder_type="residual",
                decoder_type="standard",
                resample_filter=[1, 3, 3, 1],
            )
            model = cls(**default_64x64_config)
            model_url = f"https://huggingface.co/mhnguyen712/edm/resolve/main/ncsnpp-{model_name.lower()}.pt"
            state_dict = torch.hub.load_state_dict_from_url(
                model_url,
                file_name=model_name,
                map_location=lambda storage, loc: storage,
            )
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        return model

    @staticmethod
    def _handle_sigma(sigma, dtype, device, batch_size):
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim == 0:
                return sigma[None].to(device, dtype).expand(batch_size)
            elif sigma.ndim == 1:
                assert (
                    sigma.size(0) == batch_size or sigma.size(0) == 1
                ), "sigma must be a Tensor with batch_size equal to 1 or the batch_size of input images"
                return sigma.to(device, dtype).expand(batch_size // sigma.size(0))

            else:
                raise ValueError(f"Unsupported sigma shape {sigma.shape}.")

        elif isinstance(sigma, (float, int)):
            return torch.tensor([sigma]).to(device, dtype).expand(batch_size)
        else:
            raise ValueError("Unsupported sigma type.")
