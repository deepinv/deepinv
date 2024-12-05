import torch
from torch.nn.functional import silu
import numpy as np
from .utils import (
    PositionalEmbedding,
    Linear,
    UNetBlock,
    Conv2d,
    GroupNorm,
)


class ADMUNet(torch.nn.Module):
    r"""
    Re-implementation of the architecture from the paper: Diffusion Models Beat GANS on Image Synthesis (https://arxiv.org/abs/2105.05233).
    Equivalent to the original implementation by Dhariwal and Nichol, available at: https://github.com/openai/guided-diffusion.
    """

    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=model_channels, out_features=emb_channels, **init
        )
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init
        )
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        skips = [block.out_channels for block in self.enc.values()]

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
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs,
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
        )

    def forward(self, x, noise_level, class_labels=None, augment_labels=None):
        # Mapping.
        noise_level = self._handle_sigma(noise_level, x.dtype, x.device, x.size(0))
        emb = self.map_noise(noise_level)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout
                ).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

    @classmethod
    def from_pretrained(cls, model_name: str = "imagenet64-cond"):
        r"""
        Load a pretrained model from the Hugging Face Hub.

        :param str model_name: Name of the model to load.

        :return NCSNpp: The loaded model.
        """
        if "imagenet64" in model_name:
            default_64x64_config = dict(
                img_resolution=64,
                in_channels=3,
                out_channels=3,
                augment_dim=0,
                label_dim=1000,
            )
            model = cls(**default_64x64_config)
            model_url = f"https://huggingface.co/mhnguyen712/edm/resolve/main/adm-{model_name.lower()}.pt"
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
