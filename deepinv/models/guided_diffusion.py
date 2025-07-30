import torch
from torch.nn.functional import silu
import numpy as np
from .utils import (
    PositionalEmbedding,
    UNetBlock,
    UpDownConv2d,
)
from .base import Denoiser

from torch.nn import Linear, GroupNorm
from math import floor
from .utils import get_weights_url


class ADMUNet(Denoiser):
    r"""
    Implementation of the ADM UNet diffusion model.

    From the paper of :footcite:t:`dhariwal2021diffusion`.

    The model is also pre-conditioned by the method described in the EDM paper :footcite:t:`karras2022elucidating`.

    Equivalent to the original implementation by Dhariwal and Nichol, available at: https://github.com/openai/guided-diffusion.
    The architecture consists of a series of convolution layer, down-sampling residual blocks and up-sampling residual blocks with skip-connections.
    Each residual block has a self-attention mechanism with `64` channels per attention head with the up/down-sampling from BigGAN..
    The noise level is embedded using Positional Embedding with optional augmentation linear layer.

    :param int img_resolution: Image spatial resolution at input/output.
    :param int in_channels: Number of color channels at input.
    :param int out_channels: Number of color channels at output.
    :param int label_dim: Number of class labels, 0 = unconditional.
    :param int augment_dim: Augmentation label dimensionality, 0 = no augmentation.
    :param int model_channels: Base multiplier for the number of channels.
    :param list[int] channel_mult: Per-resolution multipliers for the number of channels.
    :param int channel_mult_emb: Multiplier for the dimensionality of the embedding vector.
    :param int num_blocks: Number of residual blocks per resolution.
    :param list[int] attn_resolutions: List of resolutions with self-attention.
    :param float dropout: dropout probability used in residual blocks.
    :param float label_dropout: Dropout probability of class labels for classifier-free guidance.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (the default model is a conditional model trained on ImageNet at 64x64 resolution (`imagenet64-cond`) with default architecture).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        In this case, the model is supposed to be trained on `[0,1]` pixels, if it was trained on `[-1, 1]` pixels, the user should set the attribute `_train_on_minus_one_one` to `True` after loading the weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param float pixel_std: The standard deviation of the normalized pixels (to `[0, 1]` for example) of the data distribution. Default to `0.75`.
    :param torch.device device: Instruct our module to be either on cpu or on gpu. Default to ``None``, which suggests working on cpu.


    """

    def __init__(
        self,
        img_resolution: int = 64,  # Image resolution at input/output.
        in_channels: int = 3,  # Number of color channels at input.
        out_channels: int = 3,  # Number of color channels at output.
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
        pretrained: str = "download",
        pixel_std: float = 0.75,
        device=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # The default model is a class-conditioned model with 1000 classes
        if pretrained is not None:
            if (
                pretrained.lower() == "imagenet64-cond"
                or pretrained.lower() == "download"
            ):
                label_dim = 1000

        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
        )
        self.pixel_std = pixel_std
        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels)
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = floor(img_resolution / 2**level)
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = UpDownConv2d(
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
            res = floor(img_resolution / 2**level)
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
        self.out_norm = GroupNorm(
            num_channels=cout,
            num_groups=32,
        )
        self.out_conv = UpDownConv2d(
            in_channels=cout, out_channels=out_channels, kernel=3
        )
        if pretrained is not None:
            if (
                pretrained.lower() == "edm-imagenet64-cond"
                or pretrained.lower() == "download"
            ):
                name = "adm-imagenet64-cond.pt"
                url = get_weights_url(model_name="edm", file_name=name)
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )

                self._train_on_minus_one_one = True  # Pretrained on [-1,1]
                self.pixel_std = 0.5
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
                self._train_on_minus_one_one = False  # Pretrained on [0,1]
            self.load_state_dict(ckpt, strict=True)
        else:
            self._train_on_minus_one_one = False
        self.eval()
        if device is not None:
            self.to(device)
            self.device = device

    def forward(
        self, x, sigma, class_labels=None, augment_labels=None, *args, **kwargs
    ):
        r"""
        Run the denoiser on noisy image.

        :param torch.Tensor x: noisy image
        :param Union[torch.Tensor, float]  sigma: noise level
        :param torch.Tensor class_labels: class labels
        :param torch.Tensor augment_labels: augmentation labels
        :return torch.Tensor: denoised image.
        """
        if class_labels is not None:
            class_labels = class_labels.to(torch.float32)
        sigma = self._handle_sigma(
            sigma, batch_size=x.size(0), ndim=x.ndim, device=x.device, dtype=x.dtype
        )

        # Rescale [0,1] input to [-1,-1]
        if getattr(self, "_train_on_minus_one_one", False):
            x = (x - 0.5) * 2.0
            sigma = sigma * 2.0

        c_skip = self.pixel_std**2 / (sigma**2 + self.pixel_std**2)
        c_out = sigma * self.pixel_std / (sigma**2 + self.pixel_std**2).sqrt()
        c_in = 1 / (self.pixel_std**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.forward_unet(
            c_in * x,
            c_noise.flatten(),
            class_labels=class_labels,
            augment_labels=augment_labels,
        )
        D_x = c_skip * x + c_out * F_x

        # Rescale [-1,1] output to [0,-1]
        if getattr(self, "_train_on_minus_one_one", False):
            return (D_x + 1.0) / 2.0
        else:
            return D_x

    def forward_unet(self, x, sigma, class_labels=None, augment_labels=None):
        r"""
        Run the unet on noisy image.

        :param torch.Tensor x: noisy image
        :param Union[torch.Tensor, float] sigma: noise level
        :param torch.Tensor class_labels: class labels
        :param torch.Tensor augment_labels: augmentation labels

        :return: (:class:`torch.Tensor`) denoised image.
        """
        # Mapping.
        emb = self.map_noise(sigma)
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
