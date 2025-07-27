import torch
from torch.nn.functional import silu
import numpy as np
from .utils import (
    PositionalEmbedding,
    FourierEmbedding,
    UNetBlock,
    UpDownConv2d,
)
from .base import Denoiser
from torch.nn import Linear, GroupNorm
from .utils import get_weights_url


class NCSNpp(Denoiser):
    r"""Implementation of the DDPM++ and NCSN++ architectures.

    Equivalent to the original implementation by :footcite:t:`song2020score`, available at `the official implementation <https://github.com/yang-song/score_sde_pytorch>`_.

    The model is also pre-conditioned by the method described in :footcite:t:`karras2022elucidating`.

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
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (the default model trained on FFHQ at 64x64 resolution (`ffhq64-uncond-ve`) with default architecture).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        In this case, the model is supposed to be trained on `[0,1]` pixels, if it was trained on `[-1, 1]` pixels, the user should set the attribute `_train_on_minus_one_one` to `True` after loading the weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param float pixel_std: The standard deviation of the normalized pixels (to `[0, 1]` for example) of the data distribution. Default to `0.75`.
    :param torch.device device: Instruct our module to be either on cpu or on gpu. Default to ``None``, which suggests working on cpu.



    """

    def __init__(
        self,
        img_resolution: int = 64,  # Image spatial resolution at input/output.
        in_channels: int = 3,  # Number of color channels at input.
        out_channels: int = 3,  # Number of color channels at output.
        label_dim: int = 0,  # Number of class labels, 0 = unconditional.
        augment_dim: int = 9,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels: int = 128,  # Base multiplier for the number of channels.
        channel_mult: list = [
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb: int = 4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks: int = 4,  # Number of residual blocks per resolution.
        attn_resolutions: list = [16],  # List of resolutions with self-attention.
        dropout: float = 0.10,  # Dropout probability of intermediate activations.
        label_dropout: float = 0.0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type: str = "fourier",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise: int = 2,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type: str = "residual",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type: str = "standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter: list = [
            1,
            3,
            3,
            1,
        ],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        pretrained: str = "download",
        pixel_std: float = 0.75,
        device=None,
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
        self.pixel_std = pixel_std
        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = (
            Linear(in_features=label_dim, out_features=noise_channels)
            if label_dim
            else None
        )
        self.map_augment = (
            Linear(in_features=augment_dim, out_features=noise_channels, bias=False)
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = UpDownConv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = UpDownConv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = UpDownConv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = UpDownConv2d(
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
                    self.dec[f"{res}x{res}_aux_up"] = UpDownConv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(
                    num_channels=cout,
                    eps=1e-6,
                    num_groups=32,
                )
                self.dec[f"{res}x{res}_aux_conv"] = UpDownConv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

        if pretrained is not None:
            if (
                pretrained.lower() == "edm-ffhq64-uncond-ve"
                or pretrained.lower() == "download"
            ):
                name = "ncsnpp-ffhq64-uncond-ve.pt"
                url = get_weights_url(model_name="edm", file_name=name)
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
                self._train_on_minus_one_one = True  # Pretrained on [-1,1]s
                self.pixel_std = 0.5
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
                self._train_on_minus_one_one = False
            self.load_state_dict(ckpt, strict=True)
        else:
            self._train_on_minus_one_one = False
        self.eval()
        if device is not None:
            self.to(device)
            self.device = device

    def forward_unet(self, x, sigma, class_labels=None, augment_labels=None):
        r"""
        Run the unet.

        :param torch.Tensor x: noisy image
        :param Union[torch.Tensor, float]  sigma: noise level
        :param torch.Tensor class_labels: class labels
        :param torch.Tensor augment_labels: augmentation labels
        :return torch.Tensor: denoised image.
        """
        emb = self.map_noise(sigma)
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
        dtype = x.dtype
        x = x.to(torch.float32)

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

        D_x = D_x.to(dtype)
        # Rescale [-1,1] output to [0,-1]
        if getattr(self, "_train_on_minus_one_one", False):
            return (D_x + 1.0) / 2.0
        else:
            return D_x
