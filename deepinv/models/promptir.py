"""
PromptIR Model
Code borrowed from Potlapalli et al., at: https://github.com/va1shn9v/PromptIR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import get_weights_url
from .restormer import Downsample, Upsample, TransformerBlock, OverlapPatchEmbed


class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(
            prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1
        ) * self.prompt_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt


class PromptIR(nn.Module):
    r"""
    PromptIR restoration model.

    PromptIR is a blind restoration model that was proposed in :footcite:t:`potlapalli2023promptir`.

    The authors' pretrained weights for in_channels=out_channels=3 can be downloaded via setting ``pretrained='download'``.

    :param int in_channels: number of channels of the input.
    :param int out_channels: number of channels of the output.
    :param int dim: base dimension of the model.
    :param tuple num_blocks: number of transformer blocks at each level of the encoder/decoder
    :param int num_refinement_blocks: number of transformer blocks in the refinement module.
    :param tuple heads: number of attention heads at each level of the encoder/decoder.
    :param float ffn_expansion_factor: expansion factor of the feed-forward networks.
    :param bool bias: whether to use bias in the convolutional layers.
    :param str LayerNorm_type: type of layer normalization to use ('BiasFree' or 'WithBias').
    :param bool decoder: whether to use the decoder with prompt generation blocks.
    :param torch.device | str device: device to load the model on.
    :param str pretrained: path to the pretrained weights or 'download' to download the authors' weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple[int, int, int, int] = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = "WithBias",
        decoder: bool = True,
        device: torch.device | str = None,
        pretrained: str = None,
    ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(
                prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96
            )
            self.prompt2 = PromptGenBlock(
                prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192
            )
            self.prompt3 = PromptGenBlock(
                prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384
            )

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)
        self.reduce_noise_channel_2 = nn.Conv2d(
            int(dim * 2**1) + 128, int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))
        self.reduce_noise_channel_3 = nn.Conv2d(
            int(dim * 2**2) + 256, int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**2))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**1) + 192, int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.noise_level3 = TransformerBlock(
            dim=int(dim * 2**2) + 512,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level3 = nn.Conv2d(
            int(dim * 2**2) + 512, int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.noise_level2 = TransformerBlock(
            dim=int(dim * 2**1) + 224,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level2 = nn.Conv2d(
            int(dim * 2**1) + 224, int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.noise_level1 = TransformerBlock(
            dim=int(dim * 2**1) + 64,
            num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        self.reduce_noise_level1 = nn.Conv2d(
            int(dim * 2**1) + 64, int(dim * 2**1), kernel_size=1, bias=bias
        )

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.device = device

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def load_pretrained(self, checkpoint_path):

        # Load checkpoint
        if checkpoint_path == "download":
            name = "promptir.ckpt"
            url = get_weights_url(model_name="promptir", file_name=name)
            checkpoint = torch.hub.load_state_dict_from_url(
                url, map_location=lambda storage, loc: storage, file_name=name
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle Lightning checkpoint format
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("net."):
                    new_key = key[4:]  # Remove 'net.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            self.load_state_dict(new_state_dict, strict=True)
        else:
            self.load_state_dict(checkpoint, strict=True)

    def forward(self, inp_img, noise_emb=None):

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
