from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from deepinv.models.base import Denoiser
from deepinv.models.utils import batchnorm_nd, conv_nd, conv_transpose_nd, fix_dim

HALF_LOG_TWO_PI = 0.91894


class VAE(Denoiser):
    r"""
    Simple Variational Autoencoder (VAE) model with modular encoder and decoder
    architectures.

    The implemented architecture follows the following formulation:

    - a generative model :math:`p_\theta(x|z) = \mathcal{N}(x; D_\theta(z), \mathrm{diag}(\sigma^2_\theta(z)))`
      if `decode_mean_only=False` where :math:`\sigma_\theta(z)` is predicted using a second deocder.
      If `decode_mean_only=True`, :math:`p_\theta(x|z) = \mathcal{N}(x; D_\theta(z), \gamma^2 I)`
      with :math:`\gamma` a learnable scalar parameter.

    - an inference model :math:`q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))`

    - a prior model :math:`p(z) = \mathcal{N}(z; 0, I)`

    :param int in_channels: Number of input channels.
    :param int latent_dim: Dimensionality of the latent space.
    :param bool decode_mean_only: If `True`, only the mean decoder is used and the output variance is fixed to a learnable scalar parameter.
    :param bool fully_conv_bottleneck: If `True`, uses a fully convolutional bottleneck.
        Otherwise, a linear bottleneck is used.
    :param list[int] filters: List of integers defining the number of filters in each layer of the encoder/decoder.
    :param int | None img_size: Size of the input images. Required if `fully_conv_bottleneck=False`.
    :param bool use_batchnorm: If `True`, uses batch normalization in the encoder/decoder.
    :param str | None pretrained: Path to pretrained weights or "download" to download pretrained weights.
    :param float gamma: Initial value for the standard deviation of :math:`p_\theta(x|z)` for mean decoder training.
    :param bool learn_gamma: If `True`, the parameter :math:`\gamma` is learnable.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        decode_mean_only: bool = True,
        fully_conv_bottleneck: bool = True,
        filters: tuple[int] = (64, 64, 64),
        img_size: int | None = None,
        use_batchnorm: bool = True,
        pretrained: str = "download",
        gamma: float = 0.1,
        learn_gamma: bool = True,
        dim: str | int = 2,
    ):
        super().__init__()

        dim = fix_dim(dim)

        self.encoder = Encoder(
            in_channels,
            2 * latent_dim,
            filters,
            fully_conv_bottleneck,
            use_batchnorm,
            img_size,
            dim=dim,
        )
        self.decoder = Decoder(
            in_channels,
            latent_dim,
            filters,
            fully_conv_bottleneck,
            use_batchnorm,
            img_size,
            gamma=gamma,
            learn_gamma=learn_gamma,
            dim=dim,
        )
        if decode_mean_only:
            self.variance_decoder = None
        else:
            self.variance_decoder = Decoder(
                in_channels,
                latent_dim,
                filters,
                fully_conv_bottleneck,
                use_batchnorm,
                img_size,
            )

        if pretrained == "download":
            if (
                in_channels == 3
                and latent_dim == 64
                and fully_conv_bottleneck
                and use_batchnorm
                and dim == 2
            ):
                url = "https://huggingface.co/MaudBqrd/VBLEModels/resolve/main/deepinv_vae_M-64_celeba_std-diagonal.pth.tar?download=true"
                state_dict = torch.hub.load_state_dict_from_url(
                    url, map_location="cpu"
                )["state_dict"]
            elif (
                in_channels == 1
                and latent_dim == 64
                and fully_conv_bottleneck
                and use_batchnorm
                and dim == 2
            ):
                url = "https://huggingface.co/MaudBqrd/VBLEModels/resolve/main/deepinv_vae_M-64_celeba-wb_std-diagonal.pth.tar?download=true"
                state_dict = torch.hub.load_state_dict_from_url(
                    url, map_location="cpu"
                )["state_dict"]
            else:
                raise ValueError(
                    "No pretrained model available for the given configuration."
                )

        elif pretrained is not None:
            state_dict = torch.load(pretrained, map_location="cpu")["state_dict"]
        if pretrained is not None:
            self.load_state_dict(state_dict, strict=True)

    def encode(self, x: torch.Tensor) -> dict:
        r"""
        Encodes input images and returns the parameters of the approximate posterior distribution :math:`q_\phi(z|x)`.

        :param torch.Tensor x: Input images of shape `[B, C, H, W]`.

        :returns: (dict) Dictionary containing the mean `mu_q_z` and log-standard deviation `logsd_q_z` of the approximate posterior.
        """
        mu_q_z, mu_logsd_q_z = torch.chunk(self.encoder(x), 2, dim=1)
        return {"mu_q_z": mu_q_z, "logsd_q_z": mu_logsd_q_z}

    def decode(self, latent_dict: dict, decode_mean_only: bool = False) -> dict:
        r"""
        Decodes latent variables to reconstruct images.

        :param dict latent_dict: Dictionary containing the latent variable `z`.
        :param bool decode_mean_only: If `True`, only the mean decoder is used.

        :returns: (dict) Dictionary containing the reconstructed image `x_rec` and, if applicable, the standard deviation `x_rec_std`.
        """
        x_rec = self.decoder(latent_dict["z"])
        out_dict = {"x_rec": x_rec}
        if self.variance_decoder is not None and not decode_mean_only:
            x_rec_std = torch.exp(self.variance_decoder(latent_dict["z"]))
            out_dict["x_rec_std"] = x_rec_std
        return out_dict

    def posterior_sample(
        self, mu_z: torch.Tensor, logsd_z: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Samples from the approximate posterior distribution :math:`q_\phi(z|x)` using the
        reparameterization trick.

        :param torch.Tensor mu_z: Mean of the approximate posterior.
        :param torch.Tensor logsd_z: Log-standard deviation of the approximate posterior.

        :returns: (torch.Tensor) Sampled latent variable `z`.
        """
        eps = torch.randn_like(mu_z)
        z = mu_z + torch.exp(logsd_z) * eps
        return z

    def forward(self, x: torch.Tensor, decode_mean_only: bool = False) -> dict:
        r"""
        Forward pass through the VAE model: encodes the input, samples from the approximate posterior
        and decodes to reconstruct the image.

        :param torch.Tensor x: Input images of shape `[B, C, H, W]`.
        :param bool decode_mean_only: If `True`, only the mean decoder is used.

        :returns: (dict) Dictionary containing the reconstructed image `x_rec`, and latent variables.
        """
        latent_dict = self.encode(x)
        latent_dict["z"] = self.posterior_sample(
            latent_dict["mu_q_z"], latent_dict["logsd_q_z"]
        )
        out_dict = self.decode(latent_dict, decode_mean_only)
        out_dict.update(latent_dict)
        return out_dict

    def get_gamma(self):
        r"""
        Returns the value of :math:`\gamma`, the standard deviation of :math:`p_\theta(x|z)`
        when using only the mean decoder.
        """
        return self.decoder.gamma_x

    def compute_kl_loss(self, out_dict: dict) -> torch.Tensor:
        r"""
        Computes the KL divergence loss, :math:`KL(q(z)\|p(z))` assuming both are diagonal Gaussians.

        :param dict out_dict: Dictionary containing the outputs :math:`q(z)` and :math:`p(z)` parameters.

        :returns: (torch.Tensor) Computed KL divergence loss.
        """
        num_pixels = torch.prod(torch.tensor(out_dict["x_rec"].shape[2:])).item()

        BS = out_dict["mu_q_z"].shape[0]

        if "sd_q_z" not in out_dict:
            out_dict["sd_q_z"] = torch.exp(out_dict["logsd_q_z"])
        elif "logsd_q_z" not in out_dict:
            out_dict["logsd_q_z"] = out_dict["sd_q_z"].log()
        else:
            raise ValueError("Either sd_q_z or logsd_q_z must be in out_dict")

        kl_loss = (
            0.5
            * torch.sum(
                out_dict["mu_q_z"].pow(2)
                + out_dict["sd_q_z"].pow(2)
                - 1
                - 2 * out_dict["logsd_q_z"]
            )
            / (num_pixels * BS)
        )

        return kl_loss


class Encoder(nn.Module):
    r"""
    Encoder module for the VAE model.

    :param int in_channels: Number of input channels.
    :param int latent_dim: Dimensionality of the latent space.
    :param list[int] filters: List of integers defining the number of filters in each layer of the encoder.
    :param bool fully_conv_bottleneck: If `True`, uses a fully convolutional bottleneck.
        Otherwise, a linear bottleneck is used.
    :param bool use_batchnorm: If `True`, uses batch normalization in the encoder.
    :param int | None img_size: Size of the input images. Required if `fully_conv_bottleneck=False`.
    :param callable activation: Activation function to use in the encoder.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        filters: tuple[int] = (64, 64, 64, 64),
        fully_conv_bottleneck: bool = False,
        use_batchnorm: bool = False,
        img_size: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.LeakyReLU,
        dim: int = 2,
    ):
        super().__init__()

        self.fully_conv_bottleneck = fully_conv_bottleneck

        conv = conv_nd(dim)
        batchnorm = batchnorm_nd(dim)

        self.layers = []
        for out_channels in filters:
            self.layers.append(
                conv(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
            )
            if use_batchnorm:
                self.layers.append(batchnorm(out_channels))
            self.layers.append(activation())
            in_channels = out_channels
        self.layers = nn.Sequential(*self.layers)

        if fully_conv_bottleneck:  # fully convolutionnal bottleneck
            self.bottleneck = conv(
                filters[-1],
                latent_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            )
        else:  # linear bottleneck
            if img_size is None:
                raise ValueError("img_size must be specified for linear bottleneck")
            num_downsamples = len(filters)
            reduced_size = img_size // (2**num_downsamples)
            self.bottleneck = nn.Linear(
                filters[-1] * reduced_size * reduced_size,
                latent_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.layers(x)
        if self.fully_conv_bottleneck:
            z = self.bottleneck(z)
        else:
            z = torch.flatten(z, start_dim=1)
            z = self.bottleneck(z)
        return z


class Decoder(nn.Module):
    r"""
    Decoder module for the VAE model.

    :param int in_channels: Number of output channels.
    :param int latent_dim: Dimensionality of the latent space.
    :param list[int] filters: List of integers defining the number of filters in each layer of the decoder.
    :param bool fully_conv_bottleneck: If `True`, uses a fully convolutional bottleneck.
        Otherwise, a linear bottleneck is used.
    :param bool use_batchnorm: If `True`, uses batch normalization in the decoder.
    :param int | None img_size: Size of the output images. Required if `fully_conv_bottleneck=False`.
    :param callable activation: Activation function to use in the decoder.
    :param callable | None last_activation: Activation function to use at the output of the decoder.
    :param float gamma: Initial value for the standard deviation of :math:`p_\theta(x|z)` for mean decoder training.
    :param bool learn_gamma: If `True`, the parameter :math:`\gamma` is learnable.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        filters: tuple[int] = (64, 64, 64, 64),
        fully_conv_bottleneck: bool = False,
        use_batchnorm: bool = False,
        img_size: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.LeakyReLU,
        last_activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        gamma: float = 0.1,
        learn_gamma: bool = False,
        dim: int = 2,
    ):
        super().__init__()

        self.fully_conv_bottleneck = fully_conv_bottleneck
        self.learn_gamma = learn_gamma

        conv_transpose = conv_transpose_nd(dim)
        batchnorm = batchnorm_nd(dim)

        if fully_conv_bottleneck:  # fully convolutionnal bottleneck
            self.first_layer = [
                conv_transpose(
                    latent_dim,
                    filters[-1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                )
            ]
        else:  # linear bottleneck
            if img_size is None:
                raise ValueError("img_size must be specified for linear bottleneck")
            num_upsamples = len(filters)
            reduced_size = img_size // (2**num_upsamples)
            self.first_layer = [
                nn.Linear(
                    latent_dim,
                    filters[-1] * reduced_size * reduced_size,
                )
            ]
            self.reshape_size = (filters[-1], reduced_size, reduced_size)
        if use_batchnorm:
            self.first_layer.append(batchnorm(filters[-1]))
        self.first_layer.append(activation())
        self.first_layer = nn.Sequential(*self.first_layer)

        self.layers = []

        in_ch = filters[-1]
        for out_channels in reversed(filters[:-1]):
            self.layers.append(
                conv_transpose(
                    in_ch,
                    out_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                )
            )
            if use_batchnorm:
                self.layers.append(batchnorm(out_channels))
            self.layers.append(activation())
            in_ch = out_channels
        self.layers.append(
            conv_transpose(
                in_ch, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1
            )
        )
        if last_activation is not None:
            self.layers.append(last_activation())
        self.layers = nn.Sequential(*self.layers)

        self.gamma_x = nn.Parameter(
            float(gamma) * torch.ones(1), requires_grad=self.learn_gamma
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.fully_conv_bottleneck:
            x = self.first_layer(z)
        else:
            x = self.first_layer(z)
            x = x.view(-1, *self.reshape_size)
        x = self.layers(x)
        return x
