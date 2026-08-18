# The following code is adapted from CompressAI library with the following license:
# ---------------------------------------------------------------

# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ---------------------------------------------------------------

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import scipy.stats
import numpy as np

from deepinv.models.base import Denoiser
from deepinv.models.utils import conv_nd, conv_transpose_nd


class JointAutoregressiveAndHyperpriorCAE(Denoiser):
    r"""
    Base class for Compressive Autoencoders with joint autoregressive and hyperprior
    architectures, introduced in  `"Joint Autoregressive and Hierarchical Priors for
    Learned Image Compression" <https://arxiv.org/abs/1808.02736>`_.

    This model implementation is only meant to be used as a regularizer for inverse problems,
    as the compression quantization procedure is not implemented.

    If ``decode_mean_only`` is ``False``, the model includes a second decoder to estimate
    the pixel-wise standard deviation of the reconstruction, useful for uncertainty estimation
    using the VBLE-xz framework introduced in `"Deep priors for satellite image restoration
    with accurate uncertainties" <https://arxiv.org/abs/2412.04130>`_.

    :param int in_channels: Number of channels in the input images. Default: 3.
    :param int latent_dim: Dimensionality of the latent representation. Default: 320.
    :param int hidden_dim: Number of channels in the hidden layers. Default: 192.
    :param bool decode_mean_only: If ``True``, only a single decoder is used. It estimates
        the mean of the reconstruction, with :math:`p_\theta(x|z) = N(x; D_\theta(z), \gamma^2 I)`
        and :math:`\gamma^2=\frac{1}{(2 \log 2) \alpha (2^{n_{bits}} - 1)^2}`. If ``False``,
        a second decoder is used to estimate the pixel-wise standard deviation of the reconstruction,
        with :math:`p_\theta(x|z) = N(x; D_\theta(z), \mathrm{diag}(\sigma^2_\theta(z)))`. Default: ``True``.
    :param str pretrained: If ``"download"``, downloads and loads the pretrained weights.
        If a valid path is provided, loads the weights from the given path. Default: ``"download"``.
    :param float alpha: Rate-distortion trade-off parameter used for training the mean decoder
        (the higher, the less compression, the less regularization). Default: 0.0483.
    :param int n_bits: Number of bits per pixel in the images.

    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 320,
        hidden_dim: int = 192,
        decode_mean_only: bool = True,
        pretrained: str = "download",
        alpha: float = 0.0483,
        n_bits: int = 8,
    ):
        super().__init__()

        self.N = int(hidden_dim)
        self.M = int(latent_dim)
        self.in_channels = in_channels
        gamma = (2 * np.log(2) * alpha * (2**n_bits - 1) ** 2) ** (-0.5)
        self.log_gamma = nn.Parameter(
            torch.log(torch.tensor([gamma])), requires_grad=False
        )

    def encode(self, x: torch.Tensor) -> dict:
        r"""
        Encodes the input image :math:`x` into the latent representation :math:`z` and
        hyperprior representation :math:`h` (=second latent variable).

        :param torch.Tensor x: Input image tensor of shape `[B, C, H, W]`.

        :returns: (dict) Dictionary containing:

            - **mu_q_z** (torch.Tensor): Mean of the posterior distribution of :math:`z`
            (with posterior :math:`q(z|x)=\mathcal{U}(\mu_q_z - 0.5, \mu_q_z + 0.5)`).

            - **mu_q_h** (torch.Tensor): Mean of the posterior distribution of :math:`h`
            (with posterior :math:`q(h|x)=\mathcal{U}(\mu_q_h - 0.5, \mu_q_h + 0.5)`).
        """
        x = x.expand(-1, 3, -1, -1)
        mu_z = self.g_a(x)
        mu_h = self.h_a(mu_z)
        out_dict = {"mu_q_z": mu_z, "mu_q_h": mu_h}
        return out_dict

    def decode(self, latent_dict: dict, decode_mean_only: bool = False) -> dict:
        r"""
        Decodes the latent representation :math:`z` into the reconstructed image :math:`\hat{x}`.
        Also computes the parameters of the prior distribution of :math:`z` given
        :math:`h` and the context model.

        :param dict latent_dict: Dictionary containing:
            - **z** (torch.Tensor): Latent representation tensor of shape `[B, M, H_z, W_z]`.

            - **h** (torch.Tensor): Hyperprior representation tensor of shape `[B, N, H_h, W_h]`.

        :param bool decode_mean_only: If ``True``, only the mean of the reconstruction is decoded.
            If ``False``, also decodes the pixel-wise standard deviation of the reconstruction.

        :returns: (dict) Updated `latent_dict` with the following additional entries:

            - **sd_p_z** (torch.Tensor): Standard deviation of the prior distribution of :math:`z`
                given :math:`h` and the context model, of shape `[B, M, H_z, W_z]`.

            - **mu_p_z** (torch.Tensor): Mean of the prior distribution of :math:`z` given :math:`h`
                and the context model, of shape `[B, M, H_z, W_z]`.

            - **x_rec** (torch.Tensor): Reconstructed image tensor of shape `[B, C, H, W]`.

            - **x_rec_std** (torch.Tensor): Pixel-wise standard deviation of the reconstruction,
                of shape `[B, C, H, W]`. Only present if ``decode_mean_only`` is ``False``.
        """
        params = self.h_s(latent_dict["h"])
        ctx_params = self.context_prediction(latent_dict["z"])
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        sd_p_z, mu_p_z = gaussian_params.chunk(2, 1)
        latent_dict.update({"sd_p_z": sd_p_z, "mu_p_z": mu_p_z})
        x_rec = self.g_s(latent_dict["z"])
        if self.in_channels == 1:
            x_rec = torch.mean(x_rec, dim=1, keepdim=True)
        latent_dict.update({"x_rec": x_rec})
        if self.variance_decoder is not None and not decode_mean_only:
            x_rec_std = self.variance_decoder(latent_dict["z"])
            latent_dict.update({"x_rec_std": F.sigmoid(x_rec_std) + 2e-4})
            if self.in_channels == 1:
                x_rec_std = torch.mean(x_rec_std, dim=1, keepdim=True)
        return latent_dict

    def posterior_sample(self, mu: torch.Tensor) -> torch.Tensor:
        r"""
        Samples from the posterior distribution :math:`q(z \text{ or } h|x)=\mathcal{U}(\mu - 0.5, \mu + 0.5)`.

        :param torch.Tensor mu: Mean of the posterior distribution, of shape `[B, M, H_{z \text{ or } h}, W_{z \text{ or } h}]`.

        :returns: (torch.Tensor) Sampled tensor from the posterior distribution
        """
        return mu + torch.rand_like(mu) - 0.5

    def forward(self, x: torch.Tensor, decode_mean_only: bool = False) -> dict:
        r"""
        Forward pass through the entire autoencoder: encodes the input image :math:`x`,
        samples from the posterior distributions of :math:`z` and :math:`h`, and
        decodes the sampled latent representations into the reconstructed image :math:`\hat{x}`.

        :param torch.Tensor x: Input image tensor of shape `[B, C, H, W]`.

        :param bool decode_mean_only: If ``True``, only the mean of the reconstruction is decoded.
            If ``False``, also decodes the pixel-wise standard deviation of the reconstruction.

        :returns: (dict) Dictionary containing all outputs from the encoding and decoding processes.
        """
        latent_dict = self.encode(x)

        latent_dict["z"] = self.posterior_sample(latent_dict["mu_q_z"])
        latent_dict["h"] = self.posterior_sample(latent_dict["mu_q_h"])

        out_dict = self.decode(latent_dict, decode_mean_only=decode_mean_only)
        out_dict.update(latent_dict)
        return out_dict

    def compute_bpp_loss(self, out_dict: dict) -> torch.Tensor:
        r"""
        Computes the bits-per-pixel (bpp) loss, defined as :math:`\mathcal{L}_{bpp} = \mathbb{E}_{z,h}[-\log_2 p_\theta(z|h) -\log_2 p(h)]`.

        :param dict out_dict: Dictionary containing the outputs from the forward pass.

        :returns: (torch.Tensor) Computed bpp loss.
        """
        BS, _, H, W = out_dict["x_rec"].shape
        num_pixels = H * W

        z_likelihoods = self.gaussian_conditional_likelihood_(
            out_dict["z"], out_dict["sd_p_z"], means=out_dict["mu_p_z"]
        )
        h_likelihoods = self.entropy_bottleneck_likelihood_(out_dict["h"])
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-num_pixels * BS * np.log(2)))
            for likelihoods in [z_likelihoods, h_likelihoods]
        )
        return bpp_loss

    def compute_kl_loss(self, out_dict: dict) -> torch.Tensor:
        r"""
        Computes the KL divergence loss, defined as :math:`\mathcal{L}_{KL} = \mathbb{E}_{z,h}[\log q(z,h) - \log p_\theta(z,h)]`.
        With :math:`q(z,h)=q_\phi(z|x)q_\phi(h|x)` the encoder posterior, the KL divergence is the BPP loss up to a factor :math:`\log(2)`.
        This is not the case anymore when using VBLE and a uniform approximate posterior on :math:`z` and :math:`h` with pixel-wise
        variances.

        :param dict out_dict: Dictionary containing the outputs from the forward pass.

        :returns: (torch.Tensor) Computed KL divergence loss.
        """
        BS, _, H, W = out_dict["x_rec"].shape
        num_pixels = H * W

        kl_loss = self.compute_bpp_loss(out_dict) * np.log(2)

        # For VBLE
        if "sd_q_z" in out_dict:
            log_az = torch.sum(torch.log(out_dict["sd_q_z"])) / (
                num_pixels * out_dict["sd_q_z"].shape[0]
            )
            kl_loss -= log_az
        if "sd_q_h" in out_dict:
            log_ah = torch.sum(torch.log(out_dict["sd_q_h"])) / (
                num_pixels * out_dict["sd_q_h"].shape[0]
            )
            kl_loss -= log_ah
        return kl_loss

    def get_gamma(self):
        r"""
        Returns the value of :math:`\gamma`, the standard deviation of :math:`p_\theta(x|z)`
        when using only the mean decoder.
        """
        return torch.exp(self.log_gamma)

    def gaussian_conditional_likelihood_(self, z, scales, means=None):
        r"""
        Auxiliary unction to compute the likelihood of z given the parameters means and scales
        of the prior model (:math:`p_\theta(z|h)=\mathcal{N}(z; \mu_p_z, \mathrm{diag}(\sigma^2_p_z)) * \mathcal{U}(-0.5, 0.5)`).

        :param torch.Tensor z: Latent representation tensor of shape `[B, M, H_z, W_z]`.
        :param torch.Tensor scales: Standard deviation tensor of the prior distribution  of :math:`z`
        :param torch.Tensor means: Mean tensor of the prior distribution of :math:`z`

        :returns: (torch.Tensor) Likelihood of `z`.
        """
        likelihood = self.gaussian_conditional._likelihood(z, scales, means)
        if self.gaussian_conditional.use_likelihood_bound:
            likelihood = self.gaussian_conditional.likelihood_lower_bound(likelihood)
        return likelihood

    def entropy_bottleneck_likelihood_(self, h):
        r"""
        Auxiliary function to compute the likelihood of h given a parametric model :math:`p_\psi(h)`.
        """
        perm = torch.cat(
            (
                torch.tensor([1, 0], dtype=torch.long, device=h.device),
                torch.arange(2, h.ndim, dtype=torch.long, device=h.device),
            )
        )
        inv_perm = perm
        h = h.permute(*perm).contiguous()
        shape = h.size()
        outputs = h.reshape(h.size(0), 1, -1)
        likelihood, _, _ = self.entropy_bottleneck._likelihood(outputs)

        if self.entropy_bottleneck.use_likelihood_bound:
            likelihood = self.entropy_bottleneck.likelihood_lower_bound(likelihood)

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return likelihood


class MbtCAE(JointAutoregressiveAndHyperpriorCAE):
    r"""
    Compressive autoencoder with joint autoregressive and a hyperprior, with
    the exact architecture used in `"Joint Autoregressive and Hierarchical Priors for
    Learned Image Compression" <https://arxiv.org/abs/1808.02736>`_.

    Several pretrained models are available for download. TODO: add links
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 320,
        hidden_dim: int = 192,
        decode_mean_only: bool = True,
        pretrained: str = None,
        alpha: float = 0.0483,
        n_bits: int = 8,
    ):
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            decode_mean_only=decode_mean_only,
            pretrained=pretrained,
            alpha=alpha,
            n_bits=n_bits,
        )
        dim = 2
        conv = conv_nd(dim)
        conv_transpose = conv_transpose_nd(dim)
        conv_down = lambda in_ch, out_ch: conv(
            in_ch, out_ch, kernel_size=5, stride=2, padding=2
        )
        conv_up = lambda in_ch, out_ch: conv_transpose(
            in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1
        )

        # analysis transform (encoder)
        self.g_a = nn.Sequential(
            conv_down(3, hidden_dim),
            GDN(hidden_dim),
            conv_down(hidden_dim, hidden_dim),
            GDN(hidden_dim),
            conv_down(hidden_dim, hidden_dim),
            GDN(hidden_dim),
            conv_down(hidden_dim, latent_dim),
        )

        # synthesis transform (decoder)
        self.g_s = nn.Sequential(
            conv_up(latent_dim, hidden_dim),
            GDN(hidden_dim, inverse=True),
            conv_up(hidden_dim, hidden_dim),
            GDN(hidden_dim, inverse=True),
            conv_up(hidden_dim, hidden_dim),
            GDN(hidden_dim, inverse=True),
            conv_up(hidden_dim, 3),
        )

        if decode_mean_only:
            self.variance_decoder = None
        else:
            # synthesis transform for variance decoder
            self.variance_decoder = nn.Sequential(
                conv_up(latent_dim, hidden_dim),
                GDN(hidden_dim, inverse=True),
                conv_up(hidden_dim, hidden_dim),
                GDN(hidden_dim, inverse=True),
                conv_up(hidden_dim, hidden_dim),
                GDN(hidden_dim, inverse=True),
                conv_up(hidden_dim, 3),
            )

        # analysis transform for hyperprior (hyper encoder)
        self.h_a = nn.Sequential(
            conv(latent_dim, hidden_dim, stride=1, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            conv_down(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            conv_down(hidden_dim, hidden_dim),
        )

        # synthesis transform for hyperprior (hyper decoder)
        self.h_s = nn.Sequential(
            conv_up(hidden_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            conv_up(latent_dim, latent_dim * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(
                latent_dim * 3 // 2, latent_dim * 2, stride=1, kernel_size=3, padding=1
            ),
        )

        # to merge hyperprior and context prediction
        self.entropy_parameters = nn.Sequential(
            conv(latent_dim * 12 // 3, latent_dim * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            conv(latent_dim * 10 // 3, latent_dim * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            conv(latent_dim * 8 // 3, latent_dim * 6 // 3, 1),
        )

        # auto-regressive context model
        self.context_prediction = MaskedConv2d(
            latent_dim, 2 * latent_dim, kernel_size=5, padding=2, stride=1
        )

        # entropy models (= prior models)
        # for z (first latent variable): p(z|h) = N(mu_p_z(h), sd_p_z(h))*U(-0.5, 0.5)
        self.gaussian_conditional = GaussianConditional()

        self.entropy_bottleneck = EntropyBottleneck(hidden_dim)

        if pretrained == "download":
            if in_channels == 3 and latent_dim == 320 and hidden_dim == 192:
                url = "https://huggingface.co/MaudBqrd/VBLEModels/resolve/main/deepinv_mbtcae_bsd_std-diagonal_alpha-483e-4.pth.tar?download=true"
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
            self.load_state_dict(state_dict, strict=not decode_mean_only)


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)


class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """

    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders"
    <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x: Tensor) -> Tensor:
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data = self.weight.data * self.mask
        return super().forward(x)


class GaussianConditional(nn.Module):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        scale_bound: float = 0.11,
    ):
        super().__init__()

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Tensor | None = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood


class EntropyBottleneck(nn.Module):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """

    def __init__(
        self,
        channels: int,
        likelihood_bound: float = 1e-9,
        init_scale: float = 10,
        filters: tuple[int, ...] = (3, 3, 3, 3),
    ):
        super().__init__()

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.matrices.append(nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.factors.append(nn.Parameter(factor))

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias

            if i < len(self.filters):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(
        self, inputs: Tensor, stop_gradient: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper
