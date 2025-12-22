from __future__ import annotations
from typing import Optional
from math import prod

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.nn import functional as F
from torch import rand

from deepinv.physics.forward import Physics
from deepinv.loss.mc import MCLoss
from deepinv.models.base import Reconstructor
from deepinv.models.utils import fix_dim, conv_nd, batchnorm_nd, conv_transpose_nd
from deepinv.utils.decorators import _deprecated_alias


class PatchGANDiscriminator(nn.Module):
    r"""PatchGAN Discriminator model.

    This discriminator model was originally proposed by :footcite:t:`isola2017image` and classifies whether each patch of an image is real
    or fake.

    Implementation adapted from :footcite:t:`kupyn2018deblurgan`.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for how to use this for adversarial training.

    :param int input_nc: number of input channels, defaults to 3
    :param int ndf: hidden layer size, defaults to 64
    :param int n_layers: number of hidden conv layers, defaults to 3
    :param bool use_sigmoid: use sigmoid activation at end, defaults to False
    :param bool batch_norm: whether to use batch norm layers, defaults to True
    :param bool bias: whether to use bias in conv layers, defaults to True
    :param bool original: use exact network from original paper. If `False`, modify network
        to reduce spatial dims further.
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
        batch_norm: bool = True,
        bias: bool = True,
        original: bool = True,
        dim: str | int = 2,
    ):
        super().__init__()

        dim = fix_dim(dim)

        conv = conv_nd(dim)

        kw = 4  # kernel width
        padw = int(np.ceil((kw - 1) / 2))
        padw -= 1 if not original else 0
        sequence = [
            conv(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                conv(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=bias,
                ),
                batchnorm_nd(dim)(ndf * nf_mult) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]

        if original:
            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [
                conv(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=bias,
                ),
                batchnorm_nd(dim)(ndf * nf_mult) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]

        sequence += [conv(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        return self.model(x)


class ESRGANDiscriminator(nn.Module):
    r"""ESRGAN Discriminator.

    The ESRGAN discriminator model was originally proposed by :footcite:t:`wang2018esrgan`. Implementation taken from
    https://github.com/edongdongchen/EI/blob/main/models/discriminator.py.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for how to use this for adversarial training.

    :param tuple img_size: shape of input image
    :param bool batch_norm: whether to have batchnorm layers.
    :param tuple filter: Width (number of filters) at each stage. This can also be used to control the number of stages (or also: the output shape relative to input shapes). Defaults to (64, 128, 256, 512)
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)
    """

    @_deprecated_alias(input_shape="img_size")
    def __init__(
        self, img_size: tuple, batch_norm: bool = True, filters: tuple = (64, 128, 256, 512), dim: str | int = 2
    ):
        super().__init__()

        dim = fix_dim(dim)
        conv = conv_nd(dim)
        batchnorm = batchnorm_nd(dim)

        self.img_size = img_size
        in_channels, *spatials = self.img_size
        patch_spatials = tuple(s // 2 ** len(filters) for s in spatials)
        self.output_shape = (1, *patch_spatials)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(
                conv(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
            )
            if not first_block and batch_norm:
                layers.append(batchnorm(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                conv(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
            )
            if batch_norm:
                layers.append(batchnorm(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate(filters):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters

        layers.append(conv(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        return self.model(x)


class DCGANDiscriminator(nn.Module):
    r"""DCGAN Discriminator.

    The DCGAN discriminator model was originally proposed by :footcite:t:`radford2015unsupervised`. Implementation taken from
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for how to use this for adversarial training.

    :param int ndf: hidden layer size, defaults to 64
    :param int nc: number of input channels, defaults to 3
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)

    """

    def __init__(self, ndf: int = 64, nc: int = 3, dim: str | int = 2):
        super().__init__()
        dim = fix_dim(dim)
        conv = conv_nd(dim)
        batchnorm = batchnorm_nd(dim)
        self.model = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            conv(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            conv(ndf, ndf * 2, 4, 2, 1, bias=False),
            batchnorm(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            conv(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            batchnorm(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            conv(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            batchnorm(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            conv(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        return self.model(x)


class DCGANGenerator(nn.Module):
    r"""DCGAN Generator.

    The DCGAN generator model was originally proposed by :footcite:t:`radford2015unsupervised`
    and takes a latent sample as input.

    Implementation taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for how to use this for adversarial training.

    :param int output_size: desired square size of output image. Choose from 64 or 128, defaults to 64
    :param int nz: latent dimension, defaults to 100
    :param int ngf: hidden layer size, defaults to 64
    :param int nc: number of image output channels, defaults to 3
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)
    """

    def __init__(
        self,
        output_size: int = 64,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 3,
        dim: str | int = 2,
    ):
        super().__init__()
        dim = fix_dim(dim)
        batchnorm = batchnorm_nd(dim)
        convtranspose = conv_transpose_nd(dim)
        self.nz = nz
        # input is (b, nz, 1, 1), output is (b, nc, output_size, output_size)
        if output_size == 64:
            layers = [
                convtranspose(nz, ngf * 8, 4, 1, 0, bias=False),
                batchnorm(ngf * 8),
                nn.ReLU(True),
            ]
        elif output_size == 128:
            layers = [
                convtranspose(nz, ngf * 16, 4, 1, 0, bias=False),
                batchnorm(ngf * 16),
                nn.ReLU(True),
                convtranspose(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
                batchnorm(ngf * 8),
                nn.ReLU(True),
            ]
        else:
            raise ValueError("output_size must be 64 or 128.")

        layers += [
            convtranspose(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            batchnorm(ngf * 4),
            nn.ReLU(True),
            convtranspose(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            batchnorm(ngf * 2),
            nn.ReLU(True),
            convtranspose(ngf * 2, ngf, 4, 2, 1, bias=False),
            batchnorm(ngf),
            nn.ReLU(True),
            convtranspose(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, z: Tensor, *args, **kwargs) -> Tensor:
        r"""
        Generate an image

        :param torch.Tensor z: latent vector
        """
        return self.model(z, *args, **kwargs)


class CSGMGenerator(Reconstructor):
    r"""CSGMGenerator(backbone_generator=DCGANGenerator(), inf_max_iter=2500, inf_tol=1e-4, inf_lr=1e-2, inf_progress_bar=False)
    Adapts a generator model backbone (e.g DCGAN) for CSGM or AmbientGAN.

    This approach was proposed by :footcite:t:`bora2017compressed` and :footcite:t:`bora2018ambientgan`.

    At train time, the generator samples latent vector from Unif[-1, 1] and passes through backbone.

    At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input
    measurements y, then outputs the corresponding reconstruction.

    This generator can be overridden for more advanced optimisation algorithms by overriding ``optimize_z``.

    See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for how to use this for adversarial training.

    .. note::

        At train time, this generator discards the measurements ``y``, but these measurements are used at test time.
        This means that train PSNR will be meaningless but test PSNR will be correct.


    :param torch.nn.Module backbone_generator: any neural network that maps a latent vector of length ``nz`` to an image, must have ``nz`` attribute. Defaults to DCGANGenerator()
    :param int inf_max_iter: maximum iterations at inference-time optimisation, defaults to 2500
    :param float inf_tol: tolerance of inference-time optimisation, defaults to 1e-2
    :param float inf_lr: learning rate of inference-time optimisation, defaults to 1e-2
    :param bool inf_progress_bar: whether to display progress bar for inference-time optimisation, defaults to False
    """

    def __init__(
        self,
        backbone_generator: nn.Module | None = None,
        inf_max_iter: int = 2500,
        inf_tol: float = 1e-4,
        inf_lr: float = 1e-2,
        inf_progress_bar: bool = False,
    ):
        if backbone_generator is None:
            backbone_generator = DCGANGenerator()
        super().__init__()
        self.backbone_generator = backbone_generator
        self.inf_loss = MCLoss()
        self.inf_max_iter = inf_max_iter
        self.inf_tol = inf_tol
        self.inf_lr = inf_lr
        self.inf_progress_bar = inf_progress_bar

    def random_latent(self, device, requires_grad=True):
        r"""Generate a latent sample to feed into generative model.

        The model must have an attribute `nz` which is the latent dimension.

        :param torch.device device: torch device
        :param bool requires_grad: whether to require gradient, defaults to True.
        """
        return (
            torch.rand(
                1,
                self.backbone_generator.nz,
                1,
                1,
                device=device,
                requires_grad=requires_grad,
            )
            * 2
            - 1
        )

    @torch.enable_grad()
    def optimize_z(self, z: Tensor, y: Tensor, physics: Physics):
        r"""Run inference-time optimisation of latent z that is consistent with input measurement y according to physics.

        The optimisation is defined with simple stopping criteria. Override this function for more advanced optimisation.

        :param torch.Tensor z: initial latent variable guess
        :param torch.Tensor y: measurement with which to compare reconstructed image
        :param Physics physics: forward model
        :return: optimized latent z
        """
        z = nn.Parameter(z, requires_grad=True)
        optimizer = Adam([z], lr=self.inf_lr)
        err_prev = 999

        pbar = tqdm(range(self.inf_max_iter), disable=(not self.inf_progress_bar))
        for i in pbar:
            with torch.enable_grad():
                x_hat = self.backbone_generator(z)
                error = self.inf_loss(y=y, x_net=x_hat, physics=physics)

            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            err_curr = error.item()
            err_perc = abs(err_curr - err_prev) / err_curr
            err_prev = err_curr
            pbar.set_postfix({"err_curr": err_curr, "err_perc": err_perc})

            if err_perc < self.inf_tol:
                break
        return z

    def forward(self, y: Tensor, physics: Physics, *args, **kwargs) -> Tensor:
        r"""Forward pass of generator model.

        At train time, the generator samples latent vector from Unif[-1, 1] and passes through backbone.

        At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input
        measurements y, then outputs the corresponding reconstruction.

        :param y: measurement to reconstruct
        :param deepinv.physics.Physics physics: forward model
        """
        z = self.random_latent(y.device)

        if not self.training:
            z = self.optimize_z(z, y, physics)

        return self.backbone_generator(z)


class SkipConvDiscriminator(nn.Module):
    """Simple residual convolution discriminator architecture.

    Architecture taken from `Fast Unsupervised MRI Reconstruction Without Fully-Sampled Ground Truth Data Using Generative Adversarial Networks <https://openaccess.thecvf.com/content/ICCV2021W/LCI/html/Cole_Fast_Unsupervised_MRI_Reconstruction_Without_Fully-Sampled_Ground_Truth_Data_Using_ICCVW_2021_paper.html>`_.

    Consists of convolutional blocks with skip connections with a final dense layer followed by sigmoid.
    It receives an image as input and outputs a scalar value (between 0 and 1 if sigmoid is used).

    :param tuple img_size: tuple of ints of input image size
    :param int d_dim: hidden dimension
    :param int d_blocks: number of conv blocks
    :param int in_channels: number of input channels
    :param bool use_sigmoid: use sigmoid activation at output.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (320, 320),
        d_dim: int = 128,
        d_blocks: int = 4,
        in_channels: int = 2,
        use_sigmoid: bool = True,
    ):
        super().__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(),
            )

        self.initial_conv = conv_block(in_channels, d_dim)

        self.blocks = nn.ModuleList()
        for _ in range(d_blocks):
            self.blocks.append(conv_block(d_dim, d_dim))
            self.blocks.append(conv_block(d_dim, d_dim))

        self.flatten = nn.Flatten()
        self.final = nn.Linear(d_dim * prod(img_size), 1)
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of discriminator model.

        :param torch.Tensor x: input image
        """
        x = self.initial_conv(x)

        for i in range(0, len(self.blocks), 2):
            x1 = self.blocks[i](x)
            x2 = x1 + self.blocks[i + 1](x)
            x = x2

        y = self.final(self.flatten(x))
        return self.sigmoid(y).squeeze() if self.use_sigmoid else y.squeeze()


class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral normalization.

    Discriminator proposed in Real-ESRGAN :footcite:t:`wang2021realesrgan` for superresolution problems.

    Implementation and pretrained weights taken from https://github.com/xinntao/Real-ESRGAN.

    :param int num_in_ch: Channel number of inputs. Default: 3.
    :param int num_feat: Channel number of base intermediate features. Default: 64.
    :param bool skip_connection: Whether to use skip connections between U-Net. Default: `True`.
    :param int, None pretrained_factor: if not `None`, loads pretrained weights with given factor, must be `2` or `4`. Default: `None`.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_feat=64,
        skip_connection=True,
        pretrained_factor: Optional[int] = None,
        device="cpu",
    ):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        if pretrained_factor is not None:  # pragma: no cover
            if pretrained_factor == 2:
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus_netD.pth"
            elif pretrained_factor == 4:
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth"
            else:
                raise ValueError(
                    f"Unsupported pretrained_factor={pretrained_factor}. Use 2 or 4."
                )

            state_dict = torch.hub.load_state_dict_from_url(
                url, map_location=device, weights_only=True
            )
            self.load_state_dict(state_dict["params"], strict=True)

        self.to(device)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
