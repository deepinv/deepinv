import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch import Tensor
from torch import rand
from torch.optim import Adam
from deepinv.physics import Physics
from deepinv.loss import MCLoss


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator model originally from pix2pix: Isola et al., Image-to-Image Translation with Conditional Adversarial Networks https://arxiv.org/abs/1611.07004.

    Implementation taken from Kupyn et al., DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf

    See ``deepinv.examples.adversarial_learning`` for how to use this for adversarial training.

    :param int input_nc: number of input channels, defaults to 3
    :param int ndf: hidden layer size, defaults to 64
    :param int n_layers: number of hidden conv layers, defaults to 3
    :param bool use_sigmoid: use sigmoid activation at end, defaults to False
    :param bool batch_norm: whether to use batch norm layers, defaults to True
    :param bool bias: whether to use bias in conv layers, defaults to True
    """

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
        batch_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        kw = 4  # kernel width
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=bias,
                ),
                nn.BatchNorm2d(ndf * nf_mult) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=bias,
            ),
            nn.BatchNorm2d(ndf * nf_mult) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ESRGANDiscriminator(nn.Module):
    """ESRGAN Discriminator originally from Wang et al., ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks https://arxiv.org/abs/1809.00219

    Implementation taken from https://github.com/edongdongchen/EI/blob/main/models/discriminator.py

    See ``deepinv.examples.adversarial_learning`` for how to use this for adversarial training.

    :param tuple input_shape: shape of input image
    """

    def __init__(self, input_shape: tuple):
        super().__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2**4), int(in_width / 2**4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
            )
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(
                discriminator_block(in_filters, out_filters, first_block=(i == 0))
            )
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator from Radford et al. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks https://arxiv.org/abs/1511.06434
    Implementation taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    See ``deepinv.examples.adversarial_learning`` for how to use this for adversarial training.

    :param int ndf: hidden layer size, defaults to 64
    :param int nc: number of input channels, defaults to 3
    """

    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input)


class DCGANGenerator(nn.Module):
    """DCGAN Generator from Radford et al. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks https://arxiv.org/abs/1511.06434

    Unconditional generator model which takes latent samples as input.

    Implementation taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    See ``deepinv.examples.adversarial_learning`` for how to use this for adversarial training.

    :param int nz: latent dimension, defaults to 100
    :param int ngf: hidden layer size, defaults to 64
    :param int nc: number of input channels, defaults to 3
    """

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input, *args, **kwargs):
        return self.model(input, *args, **kwargs)


class CSGMGenerator(nn.Module):
    """
    Adapts a generator model backbone (e.g DCGAN) for CSGM or AmbientGAN:

    Bora et al., "Compressed Sensing using Generative Models", "AmbientGAN: Generative models from lossy measurements"

    At train time, this samples latent vector from Unif[-1, 1] and passes through backbone. Note this generator discards the input.

    At test time, CSGM/AmbientGAN runs an optimisation to find the best latent vector that fits the input y, then outputs the corresponding reconstruction. Note this means that test PSNR will be correct but train PSNR will be meaningless.

    This generator can be overridden for more advanced optimisation algorithms by overriding ``optimize_z``.

    See ``deepinv.examples.adversarial_learning`` for how to use this for adversarial training.

    :param nn.Module backbone_generator: any neural network that maps a latent vector of length ``nz`` to an image, must have ``nz`` attribute. Defaults to DCGANGenerator()
    :param int inf_max_iter: maximum iterations at inference-time optimisation, defaults to 2500
    :param float inf_tol: tolerance of inference-time optimisation, defaults to 1e-2
    :param float inf_lr: learning rate of inference-time optimisation, defaults to 1e-2
    :param bool inf_progress_bar: whether to display progress bar for inference-time optimisation, defaults to False
    """

    def __init__(
        self,
        backbone_generator: nn.Module = DCGANGenerator(),
        inf_max_iter: int = 2500,
        inf_tol: float = 1e-4,
        inf_lr: float = 1e-2,
        inf_progress_bar: bool = False,
    ):
        super().__init__()
        self.backbone_generator = backbone_generator
        self.inf_loss = MCLoss()
        self.inf_max_iter = inf_max_iter
        self.inf_tol = inf_tol
        self.inf_lr = inf_lr
        self.inf_progress_bar = inf_progress_bar

    def random_latent(self, device, requires_grad=True) -> Tensor:
        """Generate a latent sample to feed into generative model. The model must have an attribute `nz` which is the latent dimension."""
        return (
            rand(
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

    def optimize_z(self, z: Tensor, y: Tensor, physics: Physics) -> Tensor:
        """Run inference-time optimisation of latent z that is consistent with input measurement y according to physics.

        The optimisation is defined with simple stopping criteria. Override this function for more advanced optimisation.

        :param Tensor z: initial latent variable guess
        :param Tensor y: measurement with which to compare reconstructed image
        :param Physics physics: forward model
        :return Tensor: optimized z
        """
        z = nn.Parameter(z)
        optimizer = Adam([z], lr=self.inf_lr)
        err_prev = 999

        for i in (
            pbar := tqdm(range(self.inf_max_iter), disable=(not self.inf_progress_bar))
        ):
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
        z = self.random_latent(y.device)

        if not self.training:
            z = self.optimize_z(z, y, physics)

        return self.backbone_generator(z)
