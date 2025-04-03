import torch
import torch.nn as nn
import numpy as np
from deepinv.loss.mc import MCLoss
from tqdm import tqdm
from .base import Reconstructor


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class ConvDecoder(nn.Module):
    r"""
    Convolutional decoder network.

    The architecture was introduced in `"Accelerated MRI with Un-trained Neural Networks" <https://arxiv.org/abs/2007.02471>`_,
    and it is well suited as a deep image prior (see :class:`deepinv.models.DeepImagePrior`).


    :param tuple img_shape: shape of the output image.
    :param tuple in_size: size of the input vector.
    :param int layers: number of layers in the network.
    :param int channels: number of channels in the network.
    """

    #  Code adapted from https://github.com/MLI-lab/ConvDecoder/tree/master by Darestani and Heckel.
    def __init__(self, img_shape, in_size=(4, 4), layers=7, channels=256):
        super(ConvDecoder, self).__init__()

        out_size = img_shape[1:]
        output_channels = img_shape[0]

        # parameter setup
        kernel_size = 3
        strides = [1] * (layers - 1)

        # compute up-sampling factor from one layer to another
        scale_x, scale_y = (
            (out_size[0] / in_size[0]) ** (1.0 / (layers - 1)),
            (out_size[1] / in_size[1]) ** (1.0 / (layers - 1)),
        )
        hidden_size = [
            (
                int(np.ceil(scale_x**n * in_size[0])),
                int(np.ceil(scale_y**n * in_size[1])),
            )
            for n in range(1, (layers - 1))
        ] + [out_size]

        # hidden layers
        self.net = nn.Sequential()
        for i in range(layers - 1):
            self.net.add(nn.Upsample(size=hidden_size[i], mode="nearest"))
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                strides[i],
                padding=(kernel_size - 1) // 2,
                bias=True,
            )
            self.net.add(conv)
            self.net.add(nn.ReLU())
            self.net.add(nn.BatchNorm2d(channels, affine=True))
        # final layer
        self.net.add(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                strides[i],
                padding=(kernel_size - 1) // 2,
                bias=True,
            )
        )
        self.net.add(nn.ReLU())
        self.net.add(nn.BatchNorm2d(channels, affine=True))
        self.net.add(nn.Conv2d(channels, output_channels, 1, 1, padding=0, bias=True))

    def forward(self, x, scale_out=1):
        r"""
        Forward pass through the ConvDecoder network.

        :param torch.Tensor x: Input tensor.
        :param float scale_out: Output scaling factor.
        """
        return self.net(x) * scale_out


class DeepImagePrior(Reconstructor):
    r"""

    Deep Image Prior reconstruction.

    This method is based on the paper `"Deep Image Prior" by Ulyanov et al. (2018)
    <https://arxiv.org/abs/1711.10925>`_, and reconstructs
    an image by minimizing the loss function

    .. math::

        \min_{\theta}  \|y-AG_{\theta}(z)\|^2

    where :math:`z` is a random input and :math:`G_{\theta}` is a convolutional decoder network with parameters
    :math:`\theta`. The minimization should be stopped early to avoid overfitting. The method uses the Adam
    optimizer.

    .. note::

        This method only works with certain convolutional decoder networks. We recommend using the
        network :class:`deepinv.models.ConvDecoder`.


    .. note::

        The number of iterations and learning rate are set to the values used in the original paper. However, these
        values may not be optimal for all problems. We recommend experimenting with different values.

    :param torch.nn.Module generator: Convolutional decoder network.
    :param list, tuple input_size: Size `(C,H,W)` of the input noise vector :math:`z`.
    :param int iterations: Number of optimization iterations.
    :param float learning_rate: Learning rate of the Adam optimizer.
    :param bool verbose: If ``True``, print progress.
    :param bool re_init: If ``True``, re-initialize the network parameters before each reconstruction.

    """

    def __init__(
        self,
        generator,
        input_size,
        iterations=2500,
        learning_rate=1e-2,
        verbose=False,
        re_init=False,
    ):
        super().__init__()
        self.generator = generator
        self.max_iter = int(iterations)
        self.lr = learning_rate
        self.loss = MCLoss()
        self.verbose = verbose
        self.re_init = re_init
        self.input_size = input_size

    def forward(self, y, physics, **kwargs):
        r"""
        Reconstruct an image from the measurement :math:`y`. The reconstruction is performed by solving a minimiza
        problem.

        .. warning::

            The optimization is run for every test batch. Thus, this method can be slow when tested on a large
            number of test batches.

        :param torch.Tensor y: Measurement.
        :param torch.Tensor physics: Physics model.
        """
        if self.re_init:
            for layer in self.generator.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        self.generator.requires_grad_(True)
        z = torch.randn(self.input_size, device=y.device).unsqueeze(0)
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr)

        for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
            x = self.generator(z)
            error = self.loss(y, x, physics)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        return self.generator(z)


# test code
# if __name__ == "__main__":
#     device = "cuda:0"
#     import torchvision
#     import deepinv as dinv
#
#     device = dinv.utils.get_freer_gpu()
#
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(device) / 255
#     x = torchvision.transforms.Resize((128, 128))(x)
#
#     physics = dinv.physics.Inpainting(
#         tensor_size=x.shape[1:],
#         device=device,
#         noise_model=dinv.physics.GaussianNoise(sigma=0.05),
#     )
#
#     y = physics(x)
#
#     iterations = 1000
#     lr = 1e-2
#     channels = 256
#     in_size = [8, 8]
#     backbone = ConvDecoder(
#         img_shape=x.shape[1:], in_size=in_size, channels=channels
#     ).to(device)
#
#     model = DeepImagePrior(
#         backbone,
#         learning_rate=lr,
#         re_init=True,
#         iterations=iterations,
#         verbose=True,
#         input_size=[channels] + in_size,
#     ).to(device)
#
#     x_hat = model(y, physics)
#
#     dinv.utils.plot([x, y, x_hat], titles=["GT", "Meas.", "Recon."])
