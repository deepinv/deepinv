from deepinv.physics.noise import GaussianNoise
from deepinv.physics.forward import StackedLinearPhysics
from deepinv.physics.blur import Downsampling
from deepinv.physics.range import Decolorize
from deepinv.optim.utils import conjugate_gradient
from deepinv.utils.tensorlist import TensorList


class Pansharpen(StackedLinearPhysics):
    r"""
    Pansharpening forward operator.

    The measurements consist of a high resolution grayscale image and a low resolution RGB image, and
    are represented using :class:`deepinv.utils.TensorList`, where the first element is the RGB image and the second
    element is the grayscale image.

    By default, the downsampling is done with a gaussian filter with standard deviation equal to the downsampling,
    however, the user can provide a custom downsampling filter.

    It is possible to assign a different noise model to the RGB and grayscale images.

    :param tuple[int] img_size: size of the high-resolution multispectral input image, must be of shape (C, H, W).
    :param torch.Tensor, str, None filter: Downsampling filter. It can be 'gaussian', 'bilinear' or 'bicubic' or a
        custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param int factor: downsampling factor/ratio.
    :param str, tuple, list srf: spectral response function of the decolorize operator to produce grayscale from multispectral.
        See :class:`deepinv.physics.Decolorize` for parameter options. Defaults to ``flat`` i.e. simply average the bands.
    :param bool use_brovey: if ``True``, use the `Brovey method <https://ieeexplore.ieee.org/document/6998089>`_
        to compute the pansharpening, otherwise use the conjugate gradient method.
    :param torch.nn.Module noise_color: noise model for the RGB image.
    :param torch.nn.Module noise_gray: noise model for the grayscale image.
    :param torch.device, str device: torch device.
    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    |sep|

    :Examples:

        Pansharpen operator applied to a random 32x32 image:

        >>> from deepinv.physics import Pansharpen
        >>> import torch
        >>> x = torch.randn(1, 3, 32, 32) # Define random 32x32 color image
        >>> physics = Pansharpen(img_size=x.shape[1:], device=x.device)
        >>> x.shape
        torch.Size([1, 3, 32, 32])
        >>> y = physics(x)
        >>> y[0].shape
        torch.Size([1, 3, 8, 8])
        >>> y[1].shape
        torch.Size([1, 1, 32, 32])

    """

    def __init__(
        self,
        img_size,
        filter="bilinear",
        factor=4,
        srf="flat",
        noise_color=GaussianNoise(sigma=0.0),
        noise_gray=GaussianNoise(sigma=0.05),
        use_brovey=True,
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        assert len(img_size) == 3, "img_size must be of shape (C,H,W)"

        noise_color = noise_color if noise_color is not None else lambda x: x
        noise_gray = noise_gray if noise_gray is not None else lambda x: x
        self.use_brovey = use_brovey

        downsampling = Downsampling(
            img_size=img_size,
            factor=factor,
            filter=filter,
            noise_model=noise_color,
            device=device,
            padding=padding,
        )
        decolorize = Decolorize(
            srf=srf, noise_model=noise_gray, channels=img_size[0], device=device
        )

        super().__init__(physics_list=[downsampling, decolorize], **kwargs)

        # Set convenience attributes
        self.downsampling = downsampling
        self.decolorize = decolorize
        self.solver = "lsqr"  # more stable than CG

    def A_dagger(self, y: TensorList, **kwargs):
        """
        If the Brovey method is used, compute the classical Brovey solution, otherwise compute the conjugate gradient solution.

        See `review paper <https://ieeexplore.ieee.org/document/6998089>`_ for details.

        :param deepinv.utils.TensorList y: input tensorlist of (MS, PAN)
        :return: Tensor of image pan-sharpening using the Brovey method.
        """

        if self.use_brovey:
            if self.downsampling.filter is not None:
                factor = self.downsampling.factor**2
            else:
                factor = 1

            x = self.downsampling.A_adjoint(y[0], **kwargs) * factor
            x *= y[1] / x.mean(1, keepdim=True)
            return x
        else:
            return super().A_dagger(y, **kwargs)


# test code
# if __name__ == "__main__":
#     device = "cuda:0"
#     import torch
#     import torchvision
#     import deepinv
#
#     device = "cuda:0"
#
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(device) / 255
#     x = torchvision.transforms.Resize((160, 180))(x)
#
#     class Toy(LinearPhysics):
#         def __init__(self, **kwargs):
#             super().__init__(**kwargs)
#             self.A = lambda x: x * 2
#             self.A_adjoint = lambda x: x * 2
#
#     sigma_noise = 0.1
#     kernel = torch.zeros((1, 1, 15, 15), device=device)
#     kernel[:, :, 7, :] = 1 / 15
#     # physics = deepinv.physics.BlurFFT(img_size=x.shape[1:], filter=kernel, device=device)
#     physics = Pansharpen(factor=8, img_size=x.shape[1:], device=device)
#
#     y = physics(x)
#
#     xhat2 = physics.A_adjoint(y)
#     xhat1 = physics.A_dagger(y)
#
#     physics.compute_norm(x)
#     physics.adjointness_test(x)
#
#     deepinv.utils.plot(
#         [y[0], y[1], xhat2, xhat1, x],
#         titles=["low res color", "high res gray", "A_adjoint", "A_dagger", "x"],
#     )
