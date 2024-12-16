import torch
from deepinv.physics.noise import GaussianNoise
from deepinv.physics.forward import StackedLinearPhysics
from deepinv.physics.blur import Downsampling
from deepinv.physics.range import Decolorize
from deepinv.optim.utils import conjugate_gradient


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
    :param torch.Tensor, str, NoneType filter: Downsampling filter. It can be 'gaussian', 'bilinear' or 'bicubic' or a
        custom ``torch.Tensor`` filter. If ``None``, no filtering is applied.
    :param int factor: downsampling factor.
    :param torch.nn.Module noise_color: noise model for the RGB image.
    :param torch.nn.Module noise_gray: noise model for the grayscale image.

    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    |sep|

    :Examples:

        Pansharpen operator applied to a random 32x32 image:

        >>> from deepinv.physics import Pansharpen
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
        srf=None,
        noise_color=GaussianNoise(sigma=0.0),
        noise_gray=GaussianNoise(sigma=0.05),
        device="cpu",
        padding="circular",
        **kwargs,
    ):
        assert len(img_size) == 3, "img_size must be of shape (C,H,W)"

        noise_color = noise_color if noise_color is not None else lambda x: x
        noise_gray = noise_gray if noise_gray is not None else lambda x: x

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

    def A_dagger(self, y, **kwargs):
        r"""
        Computes the solution in :math:`x` to :math:`y = Ax` using the
        `conjugate gradient method <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_,
        see :meth:`deepinv.optim.utils.conjugate_gradient`.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :return: (torch.Tensor) The reconstructed image :math:`x`.

        """

        A = lambda x: self.A_A_adjoint(x)
        b = y
        x = conjugate_gradient(A=A, b=b, max_iter=self.max_iter, tol=self.tol, eps=0.1)

        x = self.A_adjoint(x)
        return x

    def A_classical(self, y, **kwargs):
        """
        TODO: explain

        From https://github.com/AlexeyTrekin/pansharpen/blob/master/pysharpen/methods/sharpening/brovey.py
        ESRI Brovey from https://pro.arcgis.com/en/pro-app/latest/help/analysis/raster-functions/fundamentals-of-pan-sharpening-pro.htm
        Another unused implementation https://github.com/mapbox/rio-pansharpen/blob/master/rio_pansharpen/methods.py
        """
        hrms = self.downsampling.A_adjoint(y[0], **kwargs)
        hrms *= y[1] / hrms.mean(axis=1)
        return hrms


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
