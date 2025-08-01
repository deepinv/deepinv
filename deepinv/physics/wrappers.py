import torch

from deepinv.physics import Physics, LinearPhysics
from deepinv.physics.blur import Upsampling


class PhysicsMultiScaler(Physics):
    r"""
    Multi-scale wrapper for physics operators.

    This class applies a physics model at a given scale
    by upsampling the input signal before applying the base physics operator.

    .. math::

        A(x) = A_{base}(U_{scale}(x))

    where :math:`U_{scale}` is the upsampling operator for the given scale and :math:`A_{base}` is the base physics operator.

    By default, we assume that the factors for the different scales are [2, 4, 8].
    The 1st scale corresponds to upsampling by a factor of 2, the 2nd scale corresponds to upsampling by a factor of 4, and so on.
    The 0th scale corresponds to the base physics operator without upsampling.

    :param deepinv.physics.Physics physics: base physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(
        self,
        physics,
        img_shape,
        filter="sinc",
        factors=[2, 4, 8],
        device="cpu",
        **kwargs,
    ):
        super().__init__(noise_model=physics.noise_model, **kwargs)
        self.base = physics
        self.factors = factors
        self.img_shape = img_shape
        self.Upsamplings = [
            Upsampling(img_size=img_shape, filter=filter, factor=factor, device=device)
            for factor in factors
        ]
        self.scale = 0

    def set_scale(self, scale=None):
        if scale is not None:
            self.scale = scale

    def A(self, x, scale=None, **kwargs):
        self.set_scale(scale)
        if self.scale == 0:
            return self.base.A(x, **kwargs)
        else:
            return self.base.A(self.Upsamplings[self.scale - 1].A(x), **kwargs)

    def downsample(self, x, scale=None):
        self.set_scale(scale)
        if self.scale == 0:
            return x
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(x)

    def upsample(self, x, scale=None):
        self.set_scale(scale)
        if self.scale == 0:
            return x
        else:
            return self.Upsamplings[self.scale - 1].A(x)

    def update_parameters(self, **kwargs):
        self.base.update_parameters(**kwargs)


class LinearPhysicsMultiScaler(PhysicsMultiScaler, LinearPhysics):
    r"""
    Multi-scale wrapper for linear physics operators.

    See :class:`PhysicsMultiScaler` for details.

    :Examples:

        A multiscale BlurFFT operator can be created as follows:

        >>> import torch
        >>> import deepinv as dinv
        >>> physics = dinv.physics.BlurFFT(img_size=(1, 32, 32), filter=dinv.physics.blur.gaussian_blur(.2))
        >>> x = torch.rand((1, 1, 8, 8))  # define an image 4 times smaller than the physics input size (scale = 2)
        >>> new_physics = dinv.physics.LinearPhysicsMultiScaler(physics, (1, 32, 32), factors=[2, 4, 8])  # define a multiscale physics with base img size (1, 32, 32)
        >>> y = new_physics(x, scale=2)  # applying physics at scale 2
        >>> print(y.shape)
        torch.Size([1, 1, 32, 32])

    :param deepinv.physics.Physics physics: base physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(self, physics, img_shape, filter="sinc", factors=[2, 4, 8], **kwargs):
        super().__init__(
            physics=physics,
            img_shape=img_shape,
            filter=filter,
            factors=factors,
            **kwargs,
        )

    def A_adjoint(self, y, scale=None, **kwargs):
        self.set_scale(scale)
        y = self.base.A_adjoint(y, **kwargs)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)


class PhysicsCropper(LinearPhysics):
    r"""
    Cropping for linear physics operators.

    Given a linear physics operator :math:`A`, this operator instantiates a new operator :math:`\tilde{A} = A \circ C` where :math:`C` is a cropping operator that crops the input tensor.
    The adjoint operator is defined as :math:`\tilde{A}^{\top} = C^{\top} \circ A^{\top}` and :math:`C^{\top}` is a padding operator that pads the input tensor to the original size.

    :param deepinv.physics.LinearPhysics physics: base linear physics operator.
    :param tuple crop: padding to apply to the input tensor, e.g., (pad_height, pad_width).
    """

    def __init__(self, physics, crop):
        super().__init__(noise_model=physics.noise_model)
        self.base = physics
        self.crop = crop

    def A(self, x):
        return self.base.A(self.remove_pad(x))

    def A_adjoint(self, y):
        y = self.pad(self.base.A_adjoint(y))
        return y

    def remove_pad(self, x):
        return x[..., self.crop[0] :, self.crop[1] :]

    def pad(self, x):
        return torch.nn.functional.pad(x, (self.crop[1], 0, self.crop[0], 0))

    def update_parameters(self, **kwargs):
        self.base.update_parameters(**kwargs)
