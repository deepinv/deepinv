import torch
import torch.nn.functional as F

from deepinv.physics import Physics, LinearPhysics, Inpainting
from deepinv.physics.blur import Upsampling, Blur, BlurFFT
from typing import Sequence  # noqa: F401


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
    :param Sequence[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    :param torch.dtype, dtype: type to be associated with the signal.
    """

    def __init__(
        self,
        physics,
        img_shape,
        filter="sinc",
        factors=(2, 4, 8),
        device="cpu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(noise_model=physics.noise_model, **kwargs)
        self.base = physics
        self.factors = factors
        self.img_shape = img_shape
        self.Upsamplings = []
        for factor in factors:
            upsampling = Upsampling(
                img_size=img_shape, filter=filter, factor=factor, device=device
            )
            if dtype is not None:
                upsampling.filter = upsampling.filter.to(dtype=dtype)
            self.Upsamplings.append(upsampling)

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
    :param str, torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(
        self,
        physics,
        img_shape,
        filter="sinc",
        factors=(2, 4, 8),
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            physics=physics,
            img_shape=img_shape,
            filter=filter,
            factors=factors,
            device=device,
            **kwargs,
        )

    def A_adjoint(self, y, scale=None, **kwargs):
        self.set_scale(scale)
        y = self.base.A_adjoint(y, **kwargs)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)


def coarse_blur_filter(in_filter, downsampling_filter, scale=2):
    r"""
    Used to compute the blur filter associated with a coarse scale.

    :param in_filter: fine scale blur filter
    :param downsampling_filter: downsampling antialiasing filter (e.g. sinc)
    :param scale: scale factor using for downsampling
    :return: coarse blur filter
    """

    # pad in_filter to make sure it is at least as big as downsampling_filter
    diff_h = max(downsampling_filter.shape[-2] - in_filter.shape[-2], 0)
    diff_w = max(downsampling_filter.shape[-1] - in_filter.shape[-1], 0)

    pad_left = diff_w // 2
    pad_top = diff_h // 2
    new_filt = F.pad(
        in_filter, (pad_left, diff_w - pad_left, pad_top, diff_h - pad_top)
    )

    # pad in_filter in order to perform a "valid" convolution
    df_shape = downsampling_filter.shape
    pad_size = (df_shape[-1] // 2,) * 2 + (df_shape[-2] // 2,) * 2
    new_filt = torch.nn.functional.pad(new_filt, pad_size)

    # downsample the blur filter
    df_groups = downsampling_filter.repeat(
        [new_filt.shape[1]] + [1] * (len(new_filt.shape) - 1)
    )
    coarse_filter = torch.nn.functional.conv2d(
        new_filt, df_groups, groups=new_filt.shape[1], stride=scale, padding="valid"
    )
    coarse_filter = coarse_filter / torch.sum(coarse_filter) * torch.sum(new_filt)

    return coarse_filter


class BlurMultiScaler(LinearPhysicsMultiScaler, LinearPhysics):
    r"""
    Multi-scale wrapper for blur physics operators. This particular class handles A_adjoint_A with a particular implementation for each scale.

    See :class:`LinearPhysicsMultiScaler` for details.

    :param deepinv.physics.Physics physics: blur physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(self, physics, img_shape, filter="sinc", factors=(2, 4, 8), **kwargs):
        super().__init__(
            physics=physics,
            img_shape=img_shape,
            filter=filter,
            factors=factors,
            **kwargs,
        )

        self.scaled_physics = []
        for upsampling in self.Upsamplings:
            filt = coarse_blur_filter(
                physics.filter, upsampling.filter, upsampling.factor
            )
            p = Blur(filter=filt, padding=physics.padding, device=physics.filter.device)
            self.scaled_physics.append(p)

    def downsample_measurement(self, y, scale=None):
        r"""
        Since the observation `y` lives in image space, it can be passed to a coarse scale.
        :param y: fine scale observation
        :param scale: target scale in which express `y`
        :return: downsampled observation `y`
        """
        self.set_scale(scale)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_adjoint_A(self, x, scale=None, **kwargs):
        r"""
        Less computationnaly expensive version than parent class :class:`LinearPhysicsMultiScaler`

        :param x: input signal
        :param scale: scale in which to apply :math:`U_{scale}^* U_{scale}`
        :return: `U_{scale}^* U_{scale} x`
        """
        self.set_scale(scale)
        physics = self.scaled_physics[self.scale - 1]
        factor = self.factors[self.scale - 1]
        return physics.A_adjoint_A(x) / factor**2


class BlurFFTMultiScaler(LinearPhysicsMultiScaler, LinearPhysics):
    r"""
    Multi-scale wrapper for BlurFFT operators. This particular class handles A_adjoint_A with a particular implementation for each scale.

    See :class:`LinearPhysicsMultiScaler` for details.

    :param deepinv.physics.Physics physics: BlurFFT physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(self, physics, img_shape, filter="sinc", factors=(2, 4, 8), **kwargs):
        super().__init__(
            physics=physics,
            img_shape=img_shape,
            filter=filter,
            factors=factors,
            **kwargs,
        )

        self.scaled_physics = []
        for upsampling in self.Upsamplings:
            factor = upsampling.factor
            filt = coarse_blur_filter(self.filter, upsampling.filter, factor)
            coarse_shape = [int(torch.ceil(s / factor)) for s in self.img_size]
            coarse_shape[0] = self.img_size[0]  # keep the same nb of channels
            p = BlurFFT(
                filter=filt, img_size=coarse_shape, device=physics.filter.device
            )
            self.scaled_physics.append(p)

    def downsample_measurement(self, y, scale=None):
        r"""
        Since the observation `y` lives in image space, it can be passed to a coarse scale.
        :param y: fine scale observation
        :param scale: target scale in which express `y`
        :return: downsampled observation `y`
        """
        self.set_scale(scale)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_adjoint_A(self, x, scale=None, **kwargs):
        r"""
        Less computationnaly expensive version than parent class :class:`LinearPhysicsMultiScaler`

        :param x: input signal
        :param scale: scale in which to apply :math:`U_{scale}^* U_{scale}`
        :return: `U_{scale}^* U_{scale} x`
        """
        self.set_scale(scale)
        physics = self.scaled_physics[self.scale - 1]
        factor = self.factors[self.scale - 1]
        return physics.A_adjoint_A(x) / factor**2


class InpaintingMultiScaler(LinearPhysicsMultiScaler, LinearPhysics):
    r"""
    Multi-scale wrapper for inpainting/demosaicing operators. This particular class handles A_adjoint_A with a particular implementation for each scale.

    See :class:`LinearPhysicsMultiScaler` for details.

    :param deepinv.physics.Physics physics: inpainting or demosaicing physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(self, physics, img_shape, filter="sinc", factors=(2, 4, 8), **kwargs):
        super().__init__(
            physics=physics,
            img_shape=img_shape,
            filter=filter,
            factors=factors,
            **kwargs,
        )

        self.scaled_physics = []
        for upsampling in self.Upsamplings:
            coarse_data = upsampling.A_adjoint(physics.mask.data)
            p = Inpainting(
                tensor_size=coarse_data.shape[1:],
                mask=coarse_data,
                device=physics.mask.device,
            )
            self.scaled_physics.append(p)

    def downsample_measurement(self, y, scale=None):
        r"""
        Since the observation `y` lives in image space, it can be passed to a coarse scale.
        :param y: fine scale observation
        :param scale: target scale in which express `y`
        :return: downsampled observation `y`
        """
        self.set_scale(scale)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_adjoint_A(self, x, scale=None, **kwargs):
        r"""
        Less computationnaly expensive version than parent class :class:`LinearPhysicsMultiScaler`

        :param x: input signal
        :param scale: scale in which to apply :math:`U_{scale}^* U_{scale}`
        :return: `U_{scale}^* U_{scale} x`
        """
        self.set_scale(scale)
        physics = self.scaled_physics[self.scale - 1]
        factor = self.factors[self.scale - 1]
        return physics.A_adjoint_A(x) / factor**2


def to_multiscale(physics, img_shape, dtype=None, factors=(2, 4, 8)):
    r"""
    This function creates the proper MultiScaler (see :class:`PhysicsMultiScaler` for details) object associated with the provided physics.

    :param physics: physics that should be converted to a MultiScaler
    :param img_shape: shape of the image in the fine scale
    :param torch.dtype, dtype: type to be associated with the signal
    :param factors: downsampling factors used to get in coarser scales
    :return: a MultiScaler version of the provided physics
    """
    if isinstance(physics, Blur):
        return BlurMultiScaler(physics, img_shape, dtype=dtype, factors=factors)
    if isinstance(physics, BlurFFT):
        return BlurFFTMultiScaler(physics, img_shape, dtype=dtype, factors=factors)
    if isinstance(physics, Inpainting):
        return InpaintingMultiScaler(physics, img_shape, dtype=dtype, factors=factors)
    elif isinstance(physics, LinearPhysics):
        return LinearPhysicsMultiScaler(
            physics, img_shape, dtype=dtype, factors=factors
        )
    else:
        return PhysicsMultiScaler(physics, img_shape, dtype=dtype, factors=factors)


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
