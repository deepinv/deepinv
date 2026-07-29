from __future__ import annotations
import torch
import torch.nn.functional as F
import math

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
    :param tuple[int] img_size: shape of the input image (C, H, W).
    :param torch.Tensor, str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param tuple[int] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'mps', 'cuda'.
    :param torch.dtype, dtype: type to be associated with the signal.
    """

    def __init__(
        self,
        physics: Physics,
        img_size: tuple[int, ...],
        filter: torch.Tensor | str = "sinc",
        factors: tuple[int, ...] = (2, 4, 8),
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        # NOTE: `device` is passed to super().__init__ (even if Physics does not use it) for proper variable propagation during Method Resolution Order (MRO: https://docs.python.org/3/howto/mro.html) when inherited jointly with another class, e.g., with LinearPhysics
        super().__init__(noise_model=physics.noise_model, device=device, **kwargs)
        self.base = physics
        self.factors = factors
        self.img_size = img_size
        self.Upsamplings = [
            Upsampling(
                img_size=img_size,
                filter=filter,
                factor=factor,
            ).to(device=device, dtype=dtype)
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
        >>> physics = dinv.physics.BlurFFT(img_size=(1, 32, 32), filter=dinv.physics.functional.gaussian_blur(sigma=(0.2, 0.2)))
        >>> x = torch.rand((1, 1, 8, 8))  # define an image 4 times smaller than the physics input size (scale = 2)
        >>> new_physics = dinv.physics.LinearPhysicsMultiScaler(physics, (1, 32, 32), factors=[2, 4, 8])  # define a multiscale physics with base img size (1, 32, 32)
        >>> y = new_physics(x, scale=2)  # applying physics at scale 2
        >>> print(y.shape)
        torch.Size([1, 1, 32, 32])

    :param deepinv.physics.Physics physics: base physics operator.
    :param tuple img_size: shape of the input image (C, H, W).
    :param torch.Tensor, str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param tuple[int, ...] factors: list of factors to use for upsampling.
    :param str, torch.device device: device to use for the upsampling operator, e.g., 'cpu', 'mps', 'cuda'.
    """

    def __init__(
        self,
        physics: Physics,
        img_size: tuple[int, ...],
        filter: torch.Tensor | str = "sinc",
        factors: tuple[int, ...] = (2, 4, 8),
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        super().__init__(
            physics=physics,
            img_size=img_size,
            filter=filter,
            factors=factors,
            device=device,
            **kwargs,
        )

    def A_adjoint(self, y: torch.Tensor, scale: int | None=None, **kwargs):
        self.set_scale(scale)
        y = self.base.A_adjoint(y, **kwargs)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_dagger(self, y: torch.Tensor, scale: int | None=None, **kwargs):
        r"""
        Computes the pseudo-inverse of the linear operator :math:`A`.

        If the scale is set to 0, it uses the base physics pseudo-inverse, which might have a more efficient implementation.

        :param torch.Tensor y: measurements tensor
        :return: (:class:`torch.Tensor`) estimated signal tensor
        """
        self.set_scale(scale)
        if self.scale == 0:
            # use efficient implementation if available (eg SVD-based)
            return self.base.A_dagger(y, **kwargs)
        else:
            return self.super().A_dagger(y, **kwargs)

    def prox_l2(
        self,
        z,
        y,
        gamma,
        solver="CG",
        max_iter=None,
        tol=None,
        verbose=False,
        scale=None,
        **kwargs,
    ):
        r"""
        Computes proximal operator of :math:`f(x) = \frac{1}{2}\|Ax-y\|^2`, i.e.,

        .. math::

            \underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2

        If the scale is set to 0, it uses the base physics proximal operator, which might have a more efficient implementation.

        :param torch.Tensor y: measurements tensor
        :param torch.Tensor z: signal tensor
        :param float gamma: hyperparameter of the proximal operator
        :param str solver: solver to use for the proximal operator, see :func:`deepinv.optim.linear.least_squares` for details
        :param int max_iter: maximum number of iterations for iterative solvers
        :param float tol: tolerance for iterative solvers
        :param bool verbose: whether to print information during the solver execution
        :param int scale: scale at which to apply the physics operator
        :return: (:class:`torch.Tensor`) estimated signal tensor

        """
        self.set_scale(scale)
        if self.scale == 0:
            return self.base.prox_l2(
                z,
                y,
                gamma,
                solver=solver,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                **kwargs,
            )
        else:
            return super().prox_l2(
                z,
                y,
                gamma,
                solver=solver,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
                **kwargs,
            )


def coarse_blur_filter(
    in_filter: torch.Tensor, downsampling_filter: torch.Tensor, scale: int = 2
):
    r"""
    Used to compute the blur filter associated with a coarse scale.

    :param torch.Tensor in_filter: fine scale blur filter
    :param torch.Tensor downsampling_filter: downsampling antialiasing filter (e.g. sinc)
    :param int scale: scale factor using for downsampling
    :return torch.Tensor: coarse blur filter
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
    :param tuple[int, ...] img_size: shape of the input image (C, H, W).
    :param torch.Tensor, str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param tuple[int, ...] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(
        self,
        physics: Physics,
        img_size: tuple[int, ...],
        filter: torch.Tensor | str = "sinc",
        factors: tuple[int, ...] = (2, 4, 8),
        **kwargs,
    ):
        super().__init__(
            physics=physics,
            img_size=img_size,
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

    def downsample_measurement(self, y: torch.Tensor, scale: int | None = None):
        r"""
        Since the observation `y` lives in image space, it can be passed to a coarse scale.
        :param torch.Tensor y: fine scale observation
        :param int scale: target scale in which express `y`
        :return torch.Tensor: downsampled observation `y`
        """
        self.set_scale(scale)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_adjoint_A(self, x: torch.Tensor, scale: int | None = None, **kwargs):
        r"""
        Less computationally expensive version than parent class :class:`LinearPhysicsMultiScaler`

        :param torch.Tensor x: input signal
        :param int scale: scale in which to apply :math:`U_{scale}^* U_{scale}`
        :return torch.Tensor: `U_{scale}^* U_{scale} x`
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
    :param tuple[int, ...] img_size: shape of the input image (C, H, W).
    :param torch.Tensor, str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param tuple[int, ...] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(
        self,
        physics: Physics,
        img_size: tuple[int, ...],
        filter: torch.Tensor | str = "sinc",
        factors: tuple[int, ...] = (2, 4, 8),
        **kwargs,
    ):
        super().__init__(
            physics=physics,
            img_size=img_size,
            filter=filter,
            factors=factors,
            **kwargs,
        )

        self.scaled_physics = []
        for upsampling in self.Upsamplings:
            factor = upsampling.factor
            filt = coarse_blur_filter(physics.filter, upsampling.filter, factor)
            coarse_shape = (
                img_size[0],
                math.ceil(img_size[1] / factor),
                math.ceil(img_size[2] / factor),
            )
            p = BlurFFT(
                filter=filt, img_size=coarse_shape, device=physics.filter.device
            )
            self.scaled_physics.append(p)

    def downsample_measurement(self, y: torch.Tensor, scale: int | None = None):
        r"""
        Since the observation `y` lives in image space, it can be passed to a coarse scale.
        :param torch.Tensor y: fine scale observation
        :param int scale: target scale in which express `y`
        :return torch.Tensor: downsampled observation `y`
        """
        self.set_scale(scale)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_adjoint_A(self, x: torch.Tensor, scale: int | None = None, **kwargs):
        r"""
        Less computationnaly expensive version than parent class :class:`LinearPhysicsMultiScaler`

        :param torch.Tensor x: input signal
        :param int scale: scale in which to apply :math:`U_{scale}^* U_{scale}`
        :return torch.Tensor: `U_{scale}^* U_{scale} x`
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
    :param tuple[int, ...] img_size: shape of the input image (C, H, W).
    :param torch.Tensor, str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param tuple[int, ...] factors: list of factors to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(
        self,
        physics: Physics,
        img_size: tuple[int, ...],
        filter: torch.Tensor | str = "sinc",
        factors: tuple[int, ...] = (2, 4, 8),
        **kwargs,
    ):
        super().__init__(
            physics=physics,
            img_size=img_size,
            filter=filter,
            factors=factors,
            **kwargs,
        )

        self.scaled_physics = []
        for upsampling in self.Upsamplings:
            coarse_data = upsampling.A_adjoint(physics.mask.data)
            p = Inpainting(
                img_size=coarse_data.shape[1:],
                mask=coarse_data,
                device=physics.mask.device,
            )
            self.scaled_physics.append(p)

    def downsample_measurement(self, y: torch.Tensor, scale: int | None = None):
        r"""
        Since the observation `y` lives in image space, it can be passed to a coarse scale.
        :param torch.Tensor y: fine scale observation
        :param int scale: target scale in which express `y`
        :return torch.Tensor: downsampled observation `y`
        """
        self.set_scale(scale)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)

    def A_adjoint_A(self, x: torch.Tensor, scale: int | None = None, **kwargs):
        r"""
        Less computationnaly expensive version than parent class :class:`LinearPhysicsMultiScaler`

        :param torch.Tensor x: input signal
        :param int scale: scale in which to apply :math:`U_{scale}^* U_{scale}`
        :return torch.Tensor: `U_{scale}^* U_{scale} x`
        """
        self.set_scale(scale)
        physics = self.scaled_physics[self.scale - 1]
        factor = self.factors[self.scale - 1]
        return physics.A_adjoint_A(x) / factor**2


def to_multiscale(
    physics: Physics,
    img_size: tuple[int, ...],
    factors: tuple[int, ...] = (2, 4, 8),
    device: str = "cpu",
    dtype: torch.dtype | None = None,
):
    r"""
    This function creates the proper MultiScalerPhysics associated with the provided Physics.
    A MultiScalerPhysics adapt itself to several scales of the given signal.
    Some special cases of Physics have their own implementations, for example :

    - LinearPhysics has an adjoint method computing the observation from a signal given in a coarse scale.

    - Blur and BlurFFT adapt to scale by downsampling their blur filter.

    - Inpainting adapts to scale by downsampling its mask.

    Other futur specific implementation of MultiScalerPhysics may be added here. Currenlty, only 2D signals are supported.

    See :class:`LinearPhysicsMultiScaler` for details.

    :param Physics physics: physics that should be converted to a MultiScaler
    :param tuple[int, ...] img_size: shape of the image in the fine scale
    :param tuple[int, ...] factors: downsampling factors used to get in coarser scales
    :return PhysicsMultiScaler: a MultiScaler version of the provided physics
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    :param torch.dtype dtype: type to be associated with the signal
    """
    if isinstance(physics, Blur):
        return BlurMultiScaler(
            physics, img_size, factors=factors, device=device, dtype=dtype
        )
    if isinstance(physics, BlurFFT):
        return BlurFFTMultiScaler(
            physics, img_size, factors=factors, device=device, dtype=dtype
        )
    if isinstance(physics, Inpainting):
        return InpaintingMultiScaler(
            physics, img_size, factors=factors, device=device, dtype=dtype
        )
    elif isinstance(physics, LinearPhysics):
        return LinearPhysicsMultiScaler(
            physics, img_size, factors=factors, device=device, dtype=dtype
        )
    else:
        return PhysicsMultiScaler(
            physics, img_size, factors=factors, device=device, dtype=dtype
        )


class PhysicsCropper(LinearPhysics):
    r"""
    Cropping for linear physics operators.

    Given a linear physics operator :math:`A`, this operator instantiates a new operator :math:`\tilde{A} = A \circ C` where :math:`C` is a cropping operator that crops the input tensor.
    The adjoint operator is defined as :math:`\tilde{A}^{\top} = C^{\top} \circ A^{\top}` and :math:`C^{\top}` is a padding operator that pads the input tensor to the original size.

    :param deepinv.physics.LinearPhysics physics: base linear physics operator.
    :param tuple crop: padding to apply to the input tensor, e.g., `(pad_height, pad_width)` or `(pad_z, pad_height, pad_weight)` where `pad_z` is either channel or depth dimension pad.
    :param torch.device, str device: cpu or cuda, every registered buffer and module parameters are recursively pushed onto the device during initialization.

    """

    def __init__(
        self,
        physics,
        crop,
        device: torch.device | str = "cpu",
    ):
        super().__init__(noise_model=physics.noise_model, device=device)
        self.base = physics
        self.crop = crop
        if len(self.crop) not in (2, 3):
            raise ValueError("Crop must be a tuple of length 2 or 3.")

    def A(self, x, **kwargs):
        return self.base.A(self.remove_pad(x), **kwargs)

    def A_adjoint(self, y, **kwargs):
        y = self.pad(self.base.A_adjoint(y, **kwargs))
        return y

    def remove_pad(self, x):
        if len(self.crop) == 2:
            return x[..., self.crop[0] :, self.crop[1] :]
        elif len(self.crop) == 3:
            return x[..., self.crop[0] :, self.crop[1] :, self.crop[2] :]

    def pad(self, x):
        if len(self.crop) == 3:
            return torch.nn.functional.pad(
                x, (self.crop[2], 0, self.crop[1], 0, self.crop[0], 0)
            )
        else:
            return torch.nn.functional.pad(x, (self.crop[1], 0, self.crop[0], 0))

    def update_parameters(self, **kwargs):
        self.base.update_parameters(**kwargs)
