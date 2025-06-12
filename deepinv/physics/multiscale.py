from deepinv.physics import Physics, LinearPhysics
from deepinv.physics.blur import Upsampling


class MultiScalePhysics(Physics):
    r"""
    Multi-scale physics operator.

    This operator applies a physics model at a given scale
    by upsampling the input signal before applying the base physics operator.

    .. math::

        A(x) = A_{base}(U_{scale}(x))

    where :math:`U_{scale}` is the upsampling operator for the given scale and :math:`A_{base}` is the base physics operator.

    :param deepinv.physics.Physics physics: base physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] scales: list of scales to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(
        self,
        physics,
        img_shape,
        filter="sinc",
        scales=[2, 4, 8],
        device="cpu",
        **kwargs,
    ):
        super().__init__(noise_model=physics.noise_model, **kwargs)
        self.base = physics
        self.scales = scales
        self.img_shape = img_shape
        self.Upsamplings = [
            Upsampling(img_size=img_shape, filter=filter, factor=factor, device=device)
            for factor in scales
        ]
        self.scale = 0

    def set_scale(self, scale):
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


class MultiScaleLinearPhysics(MultiScalePhysics, LinearPhysics):
    r"""
    Multi-scale linear physics operator.

    This operator applies a physics model at a given scale
    by upsampling the input signal before applying the base physics operator.

    .. math::

        A(x) = A_{base}(U_{scale}(x))

    where :math:`U_{scale}` is the upsampling operator for the given scale and :math:`A_{base}` is the base linear physics operator.

    :param deepinv.physics.Physics physics: base physics operator.
    :param tuple img_shape: shape of the input image (C, H, W).
    :param str filter: type of filter to use for upsampling, e.g., 'sinc', 'nearest', 'bilinear'.
    :param list[int] scales: list of scales to use for upsampling.
    :param torch.device, str device: device to use for the upsampling operator, e.g., 'cpu', 'cuda'.
    """

    def __init__(self, physics, img_shape, filter="sinc", scales=[2, 4, 8], **kwargs):
        super().__init__(
            physics=physics, img_shape=img_shape, filter=filter, scales=scales, **kwargs
        )

    def A_adjoint(self, y, scale=None, **kwargs):
        self.set_scale(scale)
        y = self.base.A_adjoint(y, **kwargs)
        if self.scale == 0:
            return y
        else:
            return self.Upsamplings[self.scale - 1].A_adjoint(y)


class Pad(LinearPhysics):
    r"""
    Padding of linear physics operators.

    Given a linear physics operator :math:`A`, this operator instantiates a new operator :math:`\tilde{A} = A \circ U` where :math:`U` is a padding operator that pads the input tensor with zeros.
    The adjoint operator is defined as :math:`\tilde{A}^{\top} = U^{\top} \circ A^{\top}`.

    :param deepinv.physics.LinearPhysics physics: base linear physics operator.
    :param tuple pad: padding to apply to the input tensor, e.g., (pad_height, pad_width).
    """

    def __init__(self, physics, pad):
        super().__init__(noise_model=physics.noise_model)
        self.base = physics
        self.pad = pad

    def A(self, x):
        return self.base.A(x[..., self.pad[0] :, self.pad[1] :])

    def A_adjoint(self, y):
        y = self.base.A_adjoint(y)
        y = torch.nn.functional.pad(y, (self.pad[1], 0, self.pad[0], 0))
        return y

    def remove_pad(self, x):
        return x[..., self.pad[0] :, self.pad[1] :]

    def update_parameters(self, **kwargs):
        self.base.update_parameters(**kwargs)
