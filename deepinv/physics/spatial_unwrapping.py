import torch
from deepinv.physics.forward import Physics

modulo_floor = lambda x, t: x - t * torch.floor(x / t)
modulo_round = lambda x, t: x - t * torch.round(x / t)


class SpatialUnwrapping(Physics):
    r"""
    Spatial unwrapping forward operator.

    This class implements a forward operator for spatial unwrapping, where the input is wrapped modulo a threshold value.
    The operator can use either floor or round mode for the wrapping operation. It is useful for problems where the observed data is wrapped,
    such as in phase imaging, modulo imaging, or interferometry.

    The forward operator is defined as:

    .. math::

        y = w_t(x) = x - t \cdot \mathrm{q}(x / t)

    where :math:`w_t` is the modulo operator, :math:`t` is the threshold, and :math:`\mathrm{q}` is either the rounding or flooring function depending on the mode.


    :param float threshold: The threshold :math:`t` value for the modulo operation (default: 1.0).
    :param str mode: modulo function :math:`q(\cdot)`, either 'round' or 'floor' (default: 'round').
    :param kwargs: Additional arguments passed to the base Physics class.

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics.spatial_unwrapping import SpatialUnwrapping
        >>> x = torch.tensor([[0.5, 1.2, 2.7]])
        >>> physics = SpatialUnwrapping(threshold=1.0, mode="round")
        >>> y = physics(x)
        >>> print(torch.round(y, decimals=1))
        tensor([[ 0.5000,  0.2000, -0.3000]])

    """

    def __init__(self, threshold=1.0, mode="round", **kwargs):

        super().__init__(**kwargs)

        self.threshold = threshold
        self.mode = mode

        self.B = modulo_floor if mode == "floor" else modulo_round

    def forward(self, x, **kwargs):
        r"""
        Applies the forward model for spatial unwrapping.

        In spatial unwrapping, the noise model is first applied to the input, followed by the modulo operator.

        :param torch.Tensor x: Input tensor.
        :return: (:class:`torch.Tensor`) The result after applying noise, modulo operator, and sensor.
        """
        return self.sensor(self.A(self.noise(x, **kwargs), **kwargs))

    def A(self, x, **kwargs):
        r"""
        Applies the modulo operator to the input tensor.

        :param torch.Tensor x: Input tensor.
        :return: (:class:`torch.Tensor`) Modulo tensor.
        """
        return self.B(x, self.threshold)

    def A_adjoint(self, y, **kwargs):
        r"""
        Adjoint operator for the modulo operator. For the modulo operator, the adjoint is the identity.

        :param torch.Tensor y: Input tensor.
        :return: (:class:`torch.Tensor`) Output tensor (identity).
        """
        return y
