import torch
from deepinv.physics.forward import LinearPhysics
from deepinv.transform.base import Transform


class VirtualLinearPhysics(LinearPhysics):
    r"""
    Virtual linear operator

    A virtual operator is an operator of the form

    .. math::

        A = \tilde{A} T_g

    where :math:`\tilde{A}` is a linear operator and :math:`T_g` is an invertible transformation with parameters :math:`g`.

    Unlike general composition of linear operators, like for :class:`deepinv.physics.ComposedLinearPhysics`, the invertibility of :math:`T_g` allows to compute the pseudo-inverse of :math:`A` in a computationally efficient closed form, i.e.,

    .. math::

        A^\dagger = T_g^{-1} \tilde{A}^\dagger.

    Virtual operators are used in :class:`deepinv.models.EquivariantReconstructor`. For more details, see :footcite:t:`sechaud26Equivariant`.

    .. warning::

        The adjoint and pseudo-inverse might be incorrect if the transformation is not invertible, for instance due to boundary effects.

    :param LinearPhysics physics: linear physics operator :math:`\tilde{A}`.
    :param Transform transform: transformation :math:`T_g`
    :param dict g_params: parameters of the transformation :math:`g`.
    """

    def __init__(self, *, physics: LinearPhysics, transform: Transform, g_params: dict):
        super().__init__()
        self.physics = physics
        self.T = lambda x: transform.transform(x, **g_params)
        self.T_inv = lambda x: transform.inverse(x, **g_params)

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Apply the virtual operator to an input

        :param torch.Tensor x: input image.
        :return: (:class:`torch.Tensor`) output of the virtual operator.
        """
        Tx = self.T(x)  # T
        return self.physics.A(Tx)  # A

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Apply the adjoint of the virtual operator to an input

        :param torch.Tensor y: input measurement.
        :return: (:class:`torch.Tensor`) output of the adjoint of the virtual operator.
        """
        x = self.physics.A_adjoint(y)  # A^*
        return self.T_inv(x)  # T^{-1}

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Apply the pseudo-inverse of the virtual operator to an input

        :param torch.Tensor y: input measurement.
        :return: (:class:`torch.Tensor`) output of the pseudo-inverse of the virtual operator.
        """
        x = self.physics.A_dagger(y)  # A^\dagger
        return self.T_inv(x)  # T^{-1}
