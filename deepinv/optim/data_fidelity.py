from deepinv.optim.distance import Distance
from deepinv.optim.potential import Potential


class DataFidelity(Potential):

    def __init__(self, distance):
        super().__init__()
        self.distance = distance

    def h(self, x, y, physics, *args, **kwargs):
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.Tensor) data fidelity :math:`\datafid{x}{y}`.
        """
        return self.distance(physics.A(x), y, *args, **kwargs)

    def grad(self, x, y, physics, *args, **kwargs):
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        The gradient is computed using the chain rule:

        .. math::

            \nabla_x \distance{\forw{x}}{y} = \left. \frac{\partial A}{\partial x} \right|_x^\top \nabla_u \distance{u}{y},

        where :math:`\left. \frac{\partial A}{\partial x} \right|_x` is the Jacobian of :math:`A` at :math:`x`, and :math:`\nabla_u \distance{u}{y}` is computed using ``grad_d`` with :math:`u = \forw{x}`. The multiplication is computed using the ``A_vjp`` method of the physics.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.Tensor) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return physics.A_vjp(x, self.distance.grad(physics.A(x), y, *args, **kwargs))
