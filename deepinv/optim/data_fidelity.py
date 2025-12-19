from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import torch
import torch.nn.functional as F

from deepinv.optim.distance import (
    Distance,
    L2Distance,
    L1Distance,
    IndicatorL2Distance,
    AmplitudeLossDistance,
    PoissonLikelihoodDistance,
    LogPoissonLikelihoodDistance,
    ZeroDistance,
)

from deepinv.optim.potential import Potential
from deepinv.physics.functional import dct_2d, idct_2d


if TYPE_CHECKING:
    from deepinv.physics import Physics, StackedPhysics


class DataFidelity(Potential):
    r"""
    Base class for the data fidelity term :math:`\distance{A(x)}{y}` where :math:`A` is the forward operator,
    :math:`x\in\xset` is a variable and :math:`y\in\yset` is the data, and where :math:`d` is a distance function,
    from the class :class:`deepinv.optim.Distance`.

    :param Callable d: distance function :math:`d(x, y)` between a variable :math:`x` and an observation :math:`y`. Default None.
    """

    def __init__(self, d: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None):
        super().__init__()
        self.d = Distance(d=d)

    def fn(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """
        return self.d(physics.A(x), y, *args, **kwargs)

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        The gradient is computed using the chain rule:

        .. math::

            \nabla_x \distance{\forw{x}}{y} = \left. \frac{\partial A}{\partial x} \right|_x^\top \nabla_u \distance{u}{y},

        where :math:`\left. \frac{\partial A}{\partial x} \right|_x` is the Jacobian of :math:`A` at :math:`x`, and :math:`\nabla_u \distance{u}{y}` is computed using ``grad_d`` with :math:`u = \forw{x}`. The multiplication is computed using the ``A_vjp`` method of the physics.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return physics.A_vjp(x, self.d.grad(physics.A(x), y, *args, **kwargs))

    def grad_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the gradient :math:`\nabla_u\distance{u}{y}`, computed in :math:`u`.

        Note that this is the gradient of
        :math:`\distancename` and not :math:`\datafidname`. This function directly calls :func:`deepinv.optim.Potential.grad` for the
        specific distance function :math:`\distancename`.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (:class:`torch.Tensor`) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{y}`.
        """
        return self.d.grad(u, y, *args, **kwargs)

    def prox_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\gamma\distance{\cdot}{y}}(u)`, computed in :math:`u`.

        Note that this is the proximity operator of :math:`\distancename` and not :math:`\datafidname`.
        This function directly calls :func:`deepinv.optim.Potential.prox` for the
        specific distance function :math:`\distancename`.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (:class:`torch.Tensor`) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{y}`.
        """
        return self.d.prox(u, y, *args, **kwargs)

    def prox_d_conjugate(
        self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the proximity operator of the convex conjugate of the distance function :math:`\distance{u}{y}`.

        This function directly calls :func:`deepinv.optim.Potential.prox_conjugate` for the
        specific distance function :math:`\distancename`.
        """
        return self.d.prox_conjugate(u, y, *args, **kwargs)


class StackedPhysicsDataFidelity(DataFidelity):
    r"""
    Stacked data fidelity term :math:`\datafid{x}{y} = \sum_i d_i(A_i(x),y_i)`.

    Adapted to :class:`deepinv.physics.StackedPhysics` physics composed of multiple physics operators.

    :param list[deepinv.optim.DataFidelity] data_fidelity_list: list of data fidelity terms, one per physics operator.

    |sep|

    :Examples:

        Define a stacked data fidelity term with two data fidelity terms :math:`f_1(A_1(x),y_1) + f_2(A_2(x,y_2)`:

        >>> import torch
        >>> import deepinv as dinv
        >>> # define two observations, one with Gaussian noise and one with Poisson noise
        >>> physics1 = dinv.physics.Denoising(dinv.physics.GaussianNoise(.1))
        >>> physics2 = dinv.physics.Denoising(dinv.physics.PoissonNoise(.1))
        >>> physics = dinv.physics.StackedLinearPhysics([physics1, physics2])
        >>> fid1 = dinv.optim.L2()
        >>> fid2 = dinv.optim.PoissonLikelihood()
        >>> data_fidelity = dinv.optim.StackedPhysicsDataFidelity([fid1, fid2])
        >>> x = torch.ones(1, 1, 3, 3) # image
        >>> y = physics(x) # noisy measurements
        >>> d = data_fidelity(x, y, physics)

    """

    def __init__(self, data_fidelity_list: list[DataFidelity]):
        super(StackedPhysicsDataFidelity, self).__init__()
        self.data_fidelity_list = data_fidelity_list

    def fn(
        self, x: torch.Tensor, y: torch.Tensor, physics: StackedPhysics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \sum_i d_i(A_i(x),y_i)`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param deepinv.utils.TensorList y: Stacked measurements :math:`y`.
        :param deepinv.physics.StackedPhysics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """
        out = 0
        for i, data_fidelity in enumerate(self.data_fidelity_list):
            out += data_fidelity.fn(x, y[i], physics[i], *args, **kwargs)
        return out

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: StackedPhysics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        The gradient is computed using the chain rule:

        .. math::

            \nabla_x \distance{\forw{x}}{y} = \sum_i \left. \frac{\partial A_i}{\partial x} \right|_x^\top \nabla_u \distance{u}{y_i},

        where :math:`\left. \frac{\partial A_i}{\partial x} \right|_x` is the Jacobian of :math:`A_i` at :math:`x`,
        and :math:`\nabla_u \distance{u}{y_i}` is computed using ``grad_d`` with :math:`u = \forw{x}`.
        The multiplication is computed using the ``A_vjp`` method of each physics.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param deepinv.utils.TensorList y: Stacked measurements :math:`y`.
        :param deepinv.physics.StackedPhysics physics: Stacked physics model.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        out = 0
        for i, data_fidelity in enumerate(self.data_fidelity_list):
            out += data_fidelity.grad(x, y[i], physics[i], *args, **kwargs)
        return out

    def grad_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the gradient :math:`\nabla_u\distance{u}{y}`, computed in :math:`u`.

        Note that this is the gradient of
        :math:`\distancename` and not :math:`\datafidname`. This function directly calls :func:`deepinv.optim.Potential.grad` for the
        specific distance function :math:`\distancename_i`.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (:class:`torch.Tensor`) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{y}`.
        """
        out = 0
        for i, data_fidelity in enumerate(self.data_fidelity_list):
            out += data_fidelity.grad_d(u, y[i], *args, **kwargs)
        return out

    def prox_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\gamma\distance{\cdot}{y}}(u)`, computed in :math:`u`.

        Note that this is the proximity operator of :math:`\distancename` and not :math:`\datafidname`.
        This function directly calls :func:`deepinv.optim.Potential.prox` for the
        specific distance function :math:`\distancename`.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (:class:`torch.Tensor`) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{y}`.
        """
        out = 0
        for i, data_fidelity in enumerate(self.data_fidelity_list):
            out += data_fidelity.prox_d(u, y[i], *args, **kwargs)
        return out

    def prox_d_conjugate(
        self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the proximity operator of the convex conjugate of the distance function :math:`\distance{u}{y}`.

        This function directly calls :func:`deepinv.optim.Potential.prox_conjugate` for the
        specific distance function :math:`\distancename`.
        """
        out = 0
        for i, data_fidelity in enumerate(self.data_fidelity_list):
            out += data_fidelity.prox_d_conjugate(u, y[i], *args, **kwargs)
        return out


class L2(DataFidelity):
    r"""
    Implementation of the data-fidelity as the normalized :math:`\ell_2` norm

    .. math::

        f(x) = \frac{1}{2\sigma^2}\|\forw{x}-y\|^2

    It can be used to define a log-likelihood function associated with additive Gaussian noise
    by setting an appropriate noise level :math:`\sigma`.

    :param float sigma: Standard deviation of the noise to be used as a normalisation factor.


    .. doctest::

        >>> import torch
        >>> import deepinv as dinv
        >>> # define a loss function
        >>> fidelity = dinv.optim.data_fidelity.L2()
        >>>
        >>> x = torch.ones(1, 1, 3, 3)
        >>> mask = torch.ones_like(x)
        >>> mask[0, 0, 1, 1] = 0
        >>> physics = dinv.physics.Inpainting(img_size=(1, 3, 3), mask=mask)
        >>> y = physics(x)
        >>>
        >>> # Compute the data fidelity f(Ax, y)
        >>> fidelity(x, y, physics)
        tensor([0.])
        >>> # Compute the gradient of f
        >>> fidelity.grad(x, y, physics)
        tensor([[[[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])
        >>> # Compute the proximity operator of f
        >>> fidelity.prox(x, y, physics, gamma=1.0)
        tensor([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.d = L2Distance(sigma=sigma)
        self.norm = 1 / (sigma**2)

    def prox(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        *args,
        gamma: float | torch.Tensor = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Proximal operator of :math:`\gamma \datafid{Ax}{y} = \frac{\gamma}{2\sigma^2}\|Ax-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \datafidname}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \datafidname} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Au-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator :math:`\operatorname{prox}_{\gamma \datafidname}(x)`.
        """
        return physics.prox_l2(x, y, self.norm * gamma)


class ItohFidelity(L2):
    r"""
    Itoh data-fidelity term for spatial unwrapping problems.

    This class implements a data-fidelity term based on the :math:`\ell_2` norm, but applied to the spatial finite differences of the variable and the wrapped differences of the data.
    This is based on the Itoh condition for phase unwrapping :footcite:p:`itoh1982analysis`.
    It is designed to be used in conjunction with the :class:`deepinv.physics.SpatialUnwrapping` class for spatial unwrapping tasks.

    The data-fidelity term is defined as:

    .. math::

        f(x,y) = \frac{1}{2\sigma^2} \| D x - w_{t}(Dy) \|^2

    where :math:`D` denotes the spatial finite differences operator, :math:`w_t` denotes the wrapping operator, and :math:`\sigma` denotes the noise level.

    :param float sigma: Standard deviation of the noise to be used as a normalisation factor.
    :param float threshold: Threshold value :math:`t` used in the wrapping operator (default: 1.0).

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics.spatial_unwrapping import SpatialUnwrapping
        >>> from deepinv.optim.data_fidelity import ItohFidelity
        >>> x = torch.ones(1, 1, 3, 3)
        >>> y = x
        >>> physics = SpatialUnwrapping(threshold=1.0, mode="round")
        >>> fidelity = ItohFidelity(sigma=1.0)
        >>> f = fidelity(x, y, physics)
        >>> print(f)
        tensor([0.])

    """

    def __init__(self, sigma=1.0, threshold=1.0):
        super().__init__()

        self.d = L2Distance(sigma=sigma)
        self.norm = 1 / (sigma**2)
        self.modulo_round = lambda x: x - threshold * torch.round(x / threshold)

    def fn(self, x, y, physics, *args, **kwargs):
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{Dx}{w_{t}(Dy)}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """

        # local import to avoid circular imports between optim and physics
        from deepinv.physics.spatial_unwrapping import SpatialUnwrapping

        if not isinstance(physics, SpatialUnwrapping):
            raise ValueError(
                "ItohFidelity is designed to be used with SpatialUnwrapping physics."
            )

        Dx = self.D(x)
        WDy = self.WD(y)
        return super().fn(Dx, WDy, physics, *args, **kwargs)

    def grad(self, x, y, *args, **kwargs):
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        The gradient is computed using the chain rule:

        .. math::

            \nabla_x \distance{Dx}{w_{t}(Dy)} = \left. \frac{\partial D}{\partial x} \right|_x^\top \nabla_u \distance{u}{w_{t}(Dy)},

        where :math:`\left. \frac{\partial D}{\partial x} \right|_x` is the Jacobian of :math:`D` at :math:`x`, and :math:`\nabla_u \distance{u}{w_{t}(Dy)}` is computed using ``grad_d`` with :math:`u = Dx`. The multiplication is computed using the :func:`D_adjoint <deepinv.optim.ItohFidelity.D_adjoint>` method of the class.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        WDy = self.WD(y)
        return self.D_adjoint(self.d.grad(self.D(x), WDy, *args, **kwargs))

    def grad_d(self, u, y, *args, **kwargs):
        r"""
        Computes the gradient :math:`\nabla_u\distance{u}{w_{t}(Dy)}`, computed in :math:`u`.

        Note that this is the gradient of
        :math:`\distancename` and not :math:`\datafidname`. This function directly calls :func:`deepinv.optim.Potential.grad` for the
        specific distance function :math:`\distancename`.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (:class:`torch.Tensor`) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{w_{t}(Dy)}`.
        """
        WDy = self.WD(y)
        return self.d.grad(u, WDy, *args, **kwargs)

    def prox_d(self, u, y, *args, **kwargs):
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\gamma\distance{\cdot}{w_{t}(Dy)}}(u)`, computed at :math:`u`.

        Note that this is the proximity operator of :math:`\distancename` and not :math:`\datafidname`.
        This function directly calls :func:`deepinv.optim.Potential.prox` for the
        specific distance function :math:`\distancename`.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (:class:`torch.Tensor`) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{w_{t}(Dy)}`.
        """
        WDy = self.WD(y)
        return self.d.prox(u, WDy, *args, **kwargs)

    def D(self, x, **kwargs):
        r"""
        Apply spatial finite differences to the input tensor.

        Computes the horizontal and vertical finite differences of the input tensor `x`
        using first-order differences along the last two spatial dimensions. The result
        is a tensor containing both the horizontal and vertical gradients stacked along
        a new dimension.

        :param torch.Tensor x: Input tensor of shape (..., H, W), where H and W are spatial dimensions.
        :return: (:class:`torch.Tensor`) of shape (..., H, W, 2), where the last dimension contains
            the horizontal and vertical finite differences, respectively.
        """

        Dh_x = F.pad(torch.diff(x, 1, dim=-1), (0, 1))
        Dv_x = F.pad(torch.diff(x, 1, dim=-2), (0, 0, 0, 1))
        out = torch.stack((Dh_x, Dv_x), dim=-1)
        return out

    def D_adjoint(self, x, **kwargs):
        r"""
        Applies the adjoint (transpose) of the spatial finite difference operator to the input tensor.

        This function computes the adjoint operation corresponding to spatial finite differences,
        typically used in image processing and variational optimization problems. The input `x`
        is expected to have its last dimension of size 2, representing the horizontal and vertical
        finite differences :math:`(D_h x, D_v x)`.

        :param torch.Tensor x: Input tensor of shape (..., 2), where the last dimension contains
            the horizontal and vertical finite differences.

        :return: (:class:`torch.Tensor`) The result of applying the adjoint finite difference operator, with the
            same shape as the input except for the last dimension (which is removed).
        """

        Dh_x, Dv_x = torch.unbind(x, dim=-1)
        rho = -(
            torch.diff(F.pad(Dh_x, (1, 0)), 1, dim=-1)
            + torch.diff(F.pad(Dv_x, (0, 0, 1, 0)), 1, dim=-2)
        )
        return rho

    def D_dagger(self, y, **kwargs):
        # fast initialization using DCT
        return self.prox(None, y, physics=None, gamma=None)

    def WD(self, x, **kwargs):
        r"""
        Applies spatial finite differences to the input and wraps the result.

        This method computes the spatial finite differences of the input tensor :math:`x` using the :math:`D` operator,
        then applies modular rounding to the result. This is typically used in
        applications where periodic boundary conditions or phase wrapping are required.

        :param torch.Tensor x: Input tensor to which the spatial finite differences and wrapping are applied.
        :return: (:class:`torch.Tensor`) The wrapped finite differences of the input tensor.
        """

        Dx = self.D(x)
        WDx = self.modulo_round(Dx)
        return WDx

    def prox(self, x, y, physics=None, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of :math:`\gamma \datafid{x}{y}`

        Compute the proximal operator of the fidelity term :math:`\operatorname{prox}_{\gamma \datafidname}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \datafidname} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Du-w_{t}(Dy)\|_2^2+\frac{1}{2}\|u-x\|_2^2

        using the DCT-based closed-form solution of :footcite:t:`ramirez2024phase` as follows

        .. math::
            \hat{x}_{i,j} = \texttt{DCT}^{-1}\left(
            \frac{\texttt{DCT}(D^{\top}w_t(Dy) + \frac{\rho}{2} z)_{i,j}}
            { \frac{\rho}{2} + 4 - (2\cos(\pi i / M) + 2\cos(\pi j / N))}
            \right)

        where :math:`D` is the finite difference operator and :math:`\texttt{DCT}` is the discrete cosine transform.
        """

        psi = self.D_adjoint(self.WD(y))

        if x is not None:
            psi = psi + (gamma / 2) * x

        NX, MX = psi.shape[-1], psi.shape[-2]
        I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
        I, J = I.to(psi.device), J.to(psi.device)

        I, J = I.unsqueeze(0).unsqueeze(0), J.unsqueeze(0).unsqueeze(0)

        if x is None:
            denom = 2 * (
                2 - (torch.cos(torch.pi * I / MX) + torch.cos(torch.pi * J / NX))
            )
        else:
            denom = 2 * (
                (gamma / 4)
                + 2
                - (torch.cos(torch.pi * I / MX) + torch.cos(torch.pi * J / NX))
            )

        dct_psi = dct_2d(psi, norm="ortho")

        denom = denom.to(psi.device)
        denom[..., 0, 0] = 1  # avoid division by zero

        dct_phi = dct_psi / denom

        phi = idct_2d(dct_phi, norm="ortho")

        return phi


class IndicatorL2(DataFidelity):
    r"""
    Data-fidelity as the indicator of :math:`\ell_2` ball with radius :math:`r`.

    .. math::

          \iota_{\mathcal{B}_2(y,r)}(u)= \left.
              \begin{cases}
                0, & \text{if } \|u-y\|_2\leq r \\
                +\infty & \text{else.}
              \end{cases}
              \right.


    :param float radius: radius of the ball. Default: None.

    """

    def __init__(self, radius: float = None):
        super().__init__()
        self.d = IndicatorL2Distance(radius=radius)
        self.radius = radius

    def prox(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        *args,
        radius: float = None,
        stepsize: float = None,
        crit_conv: float = 1e-5,
        max_iter: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\gamma \iota_{\mathcal{B}_2(y, r)}(A\cdot)}(x) = \underset{u}{\text{argmin}} \,\, \iota_{\mathcal{B}_2(y, r)}(Au)+\frac{1}{2}\|u-x\|_2^2

        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm,
        as suggested in :footcite:t:`combettes2011proximal`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`\forw{x}`.
        :param torch.Tensor radius: radius of the :math:`\ell_2` ball.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :return: (:class:`torch.Tensor`) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.

        """
        radius = self.radius if radius is None else radius

        if physics.A(x).shape == x.shape and (physics.A(x) == x).all():  # Identity case
            return self.d.prox(x, y, gamma=None, radius=radius)
        else:
            norm_AtA = physics.compute_sqnorm(x, verbose=False)
            stepsize = 1.0 / norm_AtA if stepsize is None else stepsize
            u = physics.A(x)
            for it in range(max_iter):
                u_prev = u.clone()

                t = x - physics.A_adjoint(u)
                u_ = u + stepsize * physics.A(t)
                u = u_ - stepsize * self.d.prox(
                    u_ / stepsize, y, radius=radius, gamma=None
                )
                rel_crit = torch.linalg.vector_norm(u - u_prev) / (
                    torch.linalg.vector_norm(u) + 1e-12
                )
                if rel_crit < crit_conv:
                    break
            return t


class PoissonLikelihood(DataFidelity):
    r"""

    Poisson negative log-likelihood.

    .. math::

        \datafid{z}{y} =  -y^{\top} \log(z+\beta)+1^{\top}z

    where :math:`y` are the measurements, :math:`z` is the estimated (positive) density and :math:`\beta\geq 0` is
    an optional background level.

    .. note::

        The function is not Lipschitz smooth w.r.t. :math:`z` in the absence of background (:math:`\beta=0`).

    :param float gain: gain of the measurement :math:`y`. Default: 1.0.
    :param float bkg: background level :math:`\beta`. Default: 0.
    :param bool denormalize: if True, the measurement is multiplied by the gain. Default: True.
    """

    def __init__(self, gain: float = 1.0, bkg: float = 0, denormalize: bool = True):
        super().__init__()
        self.d = PoissonLikelihoodDistance(gain=gain, bkg=bkg, denormalize=denormalize)
        self.bkg = bkg
        self.gain = gain
        self.normalize = denormalize


class L1(DataFidelity):
    r"""
    :math:`\ell_1` data fidelity term.

    In this case, the data fidelity term is defined as

    .. math::

        f(x) = \|Ax-y\|_1.

    """

    def __init__(self):
        super().__init__()
        self.d = L1Distance()

    def prox(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        *args,
        gamma: float | torch.Tensor = 1.0,
        stepsize: float = None,
        crit_conv: float = 1e-5,
        max_iter: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Proximal operator of the :math:`\ell_1` norm composed with A, i.e.

        .. math::

            \operatorname{prox}_{\gamma \ell_1}(x) = \underset{u}{\text{argmin}} \,\, \gamma \|Au-y\|_1+\frac{1}{2}\|u-x\|_2^2.



        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm.


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`\forw{x}`.
        :param deepinv.physics.Physics physics: physics model.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float gamma: stepsize of the proximity operator.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :return: (:class:`torch.Tensor`) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        norm_AtA = physics.compute_sqnorm(x)
        stepsize = 1.0 / norm_AtA if stepsize is None else stepsize
        u = x.clone()
        for it in range(max_iter):
            u_prev = u.clone()

            t = x - physics.A_adjoint(u)
            u_ = u + stepsize * physics.A(t)
            u = u_ - stepsize * self.d.prox(u_ / stepsize, y, gamma / stepsize)
            rel_crit = ((u - u_prev).norm()) / (u.norm() + 1e-12)
            print(rel_crit)
            if rel_crit < crit_conv and it > 2:
                break
        return t


class AmplitudeLoss(DataFidelity):
    r"""
    Amplitude loss as the data fidelity term for :func:`deepinv.physics.PhaseRetrieval` reconstrunction.

    In this case, the data fidelity term is defined as

    .. math::

        f(x) = \sum_{i=1}^{m}{(\sqrt{|b_i x|^2}-\sqrt{y_i})^2},

    where :math:`b_i` is the i-th row of the linear operator :math:`B` of the phase retrieval class and :math:`y_i` is the i-th entry of the measurements, and :math:`m` is the number of measurements.

    """

    def __init__(self):
        super().__init__()
        self.d = AmplitudeLossDistance()


class LogPoissonLikelihood(DataFidelity):
    r"""
    Log-Poisson negative log-likelihood.

    .. math::

        \datafid{z}{y} =  N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)

    Corresponds to :class:`deepinv.physics.LogPoissonNoise` with the same arguments :math:`N_0` and :math:`\mu`.
    There is no closed-form of the proximal operator known.

    :param float N0: average number of photons
    :param float mu: normalization constant
    """

    def __init__(self, N0: float = 1024.0, mu: float = 1 / 50.0):
        super().__init__()
        self.d = LogPoissonLikelihoodDistance(N0=N0, mu=mu)
        self.mu = mu
        self.N0 = N0


class ZeroFidelity(DataFidelity):
    r"""
    Zero data fidelity term :math:`\datafid{x}{y} = 0`.
    This is used to remove the data fidelity term in the loss function.

    """

    def __init__(self):
        super().__init__()
        self.d = ZeroDistance()

    def fn(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        """
        This function returns zero for all inputs.
        """
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        """
        This function returns a zero image.
        """
        return torch.zeros_like(x)

    def grad_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        This function returns a zero image.
        """
        return torch.zeros_like(u)

    def prox_d(self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        This function returns the input image.
        """
        return u

    def prox_d_conjugate(
        self, u: torch.Tensor, y: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        This function returns the input image.
        """
        return u
