from deepinv.optim.distance import (
    Distance,
    L2Distance,
    L1Distance,
    IndicatorL2Distance,
    AmplitudeLossDistance,
    PoissonLikelihoodDistance,
    LogPoissonLikelihoodDistance,
)
from deepinv.optim.potential import Potential
import torch


class DataFidelity(Potential):
    r"""
    Base class for the data fidelity term :math:`\distance{A(x)}{y}` where :math:`A` is the forward operator,
    :math:`x\in\xset` is a variable and :math:`y\in\yset` is the data, and where :math:`d` is a distance function,
    from the class :class:`deepinv.optim.Distance`.

    :param Callable d: distance function :math:`d(x, y)` between a variable :math:`x` and an observation :math:`y`. Default None.
    """

    def __init__(self, d=None):
        super().__init__()
        self.d = Distance(d=d)

    def fn(self, x, y, physics, *args, **kwargs):
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """
        return self.d(physics.A(x), y, *args, **kwargs)

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
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return physics.A_vjp(x, self.d.grad(physics.A(x), y, *args, **kwargs))

    def grad_d(self, u, y, *args, **kwargs):
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

    def prox_d(self, u, y, *args, **kwargs):
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

    def prox_d_conjugate(self, u, y, *args, **kwargs):
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

    def __init__(self, data_fidelity_list):
        super(StackedPhysicsDataFidelity, self).__init__()
        self.data_fidelity_list = data_fidelity_list

    def fn(self, x, y, physics, *args, **kwargs):
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

    def grad(self, x, y, physics, *args, **kwargs):
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

    def grad_d(self, u, y, *args, **kwargs):
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

    def prox_d(self, u, y, *args, **kwargs):
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

    def prox_d_conjugate(self, u, y, *args, **kwargs):
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
        >>> physics = dinv.physics.Inpainting(tensor_size=(1, 3, 3), mask=mask)
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

    def __init__(self, sigma=1.0):
        super().__init__()
        self.d = L2Distance(sigma=sigma)
        self.norm = 1 / (sigma**2)

    def prox(self, x, y, physics, *args, gamma=1.0, **kwargs):
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

    def __init__(self, radius=None):
        super().__init__()
        self.d = IndicatorL2Distance(radius=radius)
        self.radius = radius

    def prox(
        self,
        x,
        y,
        physics,
        *args,
        radius=None,
        stepsize=None,
        crit_conv=1e-5,
        max_iter=100,
        **kwargs,
    ):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\gamma \iota_{\mathcal{B}_2(y, r)}(A\cdot)}(x) = \underset{u}{\text{argmin}} \,\, \iota_{\mathcal{B}_2(y, r)}(Au)+\frac{1}{2}\|u-x\|_2^2

        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm,
        as suggested in `Proximal Splitting Methods in Signal Processing <https://arxiv.org/pdf/0912.3522.pdf>`_.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`\forw{x}`.
        :param torch.Tensor radius: radius of the :math:`\ell_2` ball.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :param float gamma: factor in front of the indicator function. Notice that this does not affect the proximity
                            operator since the indicator is scale invariant. Default: None.
        :return: (:class:`torch.Tensor`) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius

        if physics.A(x).shape == x.shape and (physics.A(x) == x).all():  # Identity case
            return self.d.prox(x, y, gamma=None, radius=radius)
        else:
            norm_AtA = physics.compute_norm(x, verbose=False)
            stepsize = 1.0 / norm_AtA if stepsize is None else stepsize
            u = physics.A(x)
            for it in range(max_iter):
                u_prev = u.clone()

                t = x - physics.A_adjoint(u)
                u_ = u + stepsize * physics.A(t)
                u = u_ - stepsize * self.d.prox(
                    u_ / stepsize, y, radius=radius, gamma=None
                )
                rel_crit = ((u - u_prev).norm()) / (u.norm() + 1e-12)
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

    def __init__(self, gain=1.0, bkg=0, denormalize=True):
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
        x,
        y,
        physics,
        *args,
        gamma=1.0,
        stepsize=None,
        crit_conv=1e-5,
        max_iter=100,
        **kwargs,
    ):
        r"""
        Proximal operator of the :math:`\ell_1` norm composed with A, i.e.

        .. math::

            \operatorname{prox}_{\gamma \ell_1}(x) = \underset{u}{\text{argmin}} \,\, \gamma \|Au-y\|_1+\frac{1}{2}\|u-x\|_2^2.



        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm.


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`\forw{x}`.
        :param deepinv.physics.Physics physics: physics model.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :return: (:class:`torch.Tensor`) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        norm_AtA = physics.compute_norm(x)
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

    Corresponds to LogPoissonNoise with the same arguments N0 and mu.
    There is no closed-form of the proximal operator known.

    :param float N0: average number of photons
    :param float mu: normalization constant
    """

    def __init__(self, N0=1024.0, mu=1 / 50.0):
        super().__init__()
        self.d = LogPoissonLikelihoodDistance(N0=N0, mu=mu)
        self.mu = mu
        self.N0 = N0


if __name__ == "__main__":
    import deepinv as dinv

    # define a loss function
    data_fidelity = L2()

    # create a measurement operator dxd
    A = torch.Tensor([[2, 0], [0, 0.5]])
    A_forward = lambda v: torch.matmul(A, v)
    A_adjoint = lambda v: torch.matmul(A.transpose(0, 1), v)

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

    # Define two points of size Bxd
    x = torch.Tensor([1, 4]).unsqueeze(0).repeat(4, 1).unsqueeze(-1)
    y = torch.Tensor([1, 1]).unsqueeze(0).repeat(4, 1).unsqueeze(-1)

    # Compute the loss :math:`f(x) = \datafid{A(x)}{y}`
    f = data_fidelity(x, y, physics)  # print f gives 1.0
    # Compute the gradient of :math:`f`
    grad = data_fidelity.grad(x, y, physics)  # print grad_f gives [2.0000, 0.5000]

    # Compute the proximity operator of :math:`f`
    prox = data_fidelity.prox(
        x, y, physics, gamma=1.0
    )  # print prox_fA gives [0.6000, 3.6000]
