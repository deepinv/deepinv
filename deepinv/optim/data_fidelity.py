import torch
import torch.nn as nn

from deepinv.optim.utils import gradient_descent


class DataFidelity(nn.Module):
    r"""
    Data fidelity term :math:`\datafid{x}{y}=\distance{\forw{x}}{y}`.

    This is the base class for the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}` where :math:`A` is a
    linear or nonlinear operator, :math:`x\in\xset` is a variable , :math:`y\in\yset` is the observation and
    :math:`\distancename` is a distance function.

    .. doctest::

        >>> import torch
        >>> import deepinv as dinv
        >>> # define a loss function
        >>> data_fidelity = dinv.optim.L2()
        >>>
        >>> # Create a measurement operator
        >>> A = torch.Tensor([[2, 0], [0, 0.5]])
        >>> A_forward = lambda v: A @ v
        >>> A_adjoint = lambda v: A.transpose(0, 1) @ v
        >>>
        >>> # Define the physics model associated to this operator
        >>> physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
        >>>
        >>> # Define two points
        >>> x = torch.Tensor([[1], [4]]).unsqueeze(0)
        >>> y = torch.Tensor([[1], [1]]).unsqueeze(0)
        >>>
        >>> # Compute the loss :math:`f(x) = \datafid{A(x)}{y}`
        >>> data_fidelity(x, y, physics)
        tensor([1.0000])
        >>> # Compute the gradient of :math:`f`
        >>> grad = data_fidelity.grad(x, y, physics)
        >>>
        >>> # Compute the proximity operator of :math:`f`
        >>> prox = data_fidelity.prox(x, y, physics, gamma=1.0)

    .. warning::
        All variables have a batch dimension as first dimension.

    :param callable d: data fidelity distance function :math:`\distance{u}{y}`. Outputs a tensor of size `B`, the size of the batch. Default: None.
    """

    def __init__(self, d=None):
        super().__init__()
        self._d = d

    def d(self, u, y, *args, **kwargs):
        r"""
        Computes the data fidelity distance :math:`\distance{u}{y}`.

        :param torch.Tensor u: Variable :math:`u` at which the distance function is computed.
        :param torch.Tensor y: Data :math:`y`.
        :return: (torch.Tensor) data fidelity :math:`\distance{u}{y}`.
        """
        return self._d(u, y, *args, **kwargs)

    def grad_d(self, u, y, *args, **kwargs):
        r"""
        Computes the gradient :math:`\nabla_u\distance{u}{y}`, computed in :math:`u`. Note that this is the gradient of
        :math:`\distancename` and not :math:`\datafidname`. By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (torch.Tensor) gradient of :math:`d` in :math:`u`, i.e. :math:`\nabla_u\distance{u}{y}`.
        """
        with torch.enable_grad():
            u = u.requires_grad_()
            grad = torch.autograd.grad(
                self.d(u, y, *args, **kwargs), u, create_graph=True, only_inputs=True
            )[0]
        return grad

    def prox_d(
        self,
        u,
        y,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\gamma\distance{\cdot}{y}}(u)`, computed in :math:`u`. Note
        that this is the proximity operator of :math:`\distancename` and not :math:`\datafidname`. By default, the proximity operator is computed using internal gradient descent.

        :param torch.Tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.Tensor) proximity operator :math:`\operatorname{prox}_{\gamma\distance{\cdot}{y}}(u)`.
        """
        grad = lambda z: gamma * self.grad_d(z, y, *args, **kwargs) + (z - u)
        return gradient_descent(
            grad, u, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def forward(self, x, y, physics, *args, **kwargs):
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.Tensor) data fidelity :math:`\datafid{x}{y}`.
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
        :return: (torch.Tensor) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return physics.A_vjp(x, self.grad_d(physics.A(x), y, *args, **kwargs))

    def prox(
        self,
        x,
        y,
        physics,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Calculates the proximity operator of :math:`\datafidname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.Tensor) proximity operator :math:`\operatorname{prox}_{\gamma \datafidname}(x)`, computed in :math:`x`.
        """
        grad = lambda z: gamma * self.grad(z, y, physics, *args, **kwargs) + (z - x)
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def prox_conjugate(self, x, y, physics, *args, gamma=1.0, lamb=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the convex conjugate :math:`(\lambda \datafidname)^*` at :math:`x`,
        using the Moreau formula.

        .. warning::

            This function is only valid for convex :math:`\datafidname`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`f`
        :return: (torch.Tensor) proximity operator :math:`\operatorname{prox}_{\gamma (\lambda \datafidname)^*}(x)`,
            computed in :math:`x`.
        """
        return x - gamma * self.prox(
            x / gamma, y, physics, *args, gamma=lamb / gamma, **kwargs
        )

    def prox_d_conjugate(self, u, y, *args, gamma=1.0, lamb=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the convex conjugate :math:`(\lambda \distancename)^*` at :math:`u`,
        using the Moreau formula.

        .. warning::

            This function is only valid for convex :math:`\distancename`.

        :param torch.Tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`\distancename`
        :return: (torch.Tensor) proximity operator :math:`\operatorname{prox}_{\gamma (\lambda \distancename)^*}(x)`,
            computed in :math:`x`.
        """
        return u - gamma * self.prox_d(
            u / gamma, y, *args, gamma=lamb / gamma, **kwargs
        )


class L2(DataFidelity):
    r"""
    Implementation of :math:`\distancename` as the normalized :math:`\ell_2` norm

    .. math::

        f(x) = \frac{1}{2\sigma^2}\|\forw{x}-y\|^2

    It can be used to define a log-likelihood function associated with additive Gaussian noise
    by setting an appropriate noise level :math:`\sigma`.

    :param float sigma: Standard deviation of the noise to be used as a normalisation factor.


    .. doctest::

        >>> import torch
        >>> import deepinv as dinv
        >>> # define a loss function
        >>> fidelity = dinv.optim.L2()
        >>>
        >>> x = torch.ones(1, 1, 3, 3)
        >>> mask = torch.ones_like(x)
        >>> mask[0, 0, 1, 1] = 0
        >>> physics = dinv.physics.Inpainting(tensor_size=(1, 1, 3, 3), mask = mask)
        >>> y = physics(x)
        >>>
        >>> # Compute the data fidelity f(Ax, y)
        >>> fidelity(x, y, physics)
        tensor([0.])
        >>> # Compute the gradient of f
        >>> fidelity.grad(x, y, physics)
        tensor([[[[[0., 0., 0.],
                   [0., 0., 0.],
                   [0., 0., 0.]]]]])
        >>> # Compute the proximity operator of f
        >>> fidelity.prox(x, y, physics, gamma=1.0)
        tensor([[[[[1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.]]]]])
    """

    def __init__(self, sigma=1.0):
        super().__init__()

        self.norm = 1 / (sigma**2)

    def d(self, u, y):
        r"""
        Computes the data fidelity distance :math:`\datafid{u}{y}`, i.e.

        .. math::

            \datafid{u}{y} = \frac{1}{2\sigma^2}\|u-y\|^2


        :param torch.Tensor u: Variable :math:`u` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :return: (torch.Tensor) data fidelity :math:`\datafid{u}{y}` of size `B` with `B` the size of the batch.
        """
        x = u - y
        d = 0.5 * torch.norm(x.view(x.shape[0], -1), p=2, dim=-1) ** 2
        return self.norm * d

    def grad_d(self, u, y):
        r"""
        Computes the gradient of :math:`\distancename`, that is  :math:`\nabla_{u}\distance{u}{y}`, i.e.

        .. math::

            \nabla_{u}\distance{u}{y} = \frac{1}{\sigma^2}(u-y)


        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :return: (torch.Tensor) gradient of the distance function :math:`\nabla_{u}\distance{u}{y}`.
        """
        return self.norm * (u - y)

    def prox_d(self, x, y, gamma=1.0):
        r"""
        Proximal operator of :math:`\gamma \distance{x}{y} = \frac{\gamma}{2\sigma^2}\|x-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \distancename}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \distancename} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|u-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param float gamma: thresholding parameter.
        :return: (torch.Tensor) proximity operator :math:`\operatorname{prox}_{\gamma \distancename}(x)`.
        """
        gamma_ = self.norm * gamma
        return (x + gamma_ * y) / (1 + gamma_)

    def prox(self, x, y, physics, gamma=1.0):
        r"""
        Proximal operator of :math:`\gamma \datafid{Ax}{y} = \frac{\gamma}{2\sigma^2}\|Ax-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \datafidname}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \datafidname} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Au-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator :math:`\operatorname{prox}_{\gamma \datafidname}(x)`.
        """
        return physics.prox_l2(x, y, self.norm * gamma)


class IndicatorL2(DataFidelity):
    r"""
    Indicator of :math:`\ell_2` ball with radius :math:`r`.

    The indicator function of the $\ell_2$ ball with radius :math:`r`, denoted as \iota_{\mathcal{B}_2(y,r)(u)},
    is defined as

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
        self.radius = radius

    def d(self, u, y, radius=None):
        r"""
        Computes the batched indicator of :math:`\ell_2` ball with radius `radius`, i.e. :math:`\iota_{\mathcal{B}(y,r)}(u)`.

        :param torch.Tensor u: Variable :math:`u` at which the indicator is computed. :math:`u` is assumed to be of shape (B, ...) where B is the batch size.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :param float radius: radius of the :math:`\ell_2` ball. If `radius` is None, the radius of the ball is set to `self.radius`. Default: None.
        :return: (torch.Tensor) indicator of :math:`\ell_2` ball with radius `radius`. If the point is inside the ball, the output is 0, else it is 1e16.
        """
        diff = u - y
        dist = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=-1)
        radius = self.radius if radius is None else radius
        loss = (dist > radius) * 1e16
        return loss

    def prox_d(self, x, y, radius=None, gamma=None):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\iota_{\mathcal{B}_2(y,r)}}(x) = \operatorname{proj}_{\mathcal{B}_2(y, r)}(x)


        where :math:`\operatorname{proj}_{C}(x)` denotes the projection on the closed convex set :math:`C`.


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: step-size. Note that this parameter is not used in this function.
        :param float radius: radius of the :math:`\ell_2` ball.
        :return: (torch.Tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius
        diff = x - y
        dist = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=-1)
        return y + diff * (
            torch.min(torch.tensor([radius]).to(x.device), dist) / (dist + 1e-12)
        ).view(-1, 1, 1, 1)

    def prox(
        self,
        x,
        y,
        physics,
        radius=None,
        stepsize=None,
        crit_conv=1e-5,
        max_iter=100,
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
        :return: (torch.Tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius

        if physics.A(x).shape == x.shape and (physics.A(x) == x).all():  # Identity case
            return self.prox_d(x, y, gamma=None, radius=radius)
        else:
            norm_AtA = physics.compute_norm(x, verbose=False)
            stepsize = 1.0 / norm_AtA if stepsize is None else stepsize
            u = physics.A(x)
            for it in range(max_iter):
                u_prev = u.clone()

                t = x - physics.A_adjoint(u)
                u_ = u + stepsize * physics.A(t)
                u = u_ - stepsize * self.prox_d(
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

    :param float bkg: background level :math:`\beta`.
    """

    def __init__(self, gain=1.0, bkg=0, normalize=True):
        super().__init__()
        self.bkg = bkg
        self.gain = gain
        self.normalize = normalize

    def d(self, x, y):
        if self.normalize:
            y = y * self.gain
        return (-y * torch.log(self.gain * x + self.bkg)).flatten().sum() + (
            self.gain * x
        ).flatten().sum()

    def grad_d(self, x, y):
        if self.normalize:
            y = y * self.gain
        return (1 / self.gain) * (torch.ones_like(x) - y / (self.gain * x + self.bkg))

    def prox_d(self, x, y, gamma=1.0):
        if self.normalize:
            y = y * self.gain
        out = (
            x
            - (self.gain / gamma)
            * ((x - self.gain / gamma).pow(2) + 4 * y / gamma).sqrt()
        )
        return out / 2


class L1(DataFidelity):
    r"""
    :math:`\ell_1` data fidelity term.

    In this case, the data fidelity term is defined as

    .. math::

        f(x) = \|Ax-y\|_1.

    """

    def __init__(self):
        super().__init__()

    def d(self, x, y):
        diff = x - y
        return torch.norm(diff.view(diff.shape[0], -1), p=1, dim=-1)

    def grad_d(self, x, y):
        r"""
        Gradient of the gradient of the :math:`\ell_1` norm, i.e.

        .. math::

            \partial \datafid(x) = \operatorname{sign}(x-y)


        .. note::

            The gradient is not defined at :math:`x=y`.


        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :return: (torch.Tensor) gradient of the :math:`\ell_1` norm at `x`.
        """
        return torch.sign(x - y)

    def prox_d(self, u, y, gamma=1.0):
        r"""
        Proximal operator of the :math:`\ell_1` norm, i.e.

        .. math::

            \operatorname{prox}_{\gamma \ell_1}(x) = \underset{z}{\text{argmin}} \,\, \gamma \|z-y\|_1+\frac{1}{2}\|z-x\|_2^2


        also known as the soft-thresholding operator.

        :param torch.Tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: stepsize (or soft-thresholding parameter).
        :return: (torch.Tensor) soft-thresholding of `u` with parameter `gamma`.
        """
        d = u - y
        aux = torch.sign(d) * torch.maximum(
            d.abs() - gamma, torch.tensor([0]).to(d.device)
        )
        return aux + y

    def prox(
        self, x, y, physics, gamma=1.0, stepsize=None, crit_conv=1e-5, max_iter=100
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
        :return: (torch.Tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        norm_AtA = physics.compute_norm(x)
        stepsize = 1.0 / norm_AtA if stepsize is None else stepsize
        u = x.clone()
        for it in range(max_iter):
            u_prev = u.clone()

            t = x - physics.A_adjoint(u)
            u_ = u + stepsize * physics.A(t)
            u = u_ - stepsize * self.prox_d(u_ / stepsize, y, gamma / stepsize)
            rel_crit = ((u - u_prev).norm()) / (u.norm() + 1e-12)
            print(rel_crit)
            if rel_crit < crit_conv and it > 2:
                break
        return t


class AmplitudeLoss(DataFidelity):
    r"""
    Amplitude loss as the data fidelity term for :meth:`deepinv.physics.PhaseRetrieval` reconstrunction.

    In this case, the data fidelity term is defined as

    .. math::

        f(x) = \sum_{i=1}^{m}{(\sqrt{|b_i x|^2}-\sqrt{y_i})^2},

    where :math:`b_i` is the i-th row of the linear operator :math:`B` of the phase retrieval class and :math:`y_i` is the i-th entry of the measurements, and :math:`m` is the number of measurements.

    """

    def __init__(self):
        super().__init__()

    def d(self, u, y):
        r"""
        Computes the amplitude loss.

        :param torch.Tensor u: estimated measurements.
        :param torch.Tensor y: true measurements.
        :return: (torch.Tensor) the amplitude loss of shape B where B is the batch size.
        """
        x = torch.sqrt(u) - torch.sqrt(y)
        d = torch.norm(x.view(x.shape[0], -1), p=2, dim=-1) ** 2
        return d

    def grad_d(self, u, y, epsilon=1e-12):
        r"""
        Computes the gradient of the amplitude loss :math:`\distance{u}{y}`, i.e.,

        .. math::

            \nabla_{u}\distance{u}{y} = \frac{\sqrt{u}-\sqrt{y}}{\sqrt{u}}


        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param float epsilon: small value to avoid division by zero.
        :return: (torch.Tensor) gradient of the amplitude loss function.
        """
        return (torch.sqrt(u + epsilon) - torch.sqrt(y)) / torch.sqrt(u + epsilon)


class LogPoissonLikelihood(DataFidelity):
    r"""
    Log-Poisson negative log-likelihood.

    .. math::

        \datafid{z}{y} =  N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)

    Corresponds to LogPoissonNoise with the same arguments N0 and mu.
    There is no closed-form of prox_d known.

    :param float N0: average number of photons
    :param float mu: normalization constant
    """

    def __init__(self, N0=1024.0, mu=1 / 50.0):
        super().__init__()
        self.mu = mu
        self.N0 = N0

    def d(self, x, y):
        out1 = torch.exp(-x * self.mu) * self.N0
        out2 = torch.exp(-y * self.mu) * self.N0 * (x * self.mu)
        return (out1 + out2).sum()


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
