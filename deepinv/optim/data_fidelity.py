import torch
import torch.nn as nn

from deepinv.optim.utils import gradient_descent


class DataFidelity(nn.Module):
    r"""
    Data fidelity term :math:`\datafid{Ax}{y}`.

    This is the base class for the data fidelity term :math:`f(x) = \datafid{A(x)}{y}` where :math:`A` is a linear or nonlinear operator,
    :math:`x\in\xset` is a variable  and :math:`y\in\yset` is the observation.
    ::Warning:: All variables have as first dimension the size of the batch.

    TODO : change the example in the docstring

    ::

        # define a loss function
        data_fidelity = L2()

        # create a measurement operator
        A = torch.Tensor([[2, 0], [0, 0.5]])
        A_forward = lambda v:A@v
        A_adjoint = lambda v: A.transpose(0,1)@v

        # Define the physics model associated to this operator
        physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

        # Define two points
        x = torch.Tensor([1, 4])
        y = torch.Tensor([1, 1])

        # Compute the loss :math:`f(x) = \datafid{A(x)}{y}`
        f = data_fidelity(x, y, physics)  # print f gives 1.0

        # Compute the gradient of :math:`f`
        grad = data_fidelity.grad(x, y, physics)  # print grad gives [2.0000, 0.5000]

        # Compute the proximity operator of :math:`f`
        prox = data_fidelity.prox(x, y, physics, gamma=1.0)  # print prox_fA gives [0.6000, 3.6000]


    :param callable d: data fidelity distance :math:`\datafid{u}{y}`. Outputs a tensor of size `B` the size of the batch. Default: None.
    """

    def __init__(self, d=None):
        super().__init__()
        self._d = d

    def d(self, u, y, *args, **kwargs):
        r"""
        Computes the data fidelity distance :math:`\datafid{u}{y}`.

        :param torch.tensor u: Variable :math:`u` at which the data fidelity is computed.
        :param torch.tensor y: Data :math:`y`.
        :return: (torch.tensor) data fidelity :math:`\datafid{u}{y}`.
        """
        return self._d(u - y, *args, **kwargs)

    def grad_d(self, u, y, *args, **kwargs):
        r"""
        Computes the gradient :math:`\nabla_u\datafid{u}{y}`, computed in :math:`u`. Note that this is the gradient of
        :math:`\datafid` and not :math:`f`. By default, the gradient is computed using automatic differentiation.

        :param torch.tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (torch.tensor) gradient of :math:`\datafid` in :math:`u`, i.e. :math:`\nabla_u\datafid{u}{y}`.
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
        gamma,
        *args,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs
    ):
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\gamma\datafid{\cdot}{y}}(u)`, computed in :math:`u`. Note
        that this is the proximity operator of :math:`\datafid` and not :math:`f`. By default, the proximity operator is computed using internal gradient descent.

        :param torch.tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma\datafid{\cdot}{y}}(u)`.
        """
        grad = lambda z: gamma * self.grad_d(z, y, *args, **kwargs) + (z - u)
        return gradient_descent(
            grad, u, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def forward(self, x, y, physics, *args, **kwargs):
        r"""
        Computes the data fidelity term :math:`f(x) = \datafid{Ax}{y}`.

        :param torch.tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.tensor) data fidelity :math:`\datafid{Ax}{y}`.
        """
        return self.d(physics.A(x), y, *args, **kwargs)

    def grad(self, x, y, physics, *args, **kwargs):
        r"""
        Calculates the gradient of the data fidelity term :math:`f` at :math:`x`.

        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.tensor) gradient :math:`\nabla_x\datafid{Ax}{y}`, computed in :math:`x`.
        """
        return physics.A_adjoint(self.grad_d(physics.A(x), y, *args, **kwargs))

    def prox(
        self,
        x,
        y,
        physics,
        gamma,
        *args,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs
    ):
        r"""
        Calculates the proximity operator of :math:`f` at :math:`x`.

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma f}(x)`, computed in :math:`x`.
        """
        grad = lambda z: gamma * self.grad(z, y, *args, **kwargs) + (z - x)
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def prox_conjugate(self, x, y, physics, gamma, *args, lamb=1, **kwargs):
        r"""
        Calculates the proximity operator of the convex conjugate :math:`(\lambda f)^*` at :math:`x`, using the Moreau formula.

        ::Warning:: Only valid for convex :math:`f`

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`f`
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma (\lambda f)^*}(x)`, computed in :math:`x`.
        """
        return x - gamma * self.prox(
            x / gamma, y, physics, lamb / gamma, *args, **kwargs
        )


class L2(DataFidelity):
    r"""
    Implementation of :math:`f` as the normalized :math:`\ell_2` norm

    .. math::

        f(x) = \frac{1}{2\sigma^2}\|Ax-y\|^2

    It can be used to define a log-likelihood function associated with additive Gaussian noise
    by setting an appropriate noise level :math:`\sigma`.

    :param float sigma: Standard deviation of the noise to be used as a normalisation factor.


    ::

        # define a loss function
        loss = L2()

        # create a measurement operator
        A = torch.Tensor([[2, 0], [0, 0.5]])
        A_forward = lambda v:A@v
        A_adjoint = lambda v: A.transpose(0,1)@v

        # Define the physics model associated to this operator
        physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

        # Define two points
        x = torch.Tensor([1, 4])
        y = torch.Tensor([1, 1])

        # Compute the loss f(Ax, y)
        f = loss(x, y, physics)  # print f gives 1.0

        # Compute the gradient of f
        grad_dA = data_fidelity.grad(x, y, physics)  # print grad_d gives [2.0000, 0.5000]

        # Compute the proximity operator of f
        prox_dA = data_fidelity.prox(x, y, physics, gamma=1.0)  # print prox_dA gives [0.6000, 3.6000]
    """

    def __init__(self, sigma=1.0):
        super().__init__()

        self.norm = 1 / (sigma**2)

    def d(self, u, y):
        r"""
        Computes the data fidelity distance :math:`\datafid{u}{y}`, i.e.

        .. math::

            \datafid{u}{y} = \frac{1}{2\sigma^2}\|u-y\|^2


        :param torch.tensor u: Variable :math:`u` at which the data fidelity is computed.
        :param torch.tensor y: Data :math:`y`.
        :return: (torch.tensor) data fidelity :math:`\datafid{u}{y}` of size `B` with `B` the size of the batch.
        """
        x = u - y
        d = 0.5 * torch.norm(x.view(x.shape[0], -1), p=2, dim=-1) ** 2
        return d

    def grad_d(self, u, y):
        r"""
        Computes the gradient of :math:`\datafid`  :math:`\nabla_{u}\datafid{u}{y}`, i.e.

        .. math::

            \nabla_{u}\datafid{u}{y} = \frac{1}{\sigma^2}(u-y)


        :param torch.tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y`.
        :return: (torch.tensor) gradient of the data fidelity :math:`\nabla_{u}\datafid{u}{y}`.
        """
        return self.norm * (u - y)

    def prox_d(self, x, y, gamma):
        r"""
        Proximal operator of :math:`\gamma \datafid(x) = \frac{\gamma}{2\sigma^2}\|x-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \datafid}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \datafid} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|u-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y`.
        :param float gamma: thresholding parameter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma \datafid}(x)`.
        """
        gamma_ = self.norm * gamma
        return (x + gamma_ * y) / (1 + gamma_)

    def prox(self, x, y, physics, gamma):
        r"""
        Proximal operator of :math:`\gamma f(x) = \frac{\gamma}{2\sigma^2}\|Ax-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma f}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma f} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|Au-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma f}(x)`.
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


    ::

        # define a loss function
        loss = IndicatorL2(radius=0.5)

        # create a measurement operator
        A = torch.Tensor([[2, 0], [0, 0.5]])
        A_forward = lambda v:A@v
        A_adjoint = lambda v: A.transpose(0,1)@v

        # Define the physics model associated to this operator
        physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

        # Define two points
        x = torch.Tensor([1, 4])
        y = torch.Tensor([1, 1])

        # Compute the loss f(Ax, y)
        f = loss(x, y, physics)  # print f gives 1e16

    """

    def __init__(self, radius=None):
        super().__init__()
        self.radius = radius

    def d(self, u, y, radius=None):
        r"""
        Computes the batched indicator of :math:`\ell_2` ball with radius `radius`, i.e. :math:`\iota_{\mathcal{B}(y,r)}(u)`.

        :param torch.tensor u: Variable :math:`u` at which the indicator is computed. :math:`u` is assumed to be of shape (B, ...) where B is the batch size.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :param float radius: radius of the :math:`\ell_2` ball. If `radius` is None, the radius of the ball is set to `self.radius`. Default: None.
        :return: (torch.tensor) indicator of :math:`\ell_2` ball with radius `radius`. If the point is inside the ball, the output is 0, else it is 1e16.
        """
        diff = u - y
        dist = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=-1)
        radius = self.radius if radius is None else radius
        loss = (dist > radius) * 1e16
        return loss

    def prox_d(self, x, y, gamma=None, radius=None):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\iota_{\mathcal{B}_2(y,r)}}(x) = \operatorname{proj}_{\mathcal{B}_2(y, r)}(x)


        where :math:`\operatorname{proj}_{C}(x)` denotes the projection on the closed convex set :math:`C`.


        ::

            # define a loss function
            loss = IndicatorL2(radius=1)

            # Define two points
            x = torch.Tensor([3, 3])
            y = torch.Tensor([1, 1])

            # Compute the proximity operator f(x, y)
            prox_d = loss.prox_d(x, y)  # print prox_d gives [1.707, 1.707]

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: step-size. Note that this parameter is not used in this function.
        :param float radius: radius of the :math:`\ell_2` ball.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius
        return y + torch.min(
            torch.tensor([radius]).to(x.device), torch.norm(x.flatten() - y.flatten())
        ) * (x - y) / (torch.norm(x.flatten() - y.flatten()) + 1e-12)

    def prox(
        self, x, y, physics, radius=None, stepsize=None, crit_conv=1e-5, max_iter=100
    ):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\iota_{\mathcal{B}_2(y, r)}(A\cdot)}(x) = \underset{u}{\text{argmin}} \,\, \iota_{\mathcal{B}_2(y, r)}(Au)+\frac{1}{2}\|u-x\|_2^2

        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm,
        as suggested in `Proximal Splitting Methods in Signal Processing <https://arxiv.org/pdf/0912.3522.pdf>`_.

        ::

            # Define a loss function
            data_fidelity = IndicatorL2(radius=0.5)

            # Define two points
            x = torch.Tensor([1, 4])
            y = torch.Tensor([1, 1])

            # Define a measurement operator
            A = torch.Tensor([[2, 0], [0, 0.5]])
            A_forward = lambda v:A@v
            A_adjoint = lambda v: A.transpose(0,1)@v
            physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

            # Compute the proximity operator
            projected_point = data_fidelity.prox(x, y, physics)  # print gives [0.5290, 2.9917]


        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`A(x)`.
        :param torch.tensor radius: radius of the :math:`\ell_2` ball.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius
        norm_AtA = physics.compute_norm(x)
        stepsize = 1.0 / norm_AtA if stepsize is None else stepsize
        u = x.clone()
        for it in range(max_iter):
            u_prev = u.clone()

            t = x - physics.A_adjoint(u)
            u_ = u + stepsize * physics.A(t)
            u = u_ - stepsize * self.prox_d(u_ / stepsize, y, radius=radius)
            rel_crit = ((u - u_prev).norm()) / (u.norm() + 1e-12)
            print(rel_crit)
            if rel_crit < crit_conv and it > 2:
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

    def prox_d(self, x, y, gamma):
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
    Implementation of :math:`\datafid` as the :math:`\ell_1` norm

    .. math::

        f(x) = \|Ax-y\|_1

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


        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :return: (torch.tensor) gradient of the :math:`\ell_1` norm at `x`.
        """
        return torch.sign(x - y)

    def prox_d(self, u, y, gamma):
        r"""
        Proximal operator of the :math:`\ell_1` norm, i.e.

        .. math::

            \operatorname{prox}_{\gamma \ell_1}(x) = \underset{z}{\text{argmin}} \,\, \gamma \|z-y\|_1+\frac{1}{2}\|z-x\|_2^2


        also known as the soft-thresholding operator.

        :param torch.tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: stepsize (or soft-thresholding parameter).
        :return: (torch.tensor) soft-thresholding of `u` with parameter `gamma`.
        """
        d = u - y
        aux = torch.sign(d) * torch.maximum(
            d.abs() - gamma, torch.tensor([0]).to(d.device)
        )
        return aux + y

    def prox(self, x, y, physics, gamma, stepsize=None, crit_conv=1e-5, max_iter=100):
        r"""
        Proximal operator of the :math:`\ell_1` norm composed with A, i.e.

        .. math::

            \operatorname{prox}_{\gamma \ell_1}(x) = \underset{u}{\text{argmin}} \,\, \gamma \|Au-y\|_1+\frac{1}{2}\|u-x\|_2^2.



        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm.


        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`A(x)`.
        :param deepinv.physics.Physics physics: physics model.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
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
