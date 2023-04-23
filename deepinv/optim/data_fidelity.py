import torch
import torch.nn as nn


class DataFidelity(nn.Module):
    r"""
    Data fidelity term :math:`\datafid{Ax}{y}`.

    This is the base class for the data fidelity term

    .. math:

        \datafid{Ax}{y}


    where :math:`A` is a linear operator, :math:`x` is a variable in :math: and :math:`y` is the data, and where
    :math:`f` is a convex function.

    """

    def __init__(self, f=None, grad_f=None, prox_f=None, prox_norm=None):
        super().__init__()
        self._grad_f = grad_f  # TODO: use autograd?
        self._f = f
        self._prox_f = prox_f
        self._prox_norm = prox_norm

    def f(self, x, y):
        r"""
        Computes the data fidelity :math:`\datafid{x}{y}`.

        :param torch.tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.tensor y: Data :math:`y`.
        :return: (torch.tensor) data fidelity :math:`\datafid{x}{y}`.
        """
        return self._f(x)

    def grad_f(self, u, y):
        r"""
        Computes the gradient :math:`\nabla_u\datafid{u}{y}`, computed in :math:`u`. Note that this is the gradient of
        :math:`f` and not :math:`f\circ A`.

        :param torch.tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :return: (torch.tensor) gradient of :math:`f` in :math:`u`, i.e. :math:`\nabla_u\datafid{u}{y}`.
        """
        return self._grad_f(u, y)

    def prox_f(self, u, y, gamma):
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\datafid{.}{y}}(u)`, computed in :math:`u`.Note
        that this is the proximity operator of :math:`f` and not :math:`f\circ A`.

        :param torch.tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :param float gamma: step-size.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma f(.,y)}(u)`.
        """
        return self._prox_f(u, y, gamma)

    def forward(self, x, y, physics):
        r"""
        Computes the data fidelity :math:`\datafid{Ax}{y}`.

        :param torch.tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.tensor) data fidelity :math:`\datafid{Ax}{y}`.
        """
        Ax = physics.A(x)
        return self.f(Ax, y)

    def grad(self, x, y, physics):
        r"""
        Computes the gradient of the data fidelity :math:`\nabla_x\datafid{Ax}{y}`, computed in :math:`x`.

        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.tensor) gradient :math:`\nabla_x\datafid{Ax}{y}`, computed in :math:`x`.
        """
        Ax = physics.A(x)
        if self.grad_f is not None:
            return physics.A_adjoint(self.grad_f(Ax, y))
        else:
            raise ValueError("No gradient defined for this data fidelity term.")

    def prox(self, x, y, physics, gamma):
        r"""
        Computes the proximity operator :math:`\operatorname{prox}_{\datafid{A.}{y}}(x)`,
        computed in :math:`x`.

        :param torch.tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param float gamma: step-size.
        :return: (torch.tensor) gradient :math:`\operatorname{prox}_{\datafid{A.}{y}}(x)`, computed in :math:`x`.
        """
        if "Denoising" in physics.__class__.__name__:
            return self.prox_f(y, x, gamma)  # TODO: clarify
        else:
            raise Exception(
                "no prox operator is implemented for the data fidelity term."
            )

    # DEPRECATED
    # def prox_norm(self, x, y, gamma):
    #     return self.prox_norm(x, y, gamma)


class L2(DataFidelity):
    r"""
    Implementation of :math:`f` in :meth:`deepinv.optim.DataFidelity` as the normalized :math:`\ell_2` norm:

    .. math::

        f(x) = \frac{1}{2\sigma^2}\|x-y\|^2

    It can be used to define a log-likelihood function associated with additive Gaussian noise
    by setting an appropriate noise level :math:`\sigma`.

    :param float sigma: Standard deviation of the noise to be used as a normalisation factor.
    """

    def __init__(self, sigma=1.0):
        super().__init__()

        self.norm = 1 / (sigma**2)

    def f(self, x, y):
        return self.norm * (x - y).flatten().pow(2).sum() / 2

    def grad_f(self, x, y):
        return self.norm * (x - y)

    def prox(
        self, x, y, physics, gamma
    ):
        return physics.prox_l2(x, y, self.norm * gamma)

    def prox_f(self, x, y, gamma):
        r"""
        computes the proximal operator of

        .. math::

            f(x) = \frac{1}{2\sigma^2}\gamma\|x-y\|_2^2

        """
        gamma_ = self.norm * gamma
        return (x + gamma_ * y) / (1 + gamma_)


class IndicatorL2(DataFidelity):
    r"""
    Indicator of :math:`\ell_2` ball with radius :math:`r`.

    """

    def __init__(self, radius=None):
        super().__init__()
        self.radius = radius

    def f(self, x, y, radius=None):
        r"""
        Computes the indicator of :math:`\ell_2` ball with radius `radius`, i.e. :math:`\iota_{\mathcal{B}(y,r)}(x)`

        ..:math::

            \iota_{\mathcal{B}(y,r)}(x) = \begin{cases}

        """
        dist = (x - y).flatten().pow(2).sum().sqrt()
        radius = self.radius if radius is None else radius
        loss = 0 if dist < radius else 1e16
        return loss

    def prox_f(self, x, y, gamma=None, radius=None):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`.

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: step-size. Note that this parameter is not used in this function.
        :param float radius: radius of the :math:`\ell_2` ball.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        if radius is None:
            radius = self.radius
        return y + torch.min(
            torch.tensor([radius]), torch.linalg.norm(x.flatten() - y.flatten())
        ) * (x - y) / (torch.linalg.norm(x - y) + 1e-6)

    def prox(self, x, y, physics, radius=None, stepsize=None, crit_conv=1e-5, max_iter=100):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`.

        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm.
        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`A(x)`.
        :param torch.tensor radius: radius of the :math:`\ell_2` ball.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius
        norm_A = physics.compute_norm(x)
        stepsize = 1.0 / norm_A if stepsize is None else stepsize
        u = x.clone()
        for it in range(max_iter):
            u_prev = u.clone()

            t = x - physics.A_adjoint(u)
            u_ = u + stepsize * physics.A(t)
            u = u_ - stepsize * self.prox_f(u_ / stepsize, y, radius=radius)
            rel_crit = ((u - u_prev).norm()) / (u.norm() + 1e-12)
            if rel_crit < crit_conv and it > 10:
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

    def f(self, x, y):
        if self.normalize:
            y = y * self.gain
        return (-y * torch.log(self.gain * x + self.bkg)).flatten().sum() + (
            self.gain * x
        ).flatten().sum()

    def grad_f(self, x, y):
        if self.normalize:
            y = y * self.gain
        return (1 / self.gain) * (torch.ones_like(x) - y / (self.gain * x + self.bkg))

    def prox_f(self, x, y, gamma):
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
    :math:`\ell_1` fidelity.

    """

    def __init__(self):
        super().__init__()

    def f(self, x, y):
        return (x - y).flatten().abs().sum()

    def grad_f(self, x, y):
        return torch.sign(x - y)

    def prox_f(self, x, y, gamma):
        # soft thresholding
        d = x - y
        aux = torch.sign(d) * torch.max(d.abs() - gamma, 0)
        return aux + y
