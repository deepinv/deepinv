import torch
from deepinv.optim.potential import Potential
from deepinv.utils.tensorlist import TensorList


class Distance(Potential):
    r"""
    Distance :math:`\distance{x}{y}`.

    This is the base class for a distance :math:`\distance{x}{y}` between a variable :math:`x` and an observation :math:`y`.
    Comes with methods to compute the distance gradient, proximal operator or convex conjugate with respect to the variable :math:`x`.

    .. warning::
        All variables have a batch dimension as first dimension.

    :param Callable d: distance function :math:`\distance{x}{y}`. Outputs a tensor of size `B`, the size of the batch. Default: None.
    """

    def __init__(self, d=None):
        super().__init__(fn=d)

    def fn(self, x, y, *args, **kwargs):
        r"""
        Computes the distance :math:`\distance{x}{y}`.

        :param torch.Tensor x: Variable :math:`x`.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (:class:`torch.Tensor`) distance :math:`\distance{x}{y}` of size `B` with `B` the size of the batch.
        """
        return self._fn(x, y, *args, **kwargs)

    def forward(self, x, y, *args, **kwargs):
        r"""
        Computes the value of the distance :math:`\distance{x}{y}`.

        :param torch.Tensor x: Variable :math:`x`.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (:class:`torch.Tensor`) distance :math:`\distance{x}{y}` of size `B` with `B` the size of the batch.
        """
        return self.fn(x, y, *args, **kwargs)


class L2Distance(Distance):
    r"""
    Implementation of :math:`\distancename` as the normalized :math:`\ell_2` norm

    .. math::
        f(x) = \frac{1}{2\sigma^2}\|x-y\|^2

    :param float sigma: normalization parameter. Default: 1.
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.norm = 1 / (sigma**2)

    def fn(self, x, y, *args, **kwargs):
        r"""
        Computes the distance :math:`\distance{x}{y}` i.e.

        .. math::

            \distance{x}{y} = \frac{1}{2}\|x-y\|^2


        :param torch.Tensor u: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{u}{y}` of size `B` with `B` the size of the batch.
        """
        z = x - y
        d = 0.5 * torch.norm(z.reshape(z.shape[0], -1), p=2, dim=-1) ** 2 * self.norm
        return d

    def grad(self, x, y, *args, **kwargs):
        r"""
        Computes the gradient of :math:`\distancename`, that is  :math:`\nabla_{x}\distance{x}{y}`, i.e.

        .. math::

            \nabla_{x}\distance{x}{y} = \frac{1}{\sigma^2} x-y


        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (:class:`torch.Tensor`) gradient of the distance function :math:`\nabla_{x}\distance{x}{y}`.
        """
        return (x - y) * self.norm

    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of :math:`\gamma \distance{x}{y} = \frac{\gamma}{2 \sigma^2} \|x-y\|^2`.

        Computes :math:`\operatorname{prox}_{\gamma \distancename}`, i.e.

        .. math::

           \operatorname{prox}_{\gamma \distancename} = \underset{u}{\text{argmin}} \frac{\gamma}{2\sigma^2}\|u-y\|_2^2+\frac{1}{2}\|u-x\|_2^2


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param float gamma: thresholding parameter.
        :return: (:class:`torch.Tensor`) proximity operator :math:`\operatorname{prox}_{\gamma \distancename}(x)`.
        """
        return (x + self.norm * gamma * y) / (1 + gamma * self.norm)


class IndicatorL2Distance(Distance):
    r"""
    Indicator of :math:`\ell_2` ball with radius :math:`r`.

    The indicator function of the $\ell_2$ ball with radius :math:`r`, denoted as \iota_{\mathcal{B}_2(y,r)(x)},
    is defined as

    .. math::

          \iota_{\mathcal{B}_2(y,r)}(x)= \left.
              \begin{cases}
                0, & \text{if } \|x-y\|_2\leq r \\
                +\infty & \text{else.}
              \end{cases}
              \right.


    :param float radius: radius of the ball. Default: None.
    """

    def __init__(self, radius=None):
        super().__init__()
        self.radius = radius

    def fn(self, x, y, *args, radius=None, **kwargs):
        r"""
        Computes the batched indicator of :math:`\ell_2` ball with radius `radius`, i.e. :math:`\iota_{\mathcal{B}(y,r)}(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the indicator is computed. :math:`u` is assumed to be of shape (B, ...) where B is the batch size.
        :param torch.Tensor y: Observation :math:`y` of the same dimension as :math:`u`.
        :param float radius: radius of the :math:`\ell_2` ball. If `radius` is None, the radius of the ball is set to `self.radius`. Default: None.
        :return: (:class:`torch.Tensor`) indicator of :math:`\ell_2` ball with radius `radius`. If the point is inside the ball, the output is 0, else it is 1e16.
        """
        diff = x - y
        dist = torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=-1)
        radius = self.radius if radius is None else radius
        loss = (dist > radius) * 1e16
        return loss

    def prox(self, x, y, *args, radius=None, gamma=None, **kwargs):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\iota_{\mathcal{B}_2(y,r)}}(x) = \operatorname{proj}_{\mathcal{B}_2(y, r)}(x)


        where :math:`\operatorname{proj}_{C}(x)` denotes the projection on the closed convex set :math:`C`.


        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.Tensor y: Observation :math:`y` of the same dimension as :math:`x`.
        :param float gamma: step-size. Note that this parameter is not used in this function.
        :param float radius: radius of the :math:`\ell_2` ball.
        :return: (:class:`torch.Tensor`) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius
        diff = x - y
        dist = torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=-1)
        return y + diff * (
            torch.min(torch.tensor([radius]).to(x.device), dist) / (dist + 1e-12)
        ).view(-1, 1, 1, 1)


class PoissonLikelihoodDistance(Distance):
    r"""
    (Negative) Log-likelihood of the Poisson distribution.

    .. math::

        \distance{y}{x} =  \sum_i y_i \log(y_i / x_i) + x_i - y_i


    .. note::

        The function is not Lipschitz smooth w.r.t. :math:`x` in the absence of background (:math:`\beta=0`).

    :param float gain: gain of the measurement :math:`y`. Default: 1.0.
    :param float bkg: background level :math:`\beta`. Default: 0.
    :param bool denormalize: if True, the measurement is divided by the gain. By default, in the
        :class:`deepinv.physics.PoissonNoise`, the measurements are multiplied by the gain after being sampled by
        the Poisson distribution. Default: True.
    """

    def __init__(self, gain=1.0, bkg=0, denormalize=False):
        super().__init__()
        self.bkg = bkg
        self.gain = gain
        self.denormalize = denormalize

    def fn(self, x, y, *args, **kwargs):
        r"""
        Computes the Kullback-Leibler divergence

        :param torch.Tensor x: Variable :math:`x` at which the distance is computed.
        :param torch.Tensor y: Observation :math:`y`.
        """
        if self.denormalize:
            y = y / self.gain
        return (-y * torch.log(x / self.gain + self.bkg)).flatten().sum() + (
            (x / self.gain) + self.bkg - y
        ).reshape(x.shape[0], -1).sum(dim=1)

    def grad(self, x, y, *args, **kwargs):
        r"""
        Gradient of the Kullback-Leibler divergence

        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        """
        if self.denormalize:
            y = y / self.gain
        return self.gain * (torch.ones_like(x) - y / (x / self.gain + self.bkg))

    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of the Kullback-Leibler divergence

        :param torch.Tensor x: signal :math:`x` at which the function is computed.
        :param torch.Tensor y: measurement :math:`y`.
        :param float gamma: proximity operator step size.
        """
        if self.denormalize:
            y = y / self.gain
        out = (
            x
            - (1 / (self.gain * gamma))
            * ((x - (1 / (self.gain * gamma))).pow(2) + 4 * y / gamma).sqrt()
        )
        return out / 2


class L1Distance(Distance):
    r"""
    :math:`\ell_1` distance

    .. math::

        f(x) = \|x-y\|_1.

    """

    def __init__(self):
        super().__init__()

    def fn(self, x, y, *args, **kwargs):
        diff = x - y
        return torch.norm(diff.reshape(diff.shape[0], -1), p=1, dim=-1)

    def grad(self, x, y, *args, **kwargs):
        r"""
        Gradient of the gradient of the :math:`\ell_1` norm, i.e.

        .. math::

            \partial \datafid(x) = \operatorname{sign}(x-y)


        .. note::

            The gradient is not defined at :math:`x=y`.


        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :return: (:class:`torch.Tensor`) gradient of the :math:`\ell_1` norm at `x`.
        """
        return torch.sign(x - y)

    def prox(self, u, y, *args, gamma=1.0, **kwargs):
        r"""
        Proximal operator of the :math:`\ell_1` norm, i.e.

        .. math::

            \operatorname{prox}_{\gamma \ell_1}(x) = \underset{z}{\text{argmin}} \,\, \gamma \|z-y\|_1+\frac{1}{2}\|z-x\|_2^2


        also known as the soft-thresholding operator.

        :param torch.Tensor u: Variable :math:`u` at which the proximity operator is computed.
        :param torch.Tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: stepsize (or soft-thresholding parameter).
        :return: (:class:`torch.Tensor`) soft-thresholding of `u` with parameter `gamma`.
        """
        d = u - y
        aux = torch.sign(d) * torch.maximum(
            d.abs() - gamma, torch.tensor([0]).to(d.device)
        )
        return aux + y


class AmplitudeLossDistance(Distance):
    r"""
    Amplitude loss for :class:`deepinv.physics.PhaseRetrieval` reconstruction, defined as

    .. math::

        f(x) = \sum_{i=1}^{m}{(\sqrt{|y_i - x|^2}-\sqrt{y_i})^2},

    where :math:`y_i` is the i-th entry of the measurements, and :math:`m` is the number of measurements.

    """

    def __init__(self):
        super().__init__()

    def fn(self, u, y, *args, **kwargs):
        r"""
        Computes the amplitude loss.

        :param torch.Tensor u: estimated measurements.
        :param torch.Tensor y: true measurements.
        :return: (:class:`torch.Tensor`) the amplitude loss of shape B where B is the batch size.
        """
        x = torch.sqrt(u) - torch.sqrt(y)
        d = torch.norm(x.reshape(x.shape[0], -1), p=2, dim=-1) ** 2
        return d

    def grad(self, u, y, *args, epsilon=1e-12, **kwargs):
        r"""
        Computes the gradient of the amplitude loss :math:`\distance{u}{y}`, i.e.,

        .. math::

            \nabla_{u}\distance{u}{y} = \frac{\sqrt{u}-\sqrt{y}}{\sqrt{u}}


        :param torch.Tensor u: Variable :math:`u` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param float epsilon: small value to avoid division by zero.
        :return: (:class:`torch.Tensor`) gradient of the amplitude loss function.
        """
        return (torch.sqrt(u + epsilon) - torch.sqrt(y)) / torch.sqrt(u + epsilon)


class LogPoissonLikelihoodDistance(Distance):
    r"""
    Log-Poisson negative log-likelihood.

    .. math::

        \distancz{z}{y} =  N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)

    Corresponds to LogPoissonNoise with the same arguments N0 and mu.
    There is no closed-form of the prox known.

    :param float N0: average number of photons
    :param float mu: normalization constant
    """

    def __init__(self, N0=1024.0, mu=1 / 50.0):
        super().__init__()
        self.mu = mu
        self.N0 = N0

    def fn(self, x, y, *args, **kwargs):
        out1 = torch.exp(-x * self.mu) * self.N0
        out2 = torch.exp(-y * self.mu) * self.N0 * (x * self.mu)
        return (out1 + out2).reshape(x.shape[0], -1).sum(dim=1)
