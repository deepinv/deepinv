import torch
import torch.nn as nn
import numpy as np
from deepinv.loss.loss import Loss


def hutch_div(y, physics, f, mc_iter=1):
    r"""
    Hutch divergence for A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :return: (float) hutch divergence.
    """
    input = y.requires_grad_(True)
    output = physics.A(f(input, physics))
    out = 0
    for i in range(mc_iter):
        b = torch.randn_like(input)
        x = torch.autograd.grad(output, input, b, retain_graph=True, create_graph=True)[
            0
        ]
        out += (b * x).reshape(y.size(0), -1).mean(1)

    return out / mc_iter


def exact_div(y, physics, model):
    r"""
    Exact divergence for A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module model: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :return: (float) exact divergence.
    """
    input = y.requires_grad_(True)
    output = physics.A(model(input, physics))
    out = 0
    _, c, h, w = input.shape
    for i in range(c):
        for j in range(h):
            for k in range(w):
                b = torch.zeros_like(input)
                b[:, i, j, k] = 1
                x = torch.autograd.grad(
                    output, input, b, retain_graph=True, create_graph=True
                )[0]
                out += (b * x).sum()

    return out / (c * h * w)


def mc_div(y1, y, f, physics, tau):
    r"""
    Monte-Carlo estimation for the divergence of A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :return: (float) hutch divergence.
    """
    b = torch.randn_like(y)
    y2 = physics.A(f(y + b * tau, physics))
    out = (b * (y2 - y1) / tau).reshape(y.size(0), -1).mean(1)
    return out


class SureGaussianLoss(Loss):
    r"""
    SURE loss for Gaussian noise


    The loss is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|y - A\inverse{y}\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(A\inverse{y+\tau b_i} -
        A\inverse{y}\right)

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`, :math:`A` is the forward operator,
    :math:`b\sim\mathcal{N}(0,I)` and :math:`\tau\geq 0` is a hyperparameter controlling the
    Monte Carlo approximation of the divergence.

    This loss approximates the divergence of :math:`A\inverse{y}` (in the original SURE loss)
    using the Monte Carlo approximation in
    https://ieeexplore.ieee.org/abstract/document/4099398/

    If the measurement data is truly Gaussian with standard deviation :math:`\sigma`,
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-A\inverse{y}\|_2^2`
    where :math:`z` is the noiseless measurement.

    .. warning::

        The loss can be sensitive to the choice of :math:`\tau`, which should be proportional to the size of :math:`y`.
        The default value of 0.01 is adapted to :math:`y` vectors with entries in :math:`[0,1]`.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    """

    def __init__(self, sigma, tau=1e-2):
        super(SureGaussianLoss, self).__init__()
        self.name = "SureGaussian"
        self.sigma2 = sigma**2
        self.tau = tau

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the SURE Loss.

        :param torch.Tensor y: Measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction network.
        :return: torch.nn.Tensor loss of size (batch_size,)
        """

        y1 = physics.A(x_net)
        div = 2 * self.sigma2 * mc_div(y1, y, model, physics, self.tau)
        mse = (y1 - y).pow(2).reshape(y.size(0), -1).mean(1)
        loss_sure = mse + div - self.sigma2
        return loss_sure


class SurePoissonLoss(Loss):
    r"""
    SURE loss for Poisson noise

    The loss is designed for the following noise model:

    .. math::

      y = \gamma z \quad \text{with}\quad z\sim \mathcal{P}(\frac{u}{\gamma}), \quad u=A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|y-A\inverse{y}\|_2^2-\frac{\gamma}{m} 1^{\top}y
        +\frac{2\gamma}{m\tau}(b\odot y)^{\top} \left(A\inverse{y+\tau b}-A\inverse{y}\right)

    where :math:`R` is the trainable network, :math:`y` is the noisy measurement vector of size :math:`m`,
    :math:`b` is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
    :math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

    See https://ieeexplore.ieee.org/abstract/document/6714502/ for details.
    If the measurement data is truly Poisson
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-A\inverse{y}\|_2^2`
    where :math:`z` is the noiseless measurement.

    .. warning::

        The loss can be sensitive to the choice of :math:`\tau`, which should be proportional to the size of :math:`y`.
        The default value of 0.01 is adapted to :math:`y` vectors with entries in :math:`[0,1]`.

    :param float gain: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    """

    def __init__(self, gain, tau=1e-3):
        super(SurePoissonLoss, self).__init__()
        self.name = "SurePoisson"
        self.gain = gain
        self.tau = tau

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the SURE loss.

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param torch.nn.Module model: Reconstruction network
        :return: torch.nn.Tensor loss of size (batch_size,)
        """

        # generate a random vector b
        b = torch.rand_like(y) > 0.5
        b = (2 * b - 1) * 1.0  # binary [-1, 1]

        y1 = physics.A(x_net)
        y2 = physics.A(model(y + self.tau * b, physics))

        # compute m (size of y)
        # m = y.numel() #(torch.abs(y) > 1e-5).flatten().sum()

        loss_sure = (
            (y1 - y).pow(2)
            - self.gain * y
            + 2.0 / self.tau * (b * y * self.gain * (y2 - y1))
        )
        loss_sure = loss_sure.reshape(y.size(0), -1).mean(1)

        return loss_sure


class SurePGLoss(Loss):
    r"""
    SURE loss for Poisson-Gaussian noise

    The loss is designed for the following noise model:

    .. math::

        y = \gamma z + \epsilon

    where :math:`u = A(x)`, :math:`z \sim \mathcal{P}\left(\frac{u}{\gamma}\right)`,
    and :math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I)`.

    The loss is computed as

    .. math::

        & \frac{1}{m}\|y-A\inverse{y}\|_2^2-\frac{\gamma}{m} 1^{\top}y-\sigma^2
        +\frac{2}{m\tau_1}(b\odot (\gamma y + \sigma^2 I))^{\top} \left(A\inverse{y+\tau b}-A\inverse{y} \right) \\\\
        & +\frac{2\gamma \sigma^2}{m\tau_2^2}c^{\top} \left( A\inverse{y+\tau c} + A\inverse{y-\tau c} - 2A\inverse{y} \right)

    where :math:`R` is the trainable network, :math:`y` is the noisy measurement vector,
    :math:`b` is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
    :math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

    If the measurement data is truly Poisson-Gaussian
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-A\inverse{y}\|_2^2`
    where :math:`z` is the noiseless measurement.

    See https://ieeexplore.ieee.org/abstract/document/6714502/ for details.

    .. warning::

        The loss can be sensitive to the choice of :math:`\tau`, which should be proportional to the size of :math:`y`.
        The default value of 0.01 is adapted to :math:`y` vectors with entries in :math:`[0,1]`.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float gamma: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    """

    def __init__(self, sigma, gain, tau1=1e-3, tau2=1e-2):
        super(SurePGLoss, self).__init__()
        self.name = "SurePG"
        # self.sure_loss_weight = sure_loss_weight
        self.sigma2 = sigma**2
        self.gain = gain
        self.tau1 = tau1
        self.tau2 = tau2

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the SURE loss.

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param torch.nn.Module f: Reconstruction network
        :return: torch.nn.Tensor loss of size (batch_size,)
        """

        b1 = torch.rand_like(y) > 0.5
        b1 = (2 * b1 - 1) * 1.0  # binary [-1, 1]

        p = 0.7236  # .5 + .5*np.sqrt(1/5.)

        b2 = torch.ones_like(b1) * np.sqrt(p / (1 - p))
        b2[torch.rand_like(b2) < p] = -np.sqrt((1 - p) / p)

        meas1 = physics.A(x_net)
        meas2 = physics.A(model(y + self.tau1 * b1, physics))
        meas2p = physics.A(model(y + self.tau2 * b2, physics))
        meas2n = physics.A(model(y - self.tau2 * b2, physics))

        # compute m (size of y)
        # m = (torch.abs(y) > 1e-5).flatten().sum()

        loss_mc = (meas1 - y).pow(2).reshape(y.size(0), -1).mean(1)

        loss_div1 = (
            2
            / self.tau1
            * ((b1 * (self.gain * y + self.sigma2)) * (meas2 - meas1))
            .reshape(y.size(0), -1)
            .mean(1)
        )

        offset = -self.gain * y.reshape(y.size(0), -1).mean(1) - self.sigma2

        loss_div2 = (
            -2
            * self.sigma2
            * self.gain
            / (self.tau2**2)
            * (b2 * (meas2p + meas2n - 2 * meas1)).reshape(y.size(0), -1).mean(1)
        )

        loss_sure = loss_mc + loss_div1 + loss_div2 + offset
        return loss_sure
