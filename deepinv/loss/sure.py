import torch
import torch.nn as nn
import numpy as np


def mc_div(x, y, f, tau):
    # y = f(x), avoids double computation
    # computes the divergence of f at x using a montecarlo approx.
    b = torch.randn_like(x)
    y2 = f(x + tau * b)
    div = (b * (y2 - y)).flatten().mean()/tau
    return div


class SureGaussianLoss(nn.Module):
    r'''
    SURE loss for Gaussian noise

    The loss is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|y - Af(y)\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(Af(y+\tau b_i) - Af(y)\right)

    where :math:`f` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`, :math:`A` is the forward operator,
    :math:`b\sim\mathcal{N}(0,I)` and :math:`\tau\geq 0` is a hyperparameter controlling the
    Monte Carlo approximation of the divergence.

    This loss approximates the divergence of :math:`Af(y)` (in the original SURE loss)
    using the Monte Carlo approximation in
    https://ieeexplore.ieee.org/abstract/document/4099398/

    If the measurement data is truly Gaussian with standard deviation :math:`\sigma`,
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-Af(y)\|_2^2`
    where :math:`z` is the noiseless measurement.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    '''
    def __init__(self, sigma, tau=1e-3):
        super(SureGaussianLoss, self).__init__()
        self.sigma2 = sigma ** 2
        self.tau = tau

    def forward(self, y, x_net, physics, f):
        r'''
        Computes the SURE Loss.

        :param torch.tensor y: Measurements.
        :param torch.tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network.
        :return: (float) SURE loss.
        '''

        # compute loss_sure
        y1 = physics.A(x_net)
        div = mc_div(y, y1, lambda u: physics.A(f(u, physics)), self.tau)
        loss_sure = (y1 - y).pow(2).flatten().mean() - self.sigma2\
                    + 2 * self.sigma2 * div

        return loss_sure


class SurePoissonLoss(nn.Module):
    r'''
    SURE loss for Poisson noise

    The loss is designed for the following noise model:

    .. math::

      y = \gamma z \quad \text{with}\quad z\sim \mathcal{P}(\frac{u}{\gamma}), \quad u=A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|y-Af(y)\|_2^2-\frac{\gamma}{m} 1^{\top}y
        +\frac{2\gamma}{m\tau}(b\odot y)^{\top} \left(Af(y+\tau b)-Af(y)\right)

    where :math:`f` is the trainable network, :math:`y` is the noisy measurement vector,
    :math:`b` is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
    :math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

    See https://ieeexplore.ieee.org/abstract/document/6714502/ for details.
    If the measurement data is truly Poisson
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-Af(y)\|_2^2`
    where :math:`z` is the noiseless measurement.

    :param float gain: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    '''
    def __init__(self, gain, tau=1e-3):
        super(SurePoissonLoss, self).__init__()
        self.name = 'SurePoisson'
        self.gain = gain
        self.tau = tau

    def forward(self, y, x_net, physics, f):
        r'''
        Computes the SURE loss.

        :param torch.tensor y: measurements.
        :param torch.tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network
        :return: (float) SURE loss.
        '''

        # generate a random vector b
        b = torch.rand_like(y) > 0.5
        b = (2 * b - 1) * 1.0  # binary [-1, 1]

        y1 = physics.A(x_net)
        y2 = physics.A(f(y + self.tau * b, physics))

        # compute m (size of y)
        # m = y.numel() #(torch.abs(y) > 1e-5).flatten().sum()

        loss_sure = (y1 - y).pow(2).mean() - self.gain * y.mean() \
                    + 2. * self.gain / self.tau * (b * y * (y2 - y1)).mean()

        return loss_sure


class SurePGLoss(nn.Module):
    r'''
    SURE loss for Poisson-Gaussian noise

    The loss is designed for the following noise model:

    .. math::

        y = \gamma z + \epsilon

    where :math:`u = A(x)`, :math:`z \sim \mathcal{P}\left(\frac{u}{\gamma}\right)`,
    and :math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I)`.

    The loss is computed as

    .. math::

        & \frac{1}{m}\|y-Af(y)\|_2^2-\frac{\gamma}{m} 1^{\top}y-\sigma^2
        +\frac{2}{m\tau_1}(b\odot (\gamma y + \sigma^2 I))^{\top} \left(Af(y+\tau b)-Af(y) \right) \\\\
        & +\frac{2\gamma \sigma^2}{m\tau_2^2}c^{\top} \left( Af(y+\tau c) + Af(y-\tau c) - 2Af(y) \right)

    where :math:`f` is the trainable network, :math:`y` is the noisy measurement vector,
    :math:`b` is a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
    :math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

    If the measurement data is truly Poisson-Gaussian
    this loss is an unbiased estimator of the mean squared loss :math:`\frac{1}{m}\|u-Af(y)\|_2^2`
    where :math:`z` is the noiseless measurement.

    See https://ieeexplore.ieee.org/abstract/document/6714502/ for details.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float gamma: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    '''
    def __init__(self, sigma, gain, tau1=1e-3, tau2=1e-2):
        super(SurePGLoss, self).__init__()
        self.name = 'sure'
        # self.sure_loss_weight = sure_loss_weight
        self.sigma2 = sigma ** 2
        self.gain = gain
        self.tau1 = tau1
        self.tau2 = tau2

    def forward(self, y, x_net, physics, f):
        r'''
        Computes the SURE loss.

        :param torch.tensor y: measurements.
        :param torch.tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network
        :return: (float) SURE loss.
        '''

        b1 = torch.rand_like(y) > 0.5
        b1 = (2 * b1 - 1) * 1.0  # binary [-1, 1]

        p = 0.7236  #.5 + .5*np.sqrt(1/5.)

        b2 = torch.ones_like(b1)*np.sqrt(p/(1-p))
        b2[torch.rand_like(b2) < p] = -np.sqrt((1-p)/p)

        meas1 = physics.A(x_net)
        meas2 = physics.A(f(y + self.tau1 * b1, physics))
        meas2p = physics.A(f(y + self.tau2 * b2, physics))
        meas2n = physics.A(f(y - self.tau2 * b2, physics))

        # compute m (size of y)
        #m = (torch.abs(y) > 1e-5).flatten().sum()

        loss_mc = (meas1 - y).pow(2).mean()

        loss_div1 = 2 / self.tau1 * ((b1 * (self.gain * y + self.sigma2)) * (meas2 - meas1)).mean()

        offset = - self.gain * y.mean() - self.sigma2

        loss_div2 = - 2 * self.sigma2 * self.gain / (self.tau2 ** 2) * (b2 * (meas2p + meas2n - 2 * meas1)).mean()

        loss_sure = loss_mc + loss_div1 + loss_div2 + offset
        return loss_sure

