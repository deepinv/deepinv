import torch
import torch.nn as nn


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

        y \sim\mathcal{N}(u,\sigma^2I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \|y - Af(y)\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(Af(y+\tau b_i) - Af(y)\right)

    where :math:`f` is the trainable network, :math:`y` is the noisy measurement vector and :math:`\text{div}`.

    This loss approximates the divergence of :math:`Af(y)` (in the original SURE loss)
    using the Monte Carlo approximation in
    https://ieeexplore.ieee.org/abstract/document/4099398/

    If the measurement data is truly Gaussian with standard deviation :math:`\sigma`,
    this loss is an unbiased estimator of :math:`\|u-Af(y)\|_2^2`
    where :math:`z` is the noiseless measurement.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.

    '''
    def __init__(self, sigma, tau=1e-2):
        super(SureGaussianLoss, self).__init__()
        self.sigma2 = sigma ** 2
        self.tau = tau

    # TODO: leave denoising as default
    def forward(self, y, physics, f):
        r'''
        :param torch.tensor y: Measurements
        :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :return: (float) SURE loss.
        '''
        # compute loss_sure
        y1 = physics.A(f(y))
        div = mc_div(y, y1, lambda x: physics.A(f(x, physics)), self.tau)
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

        \|y-Af(y)\|_2^2-\gamma 1^{\top}y
        +\frac{2\gamma}{\tau}(b\odot y)^{\top} \left(Af(y+\tau b)-Af(y)\right)

    where :math:`f` is the trainable network, :math:`y` is the noisy measurement vector,
    :math:`b` to be a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
    :math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

    If the measurement data is truly Poisson
    this loss is an unbiased estimator of :math:`\|u-Af(y)\|_2^2`
    where :math:`z` is the noiseless measurement.

    :param float gamma: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    '''
    def __init__(self, gamma, tau=1e-2):
        super(SurePoissonLoss, self).__init__()
        self.name = 'SurePoisson'
        self.gamma = gamma
        self.tau = tau

    def forward(self, y, f, physics):
        '''
        :param torch.tensor y: Measurements
        :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :return: (float) SURE loss.
        '''
        # generate a random vector b

        b = torch.rand_like(y) > 0.5
        # b = torch.rand_like(self.physics.A_dagger(y0)) > 0.5

        b = (2 * b.int() - 1) * 1.0  # binary [-1, 1]
        b = physics.A(b * 1.0)

        y1 = physics.A(f(y))
        y2 = physics.A(f(self.physics.A_dagger(y + self.tau * b)))

        # compute m (size of y)
        m = (torch.abs(y) > 1e-5).flatten().sum()

        loss_sure = torch.sum((y1 - y).pow(2)) / m \
                    - self.gamma * y.sum() / m \
                    + 2 * self.gamma / (self.tau * m) * ((b * y) * (y2 - y1)).sum()
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

        & \|y-Af(y)\|_2^2-\gamma 1^{\top}y-\sigma^2
        +\frac{2}{\tau}(b\odot (\gamma y + \sigma^2 I))^{\top} \left(Af(y+\tau b)-Af(y) \right) \\\\
        & +\frac{2\gamma \sigma^2}{\tau}c^{\top} \left( Af(y+\tau c) + Af(y-\tau c) - 2Af(y) \right)

    where :math:`f` is the trainable network, :math:`y` is the noisy measurement vector,
    :math:`b` to be a Bernoulli random variable taking values of -1 and 1 each with a probability of 0.5,
    :math:`\tau` is a small positive number, and :math:`\odot` is an elementwise multiplication.

    If the measurement data is truly Poisson-Gaussian
    this loss is an unbiased estimator of :math:`\|u-Af(y)\|_2^2`
    where :math:`z` is the noiseless measurement.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float gamma: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    '''
    def __init__(self, sigma, gamma, tau=1e-2):
        super(SurePGLoss, self).__init__()
        self.name = 'sure'
        # self.sure_loss_weight = sure_loss_weight
        self.sigma = sigma
        self.gamma = gamma
        self.tau = tau

    def forward(self, y, f, physics):
        r'''

        :param torch.tensor y: Measurements
        :param torch.nn.Module, deepinv.models.Denoiser f: Reconstruction network
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :return: (float) SURE loss.
        '''
        sigma2 = self.sigma ** 2
        b1 = torch.randn_like(y)
        b2 = torch.rand_like(y) > 0.5
        b2 = (2 * b2.int() - 1) * 1.0  # binary [-1, 1]

        meas1 = physics.A(f(y))
        meas2 = physics.A(f(y + self.tau * b1))
        meas2p = physics.A(f(y + self.tau * b2))
        meas2n = physics.A(f(y - self.tau * b2))

        # compute m (size of y)
        m = (torch.abs(y) > 1e-5).flatten().sum()

        loss_A = torch.sum((meas1 - y).pow(2)) / m - sigma2
        loss_div1 = 2 / (self.tau * m) * ((b1 * (self.gamma * y + sigma2)) * (meas2 - meas1)).sum()
        loss_div2 = 2 * sigma2 * self.gamma / (self.tau ** 2 * m) \
                    * (b2 * (meas2p + meas2n - 2 * meas1)).sum()

        loss_sure = loss_A + loss_div1 + loss_div2
        return loss_sure



# test code
if __name__ == "__main__":
    device = 'cuda:0'
    #TODO test SURE