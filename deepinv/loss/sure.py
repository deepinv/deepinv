import torch
import torch.nn as nn
import numpy as np
import deepinv.physics
from deepinv.loss.loss import Loss


def hutch_div(y, physics, f, mc_iter=1, rng=None):
    r"""
    Hutch divergence for A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :param torch.Generator rng: Random number generator. Default is None.
    :return: (float) hutch divergence.
    """
    input = y.requires_grad_(True)
    output = physics.A(f(input, physics))
    out = 0
    for i in range(mc_iter):
        b = torch.empty_like(y).normal_(generator=rng)
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


def mc_div(y1, y, f, physics, tau, precond=lambda x: x, rng: torch.Generator = None):
    r"""
    Monte-Carlo estimation for the divergence of A(f(x)).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
    :param torch.nn.Module f: Reconstruction network.
    :param int mc_iter: number of iterations. Default=1.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    :param bool pinv: If ``True``, the pseudo-inverse of the forward operator is used. Default ``False``.
    :param Callable precond: Preconditioner. Default is the identity.
    :param torch.Generator rng: Random number generator. Default is None.
    :return: (float) Ramani MC divergence.
    """
    b = torch.empty_like(y).normal_(generator=rng)
    y2 = physics.A(f(y + b * tau, physics))
    return (precond(b) * precond(y2 - y1) / tau).reshape(y.size(0), -1).mean(1)


def unsure_gradient_step(loss, param, saved_grad, init_flag, step_size, momentum):
    r"""
    Gradient step for estimating the noise level in the UNSURE loss.

    :param torch.Tensor loss: Loss value.
    :param torch.Tensor param: Parameter to optimize.
    :param torch.Tensor saved_grad: Saved gradient w.r.t. the parameter.
    :param bool init_flag: Initialization flag (first gradient step).
    :param float step_size: Step size.
    :param float momentum: Momentum.
    """
    grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
    if init_flag:
        init_flag = False
        saved_grad = grad
    else:
        saved_grad = momentum * saved_grad + (1.0 - momentum) * grad
    return param + step_size * grad, saved_grad, init_flag


class SureGaussianLoss(Loss):
    r"""
    SURE loss for Gaussian noise


    The loss is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|B(y - A\inverse{y})\|_2^2 -\sigma^2 +\frac{2\sigma^2}{m\tau}b^{\top} B^{\top} \left(A\inverse{y+\tau b_i} -
        A\inverse{y}\right)

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`, :math:`A` is the forward operator,
    :math:`B` is an optional linear mapping which should be approximately :math:`A^{\dagger}` (or any stable approximation),
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

    .. note::

        If the noise level is unknown, the loss can be adapted to the UNSURE loss introduced in https://arxiv.org/abs/2409.01985,
        which also learns the noise level.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    :param Callable, str B: Optional linear metric :math:`B`, which can be used to improve
        the performance of the loss. If 'A_dagger', the pseudo-inverse of the forward operator is used.
        Otherwise the metric should be a linear operator that approximates the pseudo-inverse of the forward operator
        such as :func:`deepinv.physics.LinearPhysics.prox_l2` with large :math:`\gamma`. By default, the identity is used.
    :param bool unsure: If ``True``, the loss is adapted to the UNSURE loss introduced in https://arxiv.org/abs/2409.01985
        where the noise level :math:`\sigma` is also learned (the input value is used as initialization).
    :param float step_size: Step size for the gradient ascent of the noise level if unsure is ``True``.
    :param float momentum: Momentum for the gradient ascent of the noise level if unsure is ``True``.
    :param torch.Generator rng: Optional random number generator. Default is None.
    """

    def __init__(
        self,
        sigma,
        tau=1e-2,
        B=lambda x: x,
        unsure=False,
        step_size=1e-4,
        momentum=0.9,
        rng: torch.Generator = None,
    ):
        super(SureGaussianLoss, self).__init__()
        self.name = "SureGaussian"
        self.sigma2 = sigma**2
        self.tau = tau
        self.metric = B
        self.unsure = unsure
        self.init_flag = False
        self.step_size = step_size
        self.momentum = momentum
        self.grad_sigma = 0.0
        self.rng = rng
        if unsure:
            self.sigma2 = torch.tensor(self.sigma2, requires_grad=True)

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the SURE Loss.

        :param torch.Tensor y: Measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction network.
        :return: torch.nn.Tensor loss of size (batch_size,)
        """

        if self.metric == "A_dagger":
            metric = lambda x: physics.A_dagger(x)
        else:
            metric = self.metric

        y1 = physics.A(x_net)
        div = (
            2 * self.sigma2 * mc_div(y1, y, model, physics, self.tau, metric, self.rng)
        )
        mse = metric(y1 - y).pow(2).reshape(y.size(0), -1).mean(1)
        loss_sure = mse + div - self.sigma2

        if self.unsure:  # update the estimate of the noise level
            self.sigma2, self.grad_sigma, self.init_flag = unsure_gradient_step(
                div.mean(),
                self.sigma2,
                self.grad_sigma,
                self.init_flag,
                self.step_size,
                self.momentum,
            )

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
    :param torch.Generator rng: Optional random number generator. Default is None.
    """

    def __init__(self, gain, tau=1e-3, rng: torch.Generator = None):
        super(SurePoissonLoss, self).__init__()
        self.name = "SurePoisson"
        self.gain = gain
        self.tau = tau
        self.rng = rng

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
        b = torch.empty_like(y).uniform_(generator=self.rng)
        b = b > 0.5
        b = (2 * b - 1) * 1.0  # binary [-1, 1]

        y1 = physics.A(x_net)
        y2 = physics.A(model(y + self.tau * b, physics))

        loss_sure = (
            (y1 - y).pow(2)
            - self.gain * y
            + (2.0 / self.tau) * self.gain * (b * y * (y2 - y1))
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

    .. note::

        If the noise levels are unknown, the loss can be adapted to the UNSURE loss introduced in https://arxiv.org/abs/2409.01985,
        which also learns the noise levels.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param float gamma: Gain of the Poisson Noise.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    :param float tau2: Approximation constant for the second derivative.
    :param bool second_derivative: If ``False``, the last term in the loss (approximating the second derivative) is removed
        to speed up computations, at the cost of a possibly inexact loss. Default ``True``.
    :param bool unsure: If ``True``, the loss is adapted to the UNSURE loss introduced in https://arxiv.org/abs/2409.01985
        where :math:`\gamma` and :math:`\sigma^2` are also learned (their input value is used as initialization).
    :param tuple[float] step_size: Step size for the gradient ascent of the noise levels if unsure is ``True``.
    :param tuple[float] momentum: Momentum for the gradient ascent of the noise levels if unsure is ``True``.
    :param torch.Generator rng: Optional random number generator. Default is None.
    """

    def __init__(
        self,
        sigma,
        gain,
        tau1=1e-3,
        tau2=1e-2,
        second_derivative=False,
        unsure=False,
        step_size=(1e-4, 1e-4),
        momentum=(0.9, 0.9),
        rng=None,
    ):
        super(SurePGLoss, self).__init__()
        self.name = "SurePG"
        # self.sure_loss_weight = sure_loss_weight
        self.sigma2 = sigma**2
        self.gain = gain
        self.tau1 = tau1
        self.tau2 = tau2
        self.second_derivative = second_derivative
        self.step_size = step_size
        self.grad_sigma = 0.0
        self.grad_gain = 0.0
        self.momentum = momentum
        self.init_flag_sigma = True
        self.init_flag_gain = True
        self.unsure = unsure
        self.rng = rng
        if unsure:
            self.sigma2 = torch.tensor(self.sigma2, requires_grad=True)
            self.gain = torch.tensor(self.gain, requires_grad=True)

    def forward(self, y, x_net, physics, model, **kwargs):
        r"""
        Computes the SURE loss.

        :param torch.Tensor y: measurements.
        :param torch.Tensor x_net: reconstructed image :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements
        :param torch.nn.Module f: Reconstruction network
        :return: torch.nn.Tensor loss of size (batch_size,)
        """

        b1 = torch.empty_like(y).uniform_(generator=self.rng)
        b1 = b1 > 0.5
        b1 = (2 * b1 - 1) * 1.0  # binary [-1, 1]

        p = 0.7236  # .5 + .5*np.sqrt(1/5.)

        b2 = torch.ones_like(b1) * np.sqrt(p / (1 - p))
        b2[torch.empty_like(y).uniform_(generator=self.rng) < p] = -np.sqrt((1 - p) / p)

        meas1 = physics.A(x_net)
        meas2 = physics.A(model(y + self.tau1 * b1, physics))

        loss_mc = (meas1 - y).pow(2).reshape(y.size(0), -1).mean(1)

        loss_div1 = (
            2
            / self.tau1
            * ((b1 * (self.gain * y + self.sigma2)) * (meas2 - meas1))
            .reshape(y.size(0), -1)
            .mean(1)
        )

        offset = -self.gain * y.reshape(y.size(0), -1).mean(1) - self.sigma2

        if self.unsure:  # update the estimate of the noise levels
            div = loss_div1.mean()
            self.sigma2, self.grad_sigma, self.init_flag_sigma = unsure_gradient_step(
                div,
                self.sigma2,
                self.grad_sigma,
                self.init_flag_sigma,
                self.step_size[0],
                self.momentum[0],
            )
            self.gain, self.grad_gain, self.init_flag_gain = unsure_gradient_step(
                div,
                self.gain,
                self.grad_gain,
                self.init_flag_gain,
                self.step_size[1],
                self.momentum[1],
            )

        if self.second_derivative:
            meas2p = physics.A(model(y + self.tau2 * b2, physics))
            meas2n = physics.A(model(y - self.tau2 * b2, physics))
            loss_div2 = (
                -2
                * self.sigma2
                * self.gain
                / (self.tau2**2)
                * (b2 * (meas2p + meas2n - 2 * meas1)).reshape(y.size(0), -1).mean(1)
            )
        else:
            loss_div2 = torch.zeros_like(loss_div1)

        loss_sure = loss_mc + loss_div1 + loss_div2 + offset
        return loss_sure
