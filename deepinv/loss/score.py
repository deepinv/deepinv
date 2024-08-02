import torch
from deepinv.loss.loss import Loss


class ScoreLoss(Loss):
    r"""
    Learns score of noise distribution.

    Approximates the score of the measurement distribution :math:`S(y)\approx \nabla \log p(y)`
    https://proceedings.neurips.cc/paper_files/paper/2021/file/077b83af57538aa183971a2fe0971ec1-Paper.pdf.

    The score loss is defined as

    .. math::

        \| \epsilon + \sigma S(y+ \sigma \epsilon) \|^2

    where :math:`y` is the noisy measurement,
    :math:`S` is the model approximating the score of the noisy measurement distribution :meth:`\nabla \log p(y)`,
    :math:`\epsilon` is sampled from :math:`N(0,I)` and
    :math:`\sigma` is sampled from :math:`N(0,I\delta^2)` with :math:`\delta` annealed during training
    from a maximum value to a minimum value.

    At test/evaluation time, the method uses Tweedie's formula to estimate the score:

    .. math::

        R(y) = y + \sigma^2 S(y)

    .. warning::

        The user should provide a backbone model :math:`S`
        to :meth:`adapt_model` which returns the full reconstruction network
        :meth:`R`, which is mandatory to compute the loss properly.

    :param float sigma: Noise level of the measurements, assumes Gaussian Noise.
    :param int total_batches: Total number of training batches (epochs * number of batches per epoch).
    :param tuple delta: Tuple of two floats representing the minimum and maximum noise level,
        which are annealed during training.

    |sep|


    :Example:

        >>> import torch
        >>> import deepinv as dinv
        >>> sigma = 0.1
        >>> physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
        >>> model = dinv.models.DnCNN(layers=1, download=False)
        >>> loss = dinv.loss.ScoreLoss(sigma=sigma)
        >>> model = loss.adapt_model(model) # important step!
        >>> x = torch.ones((1, 1, 8, 8))
        >>> y = physics(x)
        >>> x_net = model(y, physics, update_parameters=True) # save score loss in forward
        >>> l = loss(x_net, y, physics, model)
        >>> print(l.item() > 0)
        True
    """
    def __init__(self, sigma, total_batches, delta=(.001, .1)):
        super(ScoreLoss, self).__init__()
        self.total_batches = total_batches
        self.delta = delta
        self.sigma = sigma

    def forward(self, x_net, physics, model, **kwargs):
        r"""
        Computes the Score Loss.

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction model.
        :return: (torch.Tensor) Score loss.
        """
        return model.get_error()

    def adapt_model(self, model, **kwargs):
        r"""
        Transforms score backbone net :meth:`S` into :meth:`R` for training and evaluation.

        :param torch.nn.Module model: Backbone model approximating the score.
        :return: (torch.nn.Module) Adapted reconstruction model.
        """
        if isinstance(model, ScoreModel):
            return model
        else:
            return ScoreModel(model, self.sigma, self.delta, self.total_batches)


class ScoreModel(torch.nn.Module):
    def __init__(self, model, sigma, delta, total_batches):
        super(ScoreModel, self).__init__()
        self.base_model = model
        self.min = delta[0]
        self.max = delta[1]
        self.sigma = sigma
        self.counter = 0
        self.total_batches = total_batches

    def forward(self, y, physics, update_parameters=False):
        if self.training:
            self.counter += 1
            w = self.counter/self.total_batches
            delta = self.max * (1-w) + self.min * w
            sigma = torch.randn((y.size(0), 1, 1, 1), device=y.device) * delta
        else:
            sigma = self.min

        extra_noise = torch.randn_like(y)
        y_plus = y + extra_noise * sigma

        grad = self.base_model(y_plus, physics)

        if update_parameters:
            self.error = (extra_noise + grad * sigma).pow(2).mean()

        return y_plus + self.sigma**2 * grad

    def get_error(self):
        return self.error