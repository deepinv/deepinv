import torch
import numpy as np
from deepinv.loss.loss import Loss


class ScoreLoss(Loss):
    r"""
    Approximates score of a distribution.

    It approximates the score of the measurement distribution :math:`S(y)\approx \nabla \log p(y)`
    https://proceedings.neurips.cc/paper_files/paper/2021/file/077b83af57538aa183971a2fe0971ec1-Paper.pdf.

    The score loss is defined as

    .. math::

        \| \epsilon - \sigma S(y+ \sigma \epsilon) \|^2

    where :math:`\epsilon` is sampled from :math:`N(0,I)` and
    :math:`\sigma` is sampled from :math:`N(0,I\delta^2)`.

    :param float delta: hyperparameter :math:`\delta` controlling the level of noise.
    """

    def __init__(self, delta):
        super(ScoreLoss, self).__init__()
        self.name = "score"
        self.metric = torch.nn.MSELoss()
        self.delta = delta

    def forward(self, y, model, **kwargs):
        r"""
        Computes the Score loss.

        :param torch.Tensor y: measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (torch.Tensor) loss.
        """
        std = np.randn() * self.delta
        noise = torch.randn_like(y)
        return self.metric(noise, std * model(std * noise + y))
