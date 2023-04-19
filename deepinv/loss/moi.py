import torch
import torch.nn as nn
import numpy as np


class MOILoss(nn.Module):
    r"""
    Multi-operator imaging loss

    This loss can be used to learn when signals are observed via multiple (possibly incomplete)
    forward operators :math:`\{A_g\}_{g=1}^{G}`,
    i.e., :math:`y_i = A_{g_i}x_i` where :math:`g_i\in \{1,\dots,G\}` (see https://arxiv.org/abs/2201.12151).


    The measurement consistency loss is defined as

    .. math::

        \| \hat{x} - \inverse{\hat{x},A_g} \|^2

    where :math:`\hat{x}=\inverse{y,A_s}` is a reconstructed signal (observed via operator :math:`A_s`) and
    :math:`A_g` is a forward operator sampled at random from a set :math:`\{A_g\}_{g=1}^{G}`.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float weight: total weight of the loss
    :param bool apply_noise: if ``True``, the augmented measurement is computed with the full sensing model
        :math:`\sensor{\noise{\forw{\hat{x}}}}` (i.e., noise and sensor model),
        otherwise is generated as :math:`\forw{\hat{x}}`.
    """

    def __init__(self, metric=torch.nn.MSELoss(), apply_noise=True, weight=1.0):
        super(MOILoss, self).__init__()
        self.name = "moi"
        self.metric = metric
        self.weight = weight
        self.noise = apply_noise

    def forward(self, x_net, physics, f):
        r"""
        Computes the MOI loss.

        :param torch.tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param list of deepinv.physics.Physics physics: List containing the :math:`G` different forward operators
            associated with the measurements.
        :param torch.nn.Module f: Reconstruction function.
        :return: (torch.tensor) loss.
        """
        j = np.random.randint(len(physics))

        if self.noise:
            y = physics[j](x_net)
        else:
            y = physics[j].A(x_net)

        x2 = f(y, physics[j])

        return self.weight * self.metric(x2, x_net)
