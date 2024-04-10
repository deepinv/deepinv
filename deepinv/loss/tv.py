import torch
from deepinv.loss.loss import Loss


class TVLoss(Loss):
    r"""
    Total variation loss (:math:`\ell_2` norm).

    It computes the loss :math:`\|D\hat{x}\|_2^2`,
    where :math:`D` is a normalized linear operator that computes the vertical and horizontal first order differences
    of the reconstructed image :math:`\hat{x}`.

    :param float weight: scalar weight for the TV loss.
    """

    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = weight
        self.name = "tv"

    def forward(self, x_net, **kwargs):
        r"""
        Computes the TV loss.

        :param torch.Tensor x_net: reconstructed image.
        :return: torch.nn.Tensor loss of size (batch_size,)
        """
        batch_size = x_net.size()[0]
        h_x = x_net.size()[2]
        w_x = x_net.size()[3]
        count_h = self.tensor_size(x_net[:, :, 1:, :])
        count_w = self.tensor_size(x_net[:, :, :, 1:])
        h_tv = (
            torch.pow((x_net[:, :, 1:, :] - x_net[:, :, : h_x - 1, :]), 2)
            .reshape(x_net.size(0), -1)
            .sum(1)
        )
        w_tv = (
            torch.pow((x_net[:, :, :, 1:] - x_net[:, :, :, : w_x - 1]), 2)
            .reshape(x_net.size(0), -1)
            .sum(1)
        )
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w)

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
