import torch
import torch.nn as nn


class TVLoss(nn.Module):
    r'''
    Total variation loss (:math:`\ell_2` norm).

    It computes the following loss

    .. math::

        \|D\hat{x}\|_2^2

    where :math:`D` is a normalized linear operator that computes the vertical and horizontal first order differences
    of the reconstructed image :math:`\hat{x}`.

    :param float weight: scalar weight for the TV loss.
    '''
    def __init__(self, weight=1.):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = weight
        self.name = 'tv'

    def forward(self, x):
        r'''
        Computes the TV loss.

        :param torch.tensor x: reconstructed image.
        :return: (torch.tensor) loss.
        '''
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

