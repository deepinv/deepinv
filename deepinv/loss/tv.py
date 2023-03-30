import torch
import torch.nn as nn

# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    r'''
    Total variation loss (:math:`\ell_2` norm).

    It computes

    .. math:

        \|Dx\|_2^2

    where :math:`D` is a normalized linear operator that computes the vertical and horizontal first order differences
    of the image:math:`x`.

    :param float tv_loss_weight: scalar weight for the TV loss.
    '''
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
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