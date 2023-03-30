import torch
import torch.nn as nn

# --------------------------------------------
# MC loss
# --------------------------------------------
class MCLoss(nn.Module):
    r'''
    Measurement consistency loss


    :param int metric: metric used for computing data consistency, which is set as the mean squared error by default.

    '''
    def __init__(self, metric=torch.nn.MSELoss()):
        super(MCLoss, self).__init__()
        self.name = 'mc'
        self.metric = metric

    def forward(self, y, x, physics):
        return self.metric(physics.A(x), y)
