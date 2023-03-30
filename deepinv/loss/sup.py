import torch
import torch.nn as nn
# --------------------------------------------
# Supervised loss
# --------------------------------------------

class SupLoss(nn.Module):
    r'''
    Standard supervised loss

    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    '''
    def __init__(self, metric=torch.nn.MSELoss()):
        super(SupLoss, self).__init__()
        self.name = 'sup'
        self.metric = metric

    def forward(self, x_net, x):
        return self.metric(x_net, x)
