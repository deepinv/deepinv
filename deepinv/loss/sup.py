import torch
import torch.nn as nn

# --------------------------------------------
# Supversided loss
# --------------------------------------------
class SupLoss(nn.Module):
    def __init__(self, metric=torch.nn.MSELoss()):
        """
        supervised (paired GT x and meas. y) loss
        Args:
            sup_loss_weight (int):
        """
        super(SupLoss, self).__init__()
        self.name = 'sup'
        self.metric = metric

    def forward(self, x_net, x):
        return self.metric(x_net, x)
