import torch
import torch.nn as nn

# --------------------------------------------
# MC loss
# --------------------------------------------
class MCLoss(nn.Module):
    def __init__(self, metric=torch.nn.MSELoss()):
        """
        measurement (or data) consistency loss
        Args:
            mc_loss_weight (int):
        """
        super(MCLoss, self).__init__()
        self.name = 'mc'
        self.metric = metric

    def forward(self, y, x, physics):
        return self.metric(y, physics.A(x))