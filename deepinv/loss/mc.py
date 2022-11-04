import torch
import torch.nn as nn

# --------------------------------------------
# MC loss
# --------------------------------------------
class MCLoss(nn.Module):
    def __init__(self, physics, mc_loss_weight=1, metric=torch.nn.MSELoss()):
        """
        measurement (or data) consistency loss
        Args:
            mc_loss_weight (int):
        """
        super(MCLoss, self).__init__()
        self.name = 'mc'
        self.metric = metric
        self.A = lambda x: physics.A(x)

    def forward(self, x_net, y):
        # x = model(y) # cannot used as an esitmation to PSNR
        return self.metric(self.A(x_net), y)