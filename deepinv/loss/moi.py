import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------
# MOI loss
# --------------------------------------------
class MOILoss(nn.Module):
    def __init__(self, metric=torch.nn.MSELoss(), noise=True):
        """
        Equivariant imaging loss
        https://github.com/edongdongchen/EI
        https://https://arxiv.org/pdf/2103.14756.pdf
        Args:
            ei_loss_weight (int):
        """
        super(MOILoss, self).__init__()
        self.name = 'moi'
        self.metric = metric
        self.noise = noise

    def forward(self, x1, physics, f):

        j = np.random.randint(len(physics))

        if self.noise:
            y = physics[j](x1)
        else:
            y = physics[j].A(x1)

        x2 = f(y, physics[j])

        loss_ei = self.metric(x2, x1)
        return loss_ei
