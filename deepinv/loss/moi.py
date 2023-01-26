import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------
# MOI loss
# --------------------------------------------
class MOILoss(nn.Module):
    def __init__(self, metric=torch.nn.MSELoss(), noise=True, weight=1.):
        """
        Equivariant imaging loss
        https://github.com/edongdongchen/EI
        https://https://arxiv.org/pdf/2103.14756.pdf
        Args:
            metric (torch.nn.Module): metric used to compute the reconstruction error. default is mean squared error
            weight (float): total weight of the loss
            noise (bool): if True, the augmented measurement is computed with the full sensing model (noise+sensor model), otherwise is generated in a noiseless manner y=A_gx.
        """
        super(MOILoss, self).__init__()
        self.name = 'moi'
        self.metric = metric
        self.weight = weight
        self.noise = noise

    def forward(self, x1, physics, f):
        j = np.random.randint(len(physics))

        if self.noise:
            y = physics[j](x1)
        else:
            y = physics[j].A(x1)

        x2 = f(y, physics[j])

        return self.weight*self.metric(x2, x1)
