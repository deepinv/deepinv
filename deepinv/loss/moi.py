import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------
# MOI loss
# --------------------------------------------
class MOILoss(nn.Module):
    r'''
    Multi-operator imaging loss

    https://arxiv.org/abs/2201.12151


    :param torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param float weight: total weight of the loss
    :param bool noise: if True, the augmented measurement is computed with the full sensing model (noise+sensor model), otherwise is generated in a noiseless manner y=A_gx.
    '''
    def __init__(self, metric=torch.nn.MSELoss(), noise=True, weight=1.):
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
