import torch.nn as nn

class BaseOptim(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y, physics, init=None, x_gt=None):
        raise NotImplementedError