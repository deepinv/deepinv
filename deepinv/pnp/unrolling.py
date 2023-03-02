
import torch
import torch.nn as nn
from deepinv.optim.fixed_point import FixedPoint

class Unrolling(nn.Module):
    def __init__(self, iterator, PnP_module, max_iter=50, early_stop=True, crit_conv=1e-5, verbose=False) :
        super().__init__()
        self.FP = FixedPoint(iterator, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv, verbose=verbose)
        self.parameters = PnP_module.parameters

    def forward(self,x,physics):
        # FP takes as input : init, input, physics)
        return self.FP(x, x, physics) 
