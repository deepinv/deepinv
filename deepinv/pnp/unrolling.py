
import torch
import torch.nn as nn

class Unrolling(nn.Module):
    def __init__(self, FP, PnP_module) :
        super().__init__()
        self.FP = FP
        self.parameters = PnP_module.parameters

    def forward(self,x,physics):
        # FP takes as input : init, input, physics)
        return self.FP(x, x, physics) 
