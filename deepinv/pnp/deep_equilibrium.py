
import torch
import torch.nn as nn

class DEQ(nn.Module):
    def __init__(self, FP, iterator, max_iter_backward=50, use_anderson = False) :
        super().__init__()
        self.FP = FP
        self.iterator = iterator
        self.max_iter_backward = max_iter_backward
        self.use_anderson = use_anderson
        self.parameters = PnP_module.parameters

    def forward(self,x,physics):
        # FP(init, input, physics)
        with torch.no_grad():
            z = self.FP(x, x, physics)
        z =  self.iterator(z, 0, x, physics)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.iterator(z0, 0, x, physics)
        def backward_hook(grad):
            g = FixedPoint(lambda x,it: torch.autograd.grad(f0, z0, x, retain_graph=True)[0] + grad, max_iter=self.max_iter_backward, use_anderson=self.use_anderson)(grad)
            return g
        z.register_hook(backward_hook)
        return z