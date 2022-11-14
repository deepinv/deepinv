import torch
from deepinv.diffops.physics import inpainting

class MeaSplitLoss(torch.nn.Module):
    def __init__(self, physics, metric=torch.nn.MSELoss(), split_ratio=0.65):
        super(MeaSplitLoss, self).__init__()
        self.name = 'ms'
        self.physics = physics
        self.metric = metric
        self.split_ratio = split_ratio

    def forward(self, y, f):
        # sample a splitting
        tsize = y.size()[1:]

        mask = torch.ones(tsize).to(y.get_device())

        mask[torch.rand_like(mask) > self.split_ratio] = 0
        inp = inpainting(tsize, mask)
        inp2 = inpainting(tsize, torch.ones_like(mask)-mask)

        physics1 = inp + self.physics
        physics2 = inp2 + self.physics

        y1 = inp.A(y)
        y2 = inp2.A(y)

        loss_ms = self.metric(physics2.A(f(y1, physics1)), y2)

        loss_ms /= (1-self.split_ratio) # normalize loss

        return loss_ms

