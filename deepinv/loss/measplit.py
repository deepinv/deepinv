import torch
from deepinv.diffops.physics import Inpainting

class MeaSplitLoss(torch.nn.Module):
    def __init__(self, metric=torch.nn.MSELoss(), split_ratio=0.9, regular_mask=False):
        super(MeaSplitLoss, self).__init__()
        self.name = 'ms'
        self.metric = metric
        self.regular_mask = regular_mask
        self.split_ratio = split_ratio

    def forward(self, y, physics, f):
        tsize = y.size()[1:]

        # sample a splitting
        mask = torch.ones(tsize).to(y.get_device())
        if not self.regular_mask:
            mask[torch.rand_like(mask) > self.split_ratio] = 0
        else: # TODO: add regular mask
            mask[torch.rand_like(mask) > self.split_ratio] = 0


        # create inpainting masks
        # inp = inpainting(tsize, mask)
        # inp2 = inpainting(tsize, 1-mask)

        inp = Inpainting(tsize, mask)
        inp2 = Inpainting(tsize, 1-mask)

        # concatenate operators
        physics1 = inp + physics # A_1 = P*A
        physics2 = inp2 + physics # A_2 = (I-P)*A

        # divide measurements
        y1 = inp.A(y)
        y2 = inp2.A(y)

        loss_ms = self.metric(physics2.A(f(y1, physics1)), y2)

        loss_ms /= (1-self.split_ratio) # normalize loss

        return loss_ms

