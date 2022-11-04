import torch

class MeaSplitLoss(torch.nn.Module):
    def __init__(self, physics, metric=torch.nn.MSELoss(), division_mask_rate=0.1):
        super(MeaSplitLoss, self).__init__()
        self.name = 'ms'
        self.physics = physics
        self.metric = metric
        self.division_mask_rate = division_mask_rate
        self.A = lambda x: physics.A(x)
        self.A_dagger = lambda y: physics.A_dagger(y)

    def forward(self, y, f):
        mask1, mask2 = self.update_division_mask(self.physics.mask)

        A1 = lambda x: self.masking_y(self.A(x), mask1)
        A2 = lambda x: self.masking_y(self.A(x), mask2)

        y1 = self.masking_y(y, mask1)
        y2 = self.masking_y(y, mask2)

        # loss_ms = self.metric(A1(model(self.A_dagger(y2))), y1)
        loss_ms = self.metric(A1(f(y2)), y1)

        return loss_ms

    def masking_y(self, y, new_mask=None):
        masked_y = torch.einsum('kl,ijkl->ijkl', new_mask, y)
        return masked_y

    def update_division_mask(self, mask):
        mask_left = torch.ones_like(mask)  # 256x256 all ones
        mask_left[torch.rand_like(mask_left) >= self.division_mask_rate] = 0

        mask_right = torch.ones_like(mask) - mask_left
        return mask_left, mask_right