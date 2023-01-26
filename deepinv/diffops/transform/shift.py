import torch

class Shift(torch.nn.Module):
    def __init__(self, n_trans):
        '''
        Generates n_transf randomly shifted versions of 2D images with circular padding.
        :param n_trans: number of shifted versions generated per input image.
        '''
        super(Shift, self).__init__()
        self.n_trans = n_trans
    def forward(self, data):
        H, W = data.shape[-2:]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1
        x = torch.arange(-H, H)[torch.randperm(2*H)][:self.n_trans]
        y = torch.arange(-W, W)[torch.randperm(2*W)][:self.n_trans]
        return torch.cat([torch.roll(data, [sx, sy], [-2, -1]) for sx, sy in zip(x, y)])