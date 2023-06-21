import torch
from torchvision.transforms.functional import rotate
import numpy as np


class Rotate(torch.nn.Module):
    r"""
    2D Rotations.

    Generates n_transf randomly rotated versions of 2D images with zero padding.

    :param n_trans: number of rotated versions generated per input image.
    :param degrees: images are rotated in the range of angles (-degrees, degrees)
    """

    def __init__(self, n_trans=1, degrees=360):
        super(Rotate, self).__init__()
        self.n_trans, self.group_size = n_trans, degrees

    def forward(self, data):
        if self.group_size == 360:
            theta = np.arange(0, 360)[1:][torch.randperm(359)]
            theta = theta[: self.n_trans]
        else:
            theta = np.arange(0, 360, int(360 / (self.group_size + 1)))[1:]
            theta = theta[torch.randperm(self.group_size)][: self.n_trans]
        return torch.cat([rotate(data, float(_theta)) for _theta in theta])


# if __name__ == "__main__":
#     device = "cuda:0"
#
#     x = torch.zeros(1, 1, 64, 64, device=device)
#     x[:, :, 16:48, 16:48] = 1
#
#     t = Rotate(4)
#     y = t(x)
#
#     from deepinv.utils import plot
#
#     plot([x, y[0, :, :, :].unsqueeze(0), y[1, :, :, :].unsqueeze(0)])
