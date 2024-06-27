import torch
from torchvision.transforms.functional import rotate
import numpy as np
from deepinv.transform.base import Transform


class Rotate(Transform):
    r"""
    2D Rotations.

    Generates n_transf randomly rotated versions of 2D images with zero padding.

    :param degrees: images are rotated in the range of angles (-degrees, degrees)
    :param n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, degrees=360, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = degrees

    def forward(self, x):
        r"""
        Applies a random rotation to the input image.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :return: torch.Tensor containing the rotated images concatenated along the first dimension
        """
        if self.group_size == 360:
            theta = np.arange(0, 360)[1:][torch.randperm(359, generator=self.rng)]
            theta = theta[: self.n_trans]
        else:
            theta = np.arange(0, 360, int(360 / (self.group_size + 1)))[1:]
            theta = theta[torch.randperm(self.group_size, generator=self.rng)][
                : self.n_trans
            ]
        return torch.cat([rotate(x, float(_theta)) for _theta in theta])


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
