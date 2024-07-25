from typing import Union, Iterable
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
from deepinv.transform.base import Transform, Param


class Rotate(Transform):
    r"""
    2D Rotations.

    Generates n_trans randomly rotated versions of 2D images with zero padding.

    See :class:`deepinv.transform.Transform` for further details and examples.

    :param degrees: images are rotated in the range of angles (-degrees, degrees)
    :param n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(
        self,
        *args,
        degrees: Union[float, int] = 360,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group_size = degrees
        self.interpolation_mode = interpolation_mode

    def get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate rotation parameters.

        :param torch.Tensor x: input image
        :return dict: keyword args of angles theta in degrees
        """
        if self.group_size == 360:
            theta = np.arange(0, 360)[1:][torch.randperm(359, generator=self.rng)]
            theta = theta[: self.n_trans]
        else:
            theta = np.arange(0, 360, int(360 / (self.group_size + 1)))[1:]
            theta = theta[torch.randperm(self.group_size, generator=self.rng)][
                : self.n_trans
            ]
        return {"theta": theta}

    def transform(
        self,
        x: torch.Tensor,
        theta: Union[torch.Tensor, Iterable, Param] = [],
        **kwargs,
    ) -> torch.Tensor:
        """Rotate image given thetas.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor, list x_shift: iterable of rotation angles (degrees), one per n_trans.
        :return: torch.Tensor: transformed image.
        """
        return torch.cat(
            [
                rotate(x, float(_theta), interpolation=self.interpolation_mode)
                for _theta in theta
            ]
        )


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
