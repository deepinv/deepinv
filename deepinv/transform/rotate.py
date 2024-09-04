from typing import Union, Iterable
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
from deepinv.transform.base import Transform, TransformParam


class Rotate(Transform):
    r"""
    2D Rotations.

    Generates ``n_trans`` randomly rotated versions of 2D images with zero padding (without replacement).

    Picks integer angles between -limits and limits, by default -360 to 360. Set ``positive=True`` to clip to positive degrees.
    For exact pixel rotations (0, 90, 180, 270 etc.), set ``multiples=90``.

    By default, output will be cropped/padded to input shape. Set ``constant_shape=False`` to let output shape differ from input shape.

    See :class:`deepinv.transform.Transform` for further details and examples.

    :param float limits: images are rotated in the range of angles (-limits, limits).
    :param float multiples: angles are selected uniformly from :math:`\pm` multiples of ``multiples``. Default to 1 (i.e integers)
    :param bool positive: if True, only consider positive angles.
    :param n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(
        self,
        *args,
        limits: float = 360.0,
        multiples: float = 1.0,
        positive: bool = False,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.limits = limits
        self.multiples = multiples
        self.positive = positive
        self.interpolation_mode = interpolation_mode

    def get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate rotation parameters.

        :param torch.Tensor x: input image
        :return dict: keyword args of angles theta in degrees
        """
        theta = torch.arange(0, self.limits, self.multiples)
        if not self.positive:
            theta = torch.cat((theta, -theta))
        theta = theta[torch.randperm(len(theta), generator=self.rng)]
        theta = theta[: self.n_trans]
        return {"theta": theta}

    def transform(
        self,
        x: torch.Tensor,
        theta: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> torch.Tensor:
        """Rotate image given thetas.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor, list theta: iterable of rotation angles (degrees), one per ``n_trans``.
        :return: torch.Tensor: transformed image.
        """
        return torch.cat(
            [
                rotate(
                    x,
                    float(_theta),
                    interpolation=self.interpolation_mode,
                    expand=not self.constant_shape,
                )
                for _theta in theta
            ]
        )


# if __name__ == "__main__":
#     device = "cuda:0"
#
#     x = torch.zeros(1, 1, 64, 64, device=device)
#     x[:, :, 16:48, 16:48] = 1
#
#     t = Rotate(n_trans=4)
#     y = t(x)
#
#     from deepinv.utils import plot
#
#     plot([x, y[0, :, :, :].unsqueeze(0), y[1, :, :, :].unsqueeze(0)])
