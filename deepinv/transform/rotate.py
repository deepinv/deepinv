from typing import Union, Iterable
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
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
        When multiples is a multiple of 90, no interpolation is performed.
    :param bool positive: if True, only consider positive angles.
    :param int n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if ``None``, use :class:`torch.Generator`, defaults to ``None``
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

    def _get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate rotation parameters.

        :param torch.Tensor x: input image
        :return dict: keyword args of angles theta in degrees
        """
        theta = torch.arange(0, self.limits, self.multiples, device=self.rng.device)
        if not self.positive:
            theta = torch.cat((theta, -theta))
        theta = theta[
            torch.randperm(len(theta), generator=self.rng, device=self.rng.device)
        ]
        theta = theta[: self.n_trans]
        return {"theta": theta}

    def _transform(
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
