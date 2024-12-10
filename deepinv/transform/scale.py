# code from https://github.com/jscanvic/Scale-Equivariant-Imaging
from typing import Union, Iterable
import torch
import torch.nn.functional as F
from deepinv.transform.base import Transform, TransformParam


def sample_from(
    values,
    shape=(1,),
    dtype=torch.float32,
    device="cpu",
    generator: torch.Generator = None,
):
    """Sample a random tensor from a list of values"""
    values = torch.tensor(values, device=device, dtype=dtype)
    N = torch.tensor(len(values), device=device, dtype=dtype)
    indices = (
        torch.floor(
            N
            * torch.rand(
                shape, dtype=dtype, device=generator.device, generator=generator
            )
        )
        .to(torch.long)
        .to(device)
    )
    return values[indices]


class Scale(Transform):
    r"""
    2D Scaling.

    Resample the input image on a grid obtained using
    an isotropic dilation, with random scale factor
    and origin. By default, the input image is viewed
    as periodic and the output image is effectively padded
    by reflections. Additionally, resampling is performed
    using bicubic interpolation.

    See the paper `Self-Supervised Learning for Image Super-Resolution and Deblurring <https://arxiv.org/abs/2312.11232>`_
    for more details.

    Note each image in the batch is transformed independently.

    :param list factors: list of scale factors (default: [.75, .5])
    :param str padding_mode: padding mode for grid sampling
    :param str mode: interpolation mode for grid sampling
    :param int n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(
        self, *args, factors=None, padding_mode="reflection", mode="bicubic", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.factors = factors or [0.75, 0.5]
        self.padding_mode = padding_mode
        self.mode = mode

    def _get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate scale factor parameters.

        :param torch.Tensor x: input image
        :return dict: keyword args of scale factors
        """
        # Total number of transforms = n_trans * batch_size
        b = x.shape[0] * self.n_trans

        # Sample a random scale factor for each batch element
        factor = sample_from(
            self.factors, shape=(b,), device=x.device, generator=self.rng
        ).to(x.device)

        # Sample a random transformation center for each batch element
        # with coordinates in [-1, 1]
        center = torch.rand(
            (b, 2), dtype=x.dtype, device=self.rng.device, generator=self.rng
        ).to(x.device)

        # Scale params override negation
        return {
            "factor": TransformParam(factor, neg=lambda x: 1 / x),
            "center": TransformParam(center, neg=lambda x: x),
        }

    def _transform(
        self,
        x: torch.Tensor,
        factor: Union[torch.Tensor, Iterable, TransformParam] = [],
        center: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> torch.Tensor:
        """Scale image given scale parameters.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor, list factor: iterable of scale factors to be used, one per ``n_trans*batch_size``.
        :param torch.Tensor, list center: iterable of scale centers, one per ``n_trans*batch_size``.
        :return: torch.Tensor: scaled image.
        """
        # Prepare for multiple transforms
        x = x.repeat(self.n_trans, 1, 1, 1)

        b, _, h, w = x.shape

        if len(factor) < b:
            factor = factor.expand(b, *factor.shape[1:])

        if len(center) < b:
            center = center.expand(b, *center.shape[1:])

        factor = factor.view(b, 1, 1, 1).repeat(1, 1, 1, 2)
        center = center.view(b, 1, 1, 2)
        center = 2 * center - 1

        # Compute the sampling grid for the scale transformation
        u = torch.arange(w, dtype=x.dtype, device=x.device)
        v = torch.arange(h, dtype=x.dtype, device=x.device)
        u = 2 / w * u - 1
        v = 2 / h * v - 1
        U, V = torch.meshgrid(u, v)
        grid = torch.stack([V, U], dim=-1)
        grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)
        grid = 1 / factor * (grid - center) + center

        return F.grid_sample(
            x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=True
        )
