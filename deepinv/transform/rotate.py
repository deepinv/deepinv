from __future__ import annotations
from typing import Iterable
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from deepinv.transform.base import Transform, TransformParam
from warnings import warn


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
    :param str, torchvision.transforms.InterpolationMode interpolation_mode: interpolation mode or equivalent string used for rotation,
        defaults to `nearest`. See :class:`torchvision.transforms.InterpolationMode` for options.
    :param int n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if ``None``, use :class:`torch.Generator`, defaults to ``None``
    """

    def __init__(
        self,
        *args,
        limits: float = 360.0,
        multiples: float = 1.0,
        positive: bool = False,
        interpolation_mode: str | InterpolationMode | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.limits = limits
        self.multiples = multiples
        self.positive = positive
        if interpolation_mode is None:
            interpolation_mode = InterpolationMode.NEAREST
            if multiples % 90 != 0:
                warn(
                    "The default interpolation mode will be changed to bilinear "
                    "interpolation in the near future. Please specify the interpolation "
                    "mode explicitly if you plan to keep using nearest interpolation."
                )
        self.interpolation_mode = (
            InterpolationMode(interpolation_mode)
            if isinstance(interpolation_mode, str)
            else interpolation_mode
        )

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
        theta: torch.Tensor | Iterable | TransformParam = tuple(),
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


def rotate_via_shear(image: torch.Tensor, angle: torch.Tensor, center=None):
    r"""
    2D rotation of image by angle via shear composition through FFT.

    :param torch.Tensor image: input image of shape `(B,C,H,W)`
    :param torch.Tensor, float, int angle: input rotation angles in degrees of shape `(B,)`
    :return: torch.Tensor containing the rotated images of shape `(B, C, H, W )`
    """
    # Convert angle to radians
    if isinstance(angle, float) or isinstance(angle, int):
        angle = torch.tensor([angle], device=image.device, dtype=image.dtype).expand(
            image.shape[0]
        )
    if isinstance(angle, torch.Tensor):
        if angle.dim() == 0:
            angle = angle.unsqueeze(0).expand(image.shape[0])
    else:
        raise ValueError(
            f"angle must be a float, int, or torch.Tensor, got {type(angle)}"
        )

    angle = torch.deg2rad(angle)
    N0, N1 = image.shape[-2:]
    if center is None:
        center = (N0 // 2, N1 // 2)

    mask_angles = (angle > torch.pi / 2.0) & (angle <= 3 * torch.pi / 2)

    angle[angle > 3 * torch.pi / 2] -= 2 * torch.pi

    transformed_image = torch.zeros_like(image)
    expanded_image = image.expand(mask_angles.shape[0], -1, -1, -1)
    transformed_image[~mask_angles] = expanded_image[~mask_angles]
    transformed_image[mask_angles] = torch.rot90(
        expanded_image[mask_angles], k=-2, dims=(-2, -1)
    )

    angle[mask_angles] -= torch.pi

    tant2 = -torch.tan(-angle / 2)
    st = torch.sin(-angle)

    def shearx(image, shear):
        fft1 = torch.fft.fft(image, dim=(-1))
        freq_1 = torch.fft.fftfreq(N1, d=1.0, device=image.device)
        freq_0 = (
            shear[:, None] * (torch.arange(N0, device=image.device) - center[0])[None]
        )
        phase_shift = torch.exp(
            -2j * torch.pi * freq_0[..., None] * freq_1[None, None, :]
        )
        image_shear = fft1 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft(image_shear, dim=(-1)))

    def sheary(image, shear):
        fft0 = torch.fft.fft(image, dim=(-2))
        freq_0 = torch.fft.fftfreq(N0, d=1.0, device=image.device)
        freq_1 = (
            shear[:, None] * (torch.arange(N1, device=image.device) - center[1])[None]
        )
        phase_shift = torch.exp(
            -2j * torch.pi * freq_0[None, :, None] * freq_1[:, None, :]
        )
        image_shear = fft0 * phase_shift[:, None]
        return torch.abs(torch.fft.ifft(image_shear, dim=(-2)))

    rot = shearx(sheary(shearx(transformed_image, tant2), st), tant2)
    return rot
