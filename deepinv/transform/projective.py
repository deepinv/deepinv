from dataclasses import dataclass

from typing import Union, Iterable

import numpy as np
import torch
from PIL import Image

from deepinv.transform.base import Transform, TransformParam

try:
    from kornia.geometry.transform import warp_perspective
except ImportError:

    def warp_perspective(*args, **kwargs):
        raise ImportError("The kornia package is not installed.")


def rotation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """Numpy implementation of ``scipy`` rotation matrix from Euler angles.

    Construct 3D extrinsic rotation matrix from x, y and z angles. This is equivalent of using the ``scipy`` function:

    ``scipy.spatial.transform.Rotation.from_euler("xyz", (tx, ty, tz), degrees=True).as_matrix()``

    :param float tx: x rotation in degrees
    :param float ty: y rotation in degrees
    :param float tz: z rotation in degrees
    :return np.ndarray: 3D rotation matrix.
    """
    tx, ty, tz = np.radians((tx, ty, tz))

    # fmt: off
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(tx), -np.sin(tx)],
        [0, np.sin(tx), np.cos(tx)]
    ])

    Ry = np.array([
        [np.cos(ty), 0, np.sin(ty)],
        [0, 1, 0],
        [-np.sin(ty), 0, np.cos(ty)]
    ])

    Rz = np.array([
        [np.cos(tz), -np.sin(tz), 0],
        [np.sin(tz),  np.cos(tz), 0],
        [0, 0, 1]
    ])
    # fmt: on

    return Rz @ Ry @ Rx


def apply_homography(
    im: Union[torch.Tensor, Image.Image],
    theta_x: float = 0.0,
    theta_y: float = 0.0,
    theta_z: float = 0.0,
    zoom_factor: float = 1.0,
    skew: float = 0,
    x_stretch_factor: float = 1.0,
    y_stretch_factor: float = 1.0,
    x_t: float = 0.0,
    y_t: float = 0.0,
    padding: str = "reflection",
    interpolation: str = "bilinear",
    verbose: bool = False,
    device="cpu",
    **kwargs,
) -> Union[torch.Tensor, Image.Image]:
    r"""Perform homography (projective transformation).

    Given physical parameters describing camera variation, this function performs the geometric transformation given by the change in parameters.

    See :class:`deepinv.transform.Homography` for more details.

    The input image can be a torch Tensor, in which case ``kornia`` is used to perform the transformation, or a PIL Image where PIL transform is used.

    Following https://arxiv.org/abs/2403.09327, we assume principal point in centre, initial focal length 100, initial skew of 0, initial square pixels.

    :param torch.Tensor | Image.Image im: Input if tensor, image of shape (B,C,H,W), otherwise a PIL image.
    :param float theta_x: tilt angle in degrees, defaults to 0.
    :param float theta_y: pan angle in degrees, defaults to 0.
    :param float theta_z: 2D rotation angle in degrees, defaults to 0.
    :param float zoom_factor: relative focal length zoom (lower zooms out), defaults to 1.
    :param float skew: relative pixel skew, defaults to 0
    :param float x_stretch_factor: relative pixel x length factor, defaults to 1.
    :param float y_stretch_factor: relative pixel y length factor, defaults to 1.
    :param float x_t: relative x pixel translation, defaults to 0.
    :param float y_t: relative y pixel translation, defaults to 0.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia or PIL interpolation mode, choose from "bilinear", "nearest" or "bicubic". Defaults to "bilinear"
    :param bool verbose: if True, print homography matrix, defaults to False
    :param str device: torch device, defaults to "cpu"
    :return torch.Tensor | Image.Image: transformed image.
    """

    assert interpolation in ("bilinear", "bicubic", "nearest")

    w, h = (im.shape[2], im.shape[3]) if isinstance(im, torch.Tensor) else im.size
    u0, v0 = int(w / 2), int(h / 2)
    f = 100
    s = 0
    m_x = m_y = 1

    # fmt: off
    K = np.array([
        [f*m_x, s, u0],
        [0, f*m_y, v0],
        [0, 0, 1]
    ])

    K_dash = np.array([
        [f/zoom_factor*m_x/x_stretch_factor, s + skew, u0 + x_t],
        [0, f/zoom_factor*m_y/y_stretch_factor, v0 + y_t],
        [0, 0, 1]
    ])
    # fmt: on

    R_dash = rotation_matrix(theta_x, theta_y, theta_z)

    if isinstance(im, torch.Tensor):
        # note thetas defined in the opposite direction here, but it doesn't matter
        # for random transformations which have symmetric ranges about 0.
        H_inverse = K @ R_dash @ np.linalg.inv(K_dash)

        if verbose:
            with np.printoptions(precision=2, suppress=True):
                print(H_inverse)

        return warp_perspective(
            im.double(),
            torch.from_numpy(H_inverse)[None].to(device),
            dsize=im.shape[2:],
            mode=interpolation,
            padding_mode=padding,
        )
    else:
        if interpolation == "bilinear":
            pil_interp = Image.Resampling.BILINEAR
        elif interpolation == "bicubic":
            pil_interp = Image.Resampling.BICUBIC
        elif interpolation == "nearest":
            pil_interp = Image.Resampling.NEAREST

        H = K_dash @ R_dash @ np.linalg.inv(K)

        return im.transform(
            size=(im.size[0], im.size[1]),
            method=Image.Transform.PERSPECTIVE,
            data=H.flatten(),
            resample=pil_interp,
        )


@dataclass
class Homography(Transform):
    """
    Random projective transformations (homographies).

    The homography is parameterised by
    geometric parameters. By fixing these parameters, subgroup transformations are
    retrieved, see Wang et al. "Perspective-Equivariant Imaging: an Unsupervised
    Framework for Multispectral Pansharpening" https://arxiv.org/abs/2403.09327

    For example, setting x_stretch_factor_min = y_stretch_factor_min = zoom_factor_min = 1,
    theta_max = theta_z_max = skew_max = 0 gives a pure translation.

    Subgroup transformations include :class:`deepinv.transform.projective.Affine`, :class:`deepinv.transform.projective.Similarity`,
    :class:`deepinv.transform.projective.Euclidean` along with the basic :class:`deepinv.transform.Shift`,
    :class:`deepinv.transform.Rotate` and semigroup :class:`deepinv.transform.Scale`.

    Transformations with perspective effects (i.e. pan+tilt) are recovered by setting
    theta_max > 0.

    Generates ``n_trans`` random transformations concatenated along the batch dimension.

    |sep|

    :Example:

        Apply a random projective transformation:

        >>> from deepinv.transform.projective import Homography
        >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
        >>> transform = Homography(n_trans = 1)
        >>> x_T = transform(x)

    :param float theta_max: Maximum pan+tilt angle in degrees, defaults to 180.
    :param float theta_z_max: Maximum 2D z-rotation angle in degrees, defaults to 180.
    :param float zoom_factor_min: Minimum zoom factor (up to 1), defaults to 0.5.
    :param float shift_max: Maximum shift percentage, where 1 is full shift, defaults to 1.
    :param float skew_max: Maximum skew parameter, defaults to 50.
    :param float x_stretch_factor_min: Min stretch factor along the x-axis (up to 1), defaults to 0.5.
    :param float y_stretch_factor_min: Min stretch factor along the y-axis (up to 1), defaults to 0.5.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia or PIL interpolation mode, defaults to "bilinear"
    :param str device: torch device, defaults to "cpu".
    :param int n_trans: number of transformed versions generated per input image, defaults to 1.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    n_trans: int = 1
    theta_max: float = 180.0
    theta_z_max: float = 180.0
    zoom_factor_min: float = 0.5
    shift_max: float = 1.0
    skew_max: float = 50.0
    x_stretch_factor_min: float = 0.5
    y_stretch_factor_min: float = 0.5
    padding: str = "reflection"
    interpolation: str = "bilinear"
    device: str = "cpu"
    rng: torch.Generator = None

    def __post_init__(self, *args, **kwargs):
        super().__init__(*args, n_trans=self.n_trans, rng=self.rng, **kwargs)

    def rand(self, maxi: float, mini: float = None) -> torch.Tensor:
        if mini is None:
            mini = -maxi
        out = (mini - maxi) * torch.rand(
            self.n_trans, generator=self.rng, device=self.rng.device
        ) + maxi
        return out.cpu()  # require cpu for numpy

    def _get_params(self, x: torch.Tensor) -> dict:
        H, W = x.shape[-2:]

        Reciprocal = lambda p: TransformParam(p, neg=lambda x: 1 / x)

        return {
            "theta_x": self.rand(self.theta_max),
            "theta_y": self.rand(self.theta_max),
            "theta_z": self.rand(self.theta_z_max),
            "zoom_f": Reciprocal(self.rand(1, self.zoom_factor_min)),
            "shift_x": self.rand(W / 2 * self.shift_max),
            "shift_y": self.rand(H / 2 * self.shift_max),  ### note W and H swapped
            "skew": self.rand(self.skew_max),
            "stretch_x": Reciprocal(self.rand(1, self.x_stretch_factor_min)),
            "stretch_y": Reciprocal(self.rand(1, self.y_stretch_factor_min)),
        }

    def _transform(
        self,
        x: torch.Tensor,
        theta_x: Union[torch.Tensor, Iterable, TransformParam] = [],
        theta_y: Union[torch.Tensor, Iterable, TransformParam] = [],
        theta_z: Union[torch.Tensor, Iterable, TransformParam] = [],
        zoom_f: Union[torch.Tensor, Iterable, TransformParam] = [],
        shift_x: Union[torch.Tensor, Iterable, TransformParam] = [],
        shift_y: Union[torch.Tensor, Iterable, TransformParam] = [],
        skew: Union[torch.Tensor, Iterable, TransformParam] = [],
        stretch_x: Union[torch.Tensor, Iterable, TransformParam] = [],
        stretch_y: Union[torch.Tensor, Iterable, TransformParam] = [],
        **params,
    ) -> torch.Tensor:
        return torch.cat(
            [
                apply_homography(
                    x.double(),
                    theta_x=tx,
                    theta_y=ty,
                    theta_z=tz,
                    zoom_factor=zf,
                    x_t=xt,
                    y_t=yt,
                    skew=sk,
                    x_stretch_factor=xsf,
                    y_stretch_factor=ysf,
                    padding=self.padding,
                    interpolation=self.interpolation,
                    device=self.device,
                )
                for tx, ty, tz, zf, xt, yt, sk, xsf, ysf in zip(
                    theta_x,
                    theta_y,
                    theta_z,
                    zoom_f,
                    shift_x,
                    shift_y,
                    skew,
                    stretch_x,
                    stretch_y,
                )
            ],
            dim=0,
        ).float()


class Affine(Homography):
    """Random affine image transformations using projective transformation framework.

    Special case of homography which corresponds to the actions of the affine subgroup
    Aff(3). Affine transformations include translations, rotations, reflections,
    skews, and stretches. These transformations are parametrised using geometric parameters in the pinhole camera model.
    See :class:`deepinv.transform.Homography` for more details.

    Generates ``n_trans`` random transformations concatenated along the batch dimension.

    |sep|

    :Example:

        Apply a random affine transformation:

        >>> from deepinv.transform.projective import Affine
        >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
        >>> transform = Affine(n_trans = 1)
        >>> x_T = transform(x)

    :param float theta_z_max: Maximum 2D z-rotation angle in degrees, defaults to 180.
    :param float zoom_factor_min: Minimum zoom factor (up to 1), defaults to 0.5.
    :param float shift_max: Maximum shift percentage, where 1 is full shift, defaults to 1.
    :param float skew_max: Maximum skew parameter, defaults to 50.
    :param float x_stretch_factor_min: Min stretch factor along the x-axis (up to 1), defaults to 0.5.
    :param float y_stretch_factor_min: Min stretch factor along the y-axis (up to 1), defaults to 0.5.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia or PIL interpolation mode, defaults to "bilinear"
    :param str device: torch device, defaults to "cpu".
    :param n_trans: number of transformed versions generated per input image, defaults to 1.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def _get_params(self, x: torch.Tensor) -> dict:
        self.theta_max = 0
        return super()._get_params(x)


class Similarity(Homography):
    """Random 2D similarity image transformations using projective transformation framework.

    Special case of homography which corresponds to the actions of the similarity subgroup
    S(2). Similarity transformations include translations, rotations, reflections and
    uniform scale. These transformations are parametrised using geometric parameters in the pinhole camera model. See :class:`deepinv.transform.Homography` for more details.

    Generates ``n_trans`` random transformations concatenated along the batch dimension.

    |sep|

    :Example:

        Apply a random similarity transformation:

        >>> from deepinv.transform.projective import Similarity
        >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
        >>> transform = Similarity(n_trans = 1)
        >>> x_T = transform(x)

    :param float theta_z_max: Maximum 2D z-rotation angle in degrees, defaults to 180.
    :param float zoom_factor_min: Minimum zoom factor (up to 1), defaults to 0.5.
    :param float shift_max: Maximum shift percentage, where 1 is full shift, defaults to 1.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia or PIL interpolation mode, defaults to "bilinear"
    :param str device: torch device, defaults to "cpu".
    :param n_trans: number of transformed versions generated per input image, defaults to 1.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def _get_params(self, x: torch.Tensor) -> dict:
        self.theta_max = self.skew_max = 0
        self.x_stretch_factor_min = self.y_stretch_factor_min = 1
        return super()._get_params(x)


class Euclidean(Homography):
    """Random Euclidean image transformations using projective transformation framework.

    Special case of homography which corresponds to the actions of the Euclidean subgroup
    E(2). Euclidean transformations include translations, rotations and reflections. These transformations are parametrised using geometric parameters in the pinhole camera model.
    See :class:`deepinv.transform.Homography` for more details.

    Generates ``n_trans`` random transformations concatenated along the batch dimension.

    |sep|

    :Example:

        Apply a random Euclidean transformation:

        >>> from deepinv.transform.projective import Euclidean
        >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
        >>> transform = Euclidean(n_trans = 1)
        >>> x_T = transform(x)

    :param float theta_z_max: Maximum 2D z-rotation angle in degrees, defaults to 180.
    :param float shift_max: Maximum shift percentage, where 1 is full shift, defaults to 1.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia or PIL interpolation mode, defaults to "bilinear"
    :param str device: torch device, defaults to "cpu".
    :param n_trans: number of transformed versions generated per input image, defaults to 1.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def _get_params(self, x: torch.Tensor) -> dict:
        self.theta_max = self.skew_max = 0
        self.zoom_factor_min = self.x_stretch_factor_min = self.y_stretch_factor_min = 1
        return super()._get_params(x)


class PanTiltRotate(Homography):
    """Random 3D camera rotation image transformations using projective transformation framework.

    Special case of homography which corresponds to the actions of the 3D camera rotation,
    or "pan+tilt+rotate" subgroup from Wang et al. "Perspective-Equivariant Imaging: an
    Unsupervised Framework for Multispectral Pansharpening" https://arxiv.org/abs/2403.09327

    The transformations simulate panning, tilting or rotating the camera, leading to a
    "perspective" effect. The subgroup is isomorphic to SO(3).

    See :class:`deepinv.transform.Homography` for more details.

    Generates ``n_trans`` random transformations concatenated along the batch dimension.

    |sep|

    :Example:

        Apply a random pan+tilt+rotate transformation:

        >>> from deepinv.transform.projective import PanTiltRotate
        >>> x = torch.randn(1, 3, 16, 16) # Random 16x16 image
        >>> transform = PanTiltRotate(n_trans = 1)
        >>> x_T = transform(x)

    :param float theta_max: Maximum pan+tilt angle in degrees, defaults to 180.
    :param float theta_z_max: Maximum 2D z-rotation angle in degrees, defaults to 180.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia or PIL interpolation mode, defaults to "bilinear"
    :param str device: torch device, defaults to "cpu".
    :param n_trans: number of transformed versions generated per input image, defaults to 1.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def _get_params(self, x: torch.Tensor) -> dict:
        self.shift_max = self.skew_max = 0
        self.zoom_factor_min = self.x_stretch_factor_min = self.y_stretch_factor_min = 1
        return super()._get_params(x)
