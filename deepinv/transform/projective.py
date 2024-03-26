from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image

from scipy.spatial.transform import Rotation
from kornia.geometry.transform import warp_perspective


def apply_homography(
    im: torch.Tensor | Image.Image,
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
) -> torch.Tensor | Image.Image:
    """Perform homography (projective transformation).

    :param torch.Tensor | Image.Image im: Input image of shape (B,C,H,W)
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
    :param str interpolation: kornia interpolation mode, defaults to "bilinear"
    :param bool verbose: if True, print homography matrix, defaults to False
    :param str device: torch device, defaults to "cpu"
    :return torch.Tensor | Image.Image: transformed image.
    """

    # Assumptions: principal point in centre, initial focal length 100, initial skew of 0, initial square pixels.
    u0, v0 = int(im.shape[2] / 2), int(im.shape[3] / 2)
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

    R_dash = Rotation.from_euler(
        "xyz", [theta_x, theta_y, theta_z], degrees=True
    ).as_matrix()

    if verbose:
        with np.printoptions(precision=2, suppress=True):
            print(H_inverse)

    if isinstance(im, torch.Tensor):
        # note thetas defined in the opposite direction here, but it doesn't matter
        # for random transformations which have symmetric ranges about 0.
        H_inverse = K @ R_dash @ np.linalg.inv(K_dash)

        return warp_perspective(
            im.double(),
            torch.from_numpy(H_inverse)[None].to(device),
            dsize=im.shape[2:],
            mode=interpolation,
            padding_mode=padding,
        )
    else:
        H = K_dash @ R_dash @ np.linalg.inv(K)

        return im.transform(
            size=(im.size[0], im.size[1]),
            method=Image.Transform.PERSPECTIVE,
            data=H.flatten(),
            resample=Image.Resampling.BILINEAR,
        )


@dataclass
class Homography(torch.nn.Module):
    """
    Homography (or projective transformation). The homography is parameterised by
    geometric parameters. By fixing these parameters, subgroup transformations are
    retrieved, see Wang et al. "Perspective-Equivariant Imaging: an Unsupervised
    Framework for Multispectral Pansharpening" https://arxiv.org/abs/2403.09327

    For example, setting x_stretch_factor_min = y_stretch_factor_min = zoom_factor_min = 1,
    theta_max = theta_z_max = skew_max = 0 gives a pure translation.

    Subgroup transformations include `deepinv.transform.Affine`, `deepinv.transform.Similarity`,
    `deepinv.transform.Euclidean` along with the basic `deepinv.transform.Shift`,
    `deepinv.transform.Rotation` and semigroup `deepinv.transform.Scale`.

    Transformations with perspective effects (i.e. pan+tilt) are recovered by setting
    theta_max > 0.

    Generates n_trans random transformations concatenated along the batch dimension.

    :param int n_trans: Number of transformations, defaults to 1.
    :param float theta_max: Maximum pan+tilt angle in degrees, defaults to 180.
    :param float theta_z_max: Maximum 2D z-rotation angle in degrees, defaults to 180.
    :param float zoom_factor_min: Minimum zoom factor (up to 1), defaults to 0.5.
    :param float shift_max: Maximum shift percentage, where 1 is full shift, defaults to 1.
    :param float skew_max: Maximum skew parameter, defaults to 50.
    :param float x_stretch_factor_min: Min stretch factor along the x-axis (up to 1), defaults to 0.5.
    :param float y_stretch_factor_min: Min stretch factor along the y-axis (up to 1), defaults to 0.5.
    :param str padding: kornia padding mode, defaults to "reflection"
    :param str interpolation: kornia interpolation mode, defaults to "bilinear"
    :param str device: torch device, defaults to "cpu".
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

    def __post_init__(self, *args, **kwargs):
        super().__init__()

    def rand(self, maxi: float, mini: float = None) -> np.ndarray:
        return np.random.default_rng().uniform(
            -maxi if mini is None else mini, maxi, self.n_trans
        )

    def forward(self, data):
        H, W = data.shape[-2:]
        return torch.cat(
            [
                apply_homography(
                    data.double(),
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
                    self.rand(self.theta_max),
                    self.rand(self.theta_max),
                    self.rand(self.theta_z_max),
                    self.rand(1, self.zoom_factor_min),
                    self.rand(W / 2 * self.shift_max),
                    self.rand(H / 2 * self.shift_max),  ### note W and H swapped
                    self.rand(self.skew_max),
                    self.rand(1, self.x_stretch_factor_min),
                    self.rand(1, self.y_stretch_factor_min),
                )
            ],
            dim=0,
        ).float()


class Affine(Homography):
    """
    Special case of homography which corresponds to the actions of the affine subgroup
    Aff(3). Affine transformations include translations, rotations, reflections,
    skews, and stretches. See `deepinv.transform.Homography` for more details.
    """

    def forward(self, data):
        self.theta_max = 0
        return super().forward(data)


class Similarity(Homography):
    """
    Special case of homography which corresponds to the actions of the similarity subgroup
    S(2). Similarity transformations include translations, rotations, reflections and
    uniform scale. See `deepinv.transform.Homography` for more details.
    """

    def forward(self, data):
        self.theta_max = self.skew_max = 0
        self.x_stretch_factor_min = self.y_stretch_factor_min = 1
        return super().forward(data)


class Euclidean(Homography):
    """
    Special case of homography which corresponds to the actions of the Euclidean subgroup
    E(2). Euclidean transformations include translations, rotations and reflections.
    See `deepinv.transform.Homography` for more details.
    """

    def forward(self, data):
        self.theta_max = self.skew_max = 0
        self.zoom_factor_min = self.x_stretch_factor_min = self.y_stretch_factor_min = 1
        return super().forward(data)


class PanTiltRotate(Homography):
    """
    Special case of homography which corresponds to the actions of the 3D camera rotation,
    or "pan+tilt+rotate" subgroup from Wang et al. "Perspective-Equivariant Imaging: an
    Unsupervised Framework for Multispectral Pansharpening" https://arxiv.org/abs/2403.09327

    The transformations simulate panning, tilting or rotating the camera, leading to a
    "perspective" effect. The subgroup is isomorphic to SO(3).

    See `deepinv.transform.Homography` for more details.
    """

    def forward(self, data):
        self.shift_max = self.skew_max = 0
        self.zoom_factor_min = self.x_stretch_factor_min = self.y_stretch_factor_min = 1
        return super().forward(data)
