from collections.abc import Mapping, Iterable
from typing import Any
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from deepinv.physics.forward import LinearPhysics

from deepinv.physics.functional import Radon, IRadon, RampFilter
from deepinv.physics import adjoint_function

from deepinv.physics.functional import XrayTransform
from deepinv.physics.functional.astra import (
    AutogradTransform,
    create_projection_geometry,
    create_object_geometry,
)

try:
    import astra
except:
    astra = ImportError("The astra-toolbox package is not installed.")


PI = 4 * torch.ones(1).atan()


class Tomography(LinearPhysics):
    r"""
    (Computed) Tomography operator.

    The Radon transform is the integral transform which takes a square image :math:`x` defined on the plane to a function
    :math:`y=Rx` defined on the (two-dimensional) space of lines in the plane, whose value at a particular line is equal
    to the line integral of the function over that line.

    .. note::

        The pseudo-inverse is computed using the filtered back-projection algorithm with a Ramp filter.
        This is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation which is
        robust to noise.

    .. note::

        The measurements are not normalized by the image size, thus the norm of the operator depends on the image size.

    .. warning::

        The adjoint operator has small numerical errors due to interpolation.

    :param int, torch.Tensor angles: These are the tomography angles. If the type is ``int``, the angles are sampled uniformly between 0 and 360 degrees.
        If the type is :class:`torch.Tensor`, the angles are the ones provided (e.g., ``torch.linspace(0, 180, steps=10)``).
    :param int img_width: width/height of the square image input.
    :param bool circle: If ``True`` both forward and backward projection will be restricted to pixels inside a circle
        inscribed in the square image.
    :param bool parallel_computation: if True, all projections are performed in parallel. Requires more memory but is faster on GPUs.
    :param bool normalize: If ``True``, the outputs are normlized by the image size (i.e. it is assumed that the image lives on [0,1]^2 for the computation of the line integrals).
        In this case the operator norm is approximately given by :math:`\|A\|_2^2  \approx \frac{\pi}{2\,\text{angles}}`,
        If ``False``, then it is assumed that the image lives on [0,im_width]^2 for the computation of the line integrals
    :param bool fan_beam: If ``True``, use fan beam geometry, if ``False`` use parallel beam
    :param dict fan_parameters: Only used if fan_beam is ``True``. Contains the parameters defining the scanning geometry. The dict should contain the keys:

        - "pixel_spacing" defining the distance between two pixels in the image, default: 0.5 / (in_size)

        - "source_radius" distance between the x-ray source and the rotation axis (middle of the image), default: 57.5

        - "detector_radius" distance between the x-ray detector and the rotation axis (middle of the image), default: 57.5

        - "n_detector_pixels" number of pixels of the detector, default: 258

        - "detector_spacing" distance between two pixels on the detector, default: 0.077

        The default values are adapted from the geometry in `https://doi.org/10.5281/zenodo.8307932 <https://doi.org/10.5281/zenodo.8307932>`_,
        where pixel spacing, source and detector radius and detector spacing are given in cm.
        Note that a to small value of n_detector_pixels*detector_spacing can lead to severe circular artifacts in any reconstruction.
    :param str device: gpu or cpu.

    |sep|

    :Examples:

        Tomography operator with defined angles for 3x3 image:

        >>> from deepinv.physics import Tomography
        >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
        >>> angles = torch.linspace(0, 45, steps=3)
        >>> physics = Tomography(angles=angles, img_width=4, circle=True)
        >>> physics(x)
        tensor([[[[ 0.1650,  1.2640,  1.6995],
                  [-0.4860,  0.2674,  0.9971],
                  [ 0.9002, -0.3856, -0.9360],
                  [-2.4882, -2.1068, -2.5720]]]])

        Tomography operator with 3 uniformly sampled angles in [0, 360] for 3x3 image:

        >>> from deepinv.physics import Tomography
        >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
        >>> physics = Tomography(angles=3, img_width=4, circle=True)
        >>> physics(x)
        tensor([[[[ 0.1650,  1.9493,  1.9897],
                  [-0.4860,  0.7137, -1.6536],
                  [ 0.9002, -0.8457, -0.1666],
                  [-2.4882, -2.7340, -0.9793]]]])


    """

    def __init__(
        self,
        angles,
        img_width,
        circle=False,
        parallel_computation=True,
        normalize=False,
        fan_beam=False,
        fan_parameters=None,
        device=torch.device("cpu"),
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(angles, int) or isinstance(angles, float):
            theta = torch.nn.Parameter(
                torch.linspace(0, 180, steps=angles + 1, device=device)[:-1],
                requires_grad=False,
            ).to(device)
        else:
            theta = torch.nn.Parameter(angles, requires_grad=False).to(device)

        self.fan_beam = fan_beam
        self.img_width = img_width
        self.device = device
        self.dtype = dtype
        self.normalize = normalize
        self.radon = Radon(
            img_width,
            theta,
            circle=circle,
            parallel_computation=parallel_computation,
            fan_beam=fan_beam,
            fan_parameters=fan_parameters,
            device=device,
            dtype=dtype,
        ).to(device)
        if not self.fan_beam:
            self.iradon = IRadon(
                img_width,
                theta,
                circle=circle,
                parallel_computation=parallel_computation,
                device=device,
                dtype=dtype,
            ).to(device)
        else:
            self.filter = RampFilter(dtype=dtype, device=device)

    def A(self, x, **kwargs):
        if self.img_width is None:
            self.img_width = x.shape[-1]
        output = self.radon(x)
        if self.normalize:
            output = output / x.shape[-1]
        return output

    def A_dagger(self, y, **kwargs):
        if self.fan_beam:
            y = self.filter(y)
            output = (
                self.A_adjoint(y, **kwargs) * PI.item() / (2 * len(self.radon.theta))
            )
            if self.normalize:
                output = output * output.shape[-1] ** 2
        else:
            output = self.iradon(y)
            if self.normalize:
                output = output * output.shape[-1]
        return output

    def A_adjoint(self, y, **kwargs):
        if self.fan_beam:
            assert (
                not self.img_width is None
            ), "Image size unknown. Apply forward operator or add it for initialization."
            # lazy implementation for the adjoint...
            adj = adjoint_function(
                self.A,
                (y.shape[0], y.shape[1], self.img_width, self.img_width),
                device=self.device,
                dtype=self.dtype,
            )
            return adj(y)
        else:
            # IRadon is not exactly the adjoint but a rescaled version of it...
            output = (
                self.iradon(y, filtering=False)
                / PI.item()
                * (2 * len(self.iradon.theta))
            )
            if self.normalize:
                output = output / output.shape[-1]
            return output


class TomographyWithAstra(LinearPhysics):
    r"""Computed Tomography operator with `astra-toolbox` backend.

    Mathematically, it is described as a ray transform
    :math:`R` which linearly integrates an object :math:`x` along straight
    lines

    .. math::
        y = Rx

    where :math:`y` is the set of line integrals, called sinogram in 2D, or
    radiographs in 3D. An object is typically scanned using a surrounding circular
    trajectory. Given different acquisition systems, the lines along which
    the integrals are computed follow different geometries:

    * parallel. (2d and 3d) Per view, all rays intersecting the
        object are parallel. In 2d, all rays live on the same plane, perpendicular
        to the axis of rotation.
    * fanbeam. (2d) All rays come from a single source and intersect the object
        at a certain angle. The detector consists of a 1d line of cells. Similar to
        the 2d "parallel", all rays live on the same plane, perpendicular to the axis of rotation.
    * conebeam. (3d) All rays come from a single source. The detector consists
        of a 2d grid of cells.

    .. note::

        The pseudo-inverse is computed using the filtered back-projection
        algorithm with a Ramp filter, and its equivalent for conebeam 3d the
        Feldkamp-Davis-Kress algorithm. This is not the exact linear pseudo-inverse
        of the Ray Transform, but it is a good approximation which is robust to noise.

    .. warning::

        Due to computational efficiency reasons, the projector and backprojector
        implemented in `astra` are not matched. The projector is typically ray-driven,
        and the backprojector is pixel-driven. The adjoint of the forward Ray Transform
        is approximated by rescaling the backprojector.

    :param tuple[int, ...] img_shape: Shape of the object grid, either a 2 or 3-element tuple, for respectively 2d or 3d.
    :param int | tuple[int, ...] num_detectors: In 2d, specify an integer for a single line of detector cells. In 3d, specify a 2-element tuple for (row,col) shape of the detector.
    :param int num_angles: Number of angular positions.
    :param tuple[float, float] angular_range: Angular range, defaults to (0,`math.pi`).
    :param float | tuple[float, float] detector_spacing: In 2d the width of a detector cell. In 3d a 2-element tuple specifying the (vertical, horizontal) dimensions of a detector cell. (default: 1.0)
    :param tuple[float, ...] object_spacing: In 2d, the (x,y) dimensions of a pixel in the reconstructed image. In 3d, the (x,y,z) dimensions of a voxel. (default: `(1.,1.)`)
    :param tuple[float, ...] | None aabb: Axis-aligned bounding-box of the reconstruction area [min_x, max_x, min_y, max_y, ...]. Optional argument, if specified, overrides `object_spacing`. (default: None)
    :param list[float] | None angles: List of angular positions in radii. Optional, if specified, overrides `num_angles` and `angular_range`. (default: None)
    :param str geometry_type: The type of geometry among `'parallel'`, `'fanbeam'` and `'conebeam'`. (default: `'parallel'`)
    :param dict | None geometry_parameters: Contains extra parameters specific to certain geometries. When `geometry_type='fanbeam' | 'conebeam'`, the dictionnary should contains the keys:

        - "source_radius" distance between the x-ray source and the rotation axis (default: 57.5)

        - "detector_radius" distance between the x-ray detector and the rotation axis (default: 57.5)
    :param torch.device | str device: The operator only supports CUDA computation. (default: `torch.device('cuda')`)

    |sep|
    """

    def __init__(
        self,
        img_shape: tuple[int, ...],
        num_detectors: int | tuple[int, ...],
        num_angles: int = 180,
        angular_range: tuple[float, float] = (0, math.pi),
        detector_spacing: float | tuple[float, float] = 1.0,
        object_spacing: tuple[float, ...] = (1.0, 1.0),
        aabb: tuple[float, ...] | None = None,
        angles: Iterable[float] | None = None,
        geometry_type: str = "parallel",
        geometry_parameters: dict[str, Any] | None = None,
        device: torch.device | str = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(img_shape) in (
            2,
            3,
        ), f"len(img_shape) is {len(img_shape)}, must be either 2 or 3 (for 2d and 3d respectively)"
        assert (
            "cuda" in device or device.type == "cuda"
        ), "`TomographyWithAstra` only supports CUDA computations."

        self.img_shape = img_shape
        self.is_2d = len(img_shape) == 2
        self.num_detectors = num_detectors
        # self.num_angles = num_angles
        # self.angular_range = angular_range
        # self.detector_spacing = detector_spacing
        # self.object_spacing = object_spacing
        self.geometry_type = geometry_type
        self.device = device

        if angles is None:
            angles = np.linspace(*angular_range, num_angles, endpoint=False)

        self.object_geometry = create_object_geometry(
            *img_shape, aabb=aabb, spacing=object_spacing, is_2d=self.is_2d
        )

        self.projection_geometry = create_projection_geometry(
            geometry_type=geometry_type,
            detector_spacing=detector_spacing,
            n_detector_pixels=num_detectors,
            angles=angles,
            is_2d=self.is_2d,
            geometry_parameters=geometry_parameters,
        )

        self.xray_transform = XrayTransform(
            object_geometry=self.object_geometry,
            projection_geometry=self.projection_geometry,
            is_2d=self.is_2d,
        )

        self.filter = RampFilter(dtype=torch.float32, device=self.device)

    @property
    def measurement_shape(self) -> tuple[int, ...]:
        if self.is_2d:
            return self.xray_transform.range_shape[1:]
        else:
            return self.xray_transform.range_shape

    @property
    def num_angles(self) -> int:
        return self.xray_transform.range_shape[1]

    def fbp_weighting(self, sinogram: torch.Tensor) -> torch.Tensor:
        sinogram_scaled = torch.clone(sinogram)

        if self.is_3d:
            # dimensions (H,W) of the 2D detector
            # A is the number of angles
            B, C, H, A, W = sinogram.shape

            # vecs.shape = (num_angles, 12) with
            # (sx, sy, sz, dx, dy, dz, ux, uy, uz, vx, vy, vz) coordinates
            # - (sx,sy,sz) is the position of the source
            # - (dx,dy,dz) is the center of the detector
            # - (ux, uy, uz) is the horizontal basis vector of the detector
            # - (vx, vy, vz) is the vertical basis vector of the detector
            vecs = torch.from_numpy(
                astra.geom_2vec(self.projection_geometry)["Vectors"]
            )

            source_object_distance = torch.linalg.norm(
                vecs[:, [0, 1, 2]], axis=1, keepdims=False
            )

            v_range = torch.arange(H, dtype=torch.float64) - (H - 1) / 2
            u_range = torch.arange(W, dtype=torch.float64) - (W - 1) / 2

            v_grid, u_grid = torch.meshgrid(v_range, u_range, indexing="ij")
            weights = torch.ones((H, W), dtype=torch.float, device=sinogram.device)
            for i in range(A):
                source_position = vecs[i, [0, 1, 2]]
                detector_center_position = vecs[i, [3, 4, 5]]

                u_basis = vecs[i, [6, 7, 8]]
                v_basis = vecs[i, [9, 10, 11]]

                detector_pixel_positions = (
                    u_grid[..., None] * u_basis
                    + v_grid[..., None] * v_basis
                    + detector_center_position
                )

                pixel_ray_lengths = torch.linalg.norm(
                    detector_pixel_positions - source_position, dim=-1
                )

                weights[:] = source_object_distance[i] / pixel_ray_lengths

                sinogram_scaled[:, :, :, i, :] *= weights

        sinogram_scaled *= self.xray_transform.detector_cell_v_length
        sinogram_scaled /= self.xray_transform.object_cell_volume

        return sinogram_scaled

    @property
    def is_3d(self) -> bool:
        return not self.is_2d

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection.

        :param torch.Tensor x: input of shape `(B,C,H,W)`
        :return: projection of shape `(B,C,n_angles,n_detectors)`
        """
        out = AutogradTransform.apply(x, self.xray_transform)

        return out

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        :param torch.Tensor y: input of shape `(B,C,n_angles,n_detectors)`
        :return: torch.Tensor filtered back-projection of shape `(B,C,H,W)`
        """

        filtered_y = self.filter(y, dim=-1)
        out = self.A_adjoint(self.fbp_weighting(filtered_y))
        out = out * math.pi / (2 * self.num_angles)

        return out

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        out = AutogradTransform.apply(y, self.xray_transform.T)

        return out
