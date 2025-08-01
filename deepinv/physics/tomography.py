from typing import Any, Union, Optional
import math
import torch
from deepinv.physics.forward import LinearPhysics

from deepinv.physics.functional import Radon, IRadon, RampFilter, ApplyRadon
from deepinv.physics import adjoint_function

from deepinv.physics.functional import XrayTransform
from deepinv.physics.functional.astra import (
    AutogradTransform,
    create_projection_geometry,
    create_object_geometry,
)
from warnings import warn

try:
    import astra

    # NOTE: This import is used by its side effects.
    from astra import experimental  # noqa: F401
except ImportError:  # pragma: no cover
    astra = ImportError(
        "The astra-toolbox package is not installed."
    )  # pragma: no cover


class Tomography(LinearPhysics):
    r"""
    (Computed) Tomography operator.

    The Radon transform is the integral transform which takes a square image :math:`x` defined on the plane to a function
    :math:`y=\forw{x}` defined on the (two-dimensional) space of lines in the plane, whose value at a particular line is equal
    to the line integral of the function over that line.

    .. note::

        The pseudo-inverse is computed using the filtered back-projection algorithm with a Ramp filter.
        This is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation which is
        robust to noise.

    .. note::

        The measurements are not normalized by the image size, thus the norm of the operator depends on the image size.

    .. warning::

        The adjoint operator has small numerical errors due to interpolation. Set ``adjoint_via_backprop=True`` if you want to use the exact adjoint (computed via autograd).

    :param int, torch.Tensor angles: These are the tomography angles. If the type is ``int``, the angles are sampled uniformly between 0 and 360 degrees.
        If the type is :class:`torch.Tensor`, the angles are the ones provided (e.g., ``torch.linspace(0, 180, steps=10)``).
    :param int img_width: width/height of the square image input.
    :param bool circle: If ``True`` both forward and backward projection will be restricted to pixels inside a circle
        inscribed in the square image.
    :param bool parallel_computation: if ``True``, all projections are performed in parallel. Requires more memory but is faster on GPUs.
    :param bool adjoint_via_backprop: if ``True``, the adjoint will be computed via :func:`deepinv.physics.adjoint_function`. Otherwise the inverse Radon transform is used.
        The inverse Radon transform is computationally cheaper (particularly in memory), but has a small adjoint mismatch.
        The backprop adjoint is the exact adjoint, but might break random seeds since it backpropagates through :func:`torch.nn.functional.grid_sample`, see the note `here <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html>`_.
    :param bool fbp_interpolate_boundary: the :func:`filtered back-projection <deepinv.physics.Tomography.A_dagger>` usually contains streaking artifacts on the boundary due to padding. For ``fbp_interpolate_boundary=True``
        these artifacts are corrected by cutting off the outer two pixels of the FBP and recovering them by interpolating the remaining image. This option
        only makes sense if ``circle`` is set to ``False``. Hence it will be ignored if ``circle`` is True.
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

        The default values are adapted from the geometry in :footcite:t:`khalil2023hyperspectral`.
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
        tensor([[[[ 0.0000, -0.1791, -0.1719],
                  [-0.5713, -0.4521, -0.5177],
                  [ 0.0340,  0.1448,  0.2334],
                  [ 0.0000, -0.0448, -0.0430]]]])

        Tomography operator with 3 uniformly sampled angles in [0, 360] for 3x3 image:

        >>> from deepinv.physics import Tomography
        >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 4, 4)  # Define random 4x4 image
        >>> physics = Tomography(angles=3, img_width=4, circle=True)
        >>> physics(x)
        tensor([[[[ 0.0000, -0.1806,  0.0500],
                  [-0.5713, -0.6076, -0.6815],
                  [ 0.0340,  0.3175,  0.0167],
                  [ 0.0000, -0.0452,  0.0989]]]])


    """

    def __init__(
        self,
        angles,
        img_width,
        circle=False,
        parallel_computation=True,
        adjoint_via_backprop=False,
        fbp_interpolate_boundary=False,
        normalize=False,
        fan_beam=False,
        fan_parameters=None,
        device=torch.device("cpu"),
        dtype=torch.float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(angles, int) or isinstance(angles, float):
            theta = torch.linspace(0, 180, steps=angles + 1, device=device)[:-1].to(
                device
            )
        else:
            theta = torch.tensor(angles).to(device)

        self.register_buffer("theta", theta)

        self.fan_beam = fan_beam
        self.adjoint_via_backprop = adjoint_via_backprop
        if fan_beam or adjoint_via_backprop:
            self._auto_grad_adjoint_fn = None
            self._auto_grad_adjoint_input_shape = (1, 1, img_width, img_width)
        self.fbp_interpolate_boundary = fbp_interpolate_boundary
        if circle:
            # interpolate boundary does not make sense if circle is True
            warn(
                "The argument fbp_interpolate_boundary=True is not applicable if circle=True. The value fbp_interpolate_boundary will be changed to False..."
            )
            self.fbp_interpolate_boundary = False
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
        if self.fan_beam or self.adjoint_via_backprop:
            output = self.radon(x)
        else:
            output = ApplyRadon.apply(x, self.radon, self.iradon, False)
        if self.normalize:
            output = output / x.shape[-1]
        return output

    def A_dagger(self, y, **kwargs):
        r"""
        Computes the filtered back-projection (FBP) of the measurements.

        .. warning::

            The filtered back-projection algorithm is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation that is robust to noise.

        .. tip::

            By default, the FBP reconstruction can display artifacts at the borders. Set ``fbp_interpolate_boundary=True`` to remove them with padding.


        :param torch.Tensor y: measurements
        :return torch.Tensor: noisy measurements
        """
        if self.fan_beam or self.adjoint_via_backprop:
            if self.fan_beam:
                y = self.filter(y)
            else:
                y = self.iradon.filter(y)
            output = (
                self.A_adjoint(y, **kwargs) * torch.pi / (2 * len(self.radon.theta))
            )
            if self.normalize:
                output = output * output.shape[-1] ** 2
        else:
            y = self.iradon.filter(y)
            output = (
                ApplyRadon.apply(y, self.radon, self.iradon, True)
                * torch.pi
                / (2 * len(self.iradon.theta))
            )
            if self.normalize:
                output = output * output.shape[-1]
        if self.fbp_interpolate_boundary:
            output = output[:, :, 2:-2, 2:-2]
            output = torch.nn.functional.pad(output, (2, 2, 2, 2), mode="replicate")
        return output

    def A_adjoint(self, y, **kwargs):
        r"""
        Computes adjoint of the tomography operator.

        .. warning::

            The default adjoint operator has small numerical errors due to interpolation. Set ``adjoint_via_backprop=True`` if you want to use the exact adjoint (computed via autograd).

        :param torch.Tensor y: measurements
        :return torch.Tensor: noisy measurements
        """
        if self.fan_beam or self.adjoint_via_backprop:
            if self.img_width is None:
                raise ValueError(
                    "Image size unknown. Apply forward operator or add it for initialization."
                )
            # lazy implementation for the adjoint...
            if (
                self._auto_grad_adjoint_fn is None
                or self._auto_grad_adjoint_input_shape
                != (y.size(0), y.size(1), self.img_width, self.img_width)
            ):
                self._auto_grad_adjoint_fn = adjoint_function(
                    self.A,
                    (y.shape[0], y.shape[1], self.img_width, self.img_width),
                    device=self.device,
                    dtype=self.dtype,
                )
                self._auto_grad_adjoint_input_shape = (
                    y.size(0),
                    y.size(1),
                    self.img_width,
                    self.img_width,
                )

            return self._auto_grad_adjoint_fn(y)
        else:
            output = ApplyRadon.apply(y, self.radon, self.iradon, True)
            if self.normalize:
                output = output / output.shape[-1]
            return output


class TomographyWithAstra(LinearPhysics):
    r"""Computed Tomography operator with `astra-toolbox <https://astra-toolbox.com/>`_ backend.
    It is more memory efficient than the :class:`deepinv.physics.Tomography` operator and support 3D geometries.
    See documentation of :class:`deepinv.physics.functional.XrayTransform` for more
    information on the ``astra`` wrapper.

    Mathematically, it is described as a ray transform
    :math:`A` which linearly integrates an object :math:`x` along straight
    lines

    .. math::
        y = \forw{x}

    where :math:`y` is the set of line integrals, called sinogram in 2D, or
    radiographs in 3D. An object is typically scanned using a surrounding circular
    trajectory. Given different acquisition systems, the lines along which
    the integrals are computed follow different geometries:

    * parallel. (2D and 3D)
        Per view, all rays intersecting the object are parallel. In 2D, all rays live on the same plane, perpendicular
        to the axis of rotation.
    * fanbeam. (2D)
        Per view, all rays come from a single source and intersect the object at a certain angle. The detector consists of a 1d line of cells. Similar to
        the 2D "parallel", all rays live on the same plane, perpendicular to the axis of rotation.
    * conebeam. (3D)
        Per view, all rays come from a single source. The detector consists of a 2D grid of cells. Apart from the central plane, the set of rays coming onto
        a line of cells live on a tilted plane.

    .. note::

        The pseudo-inverse is computed using the filtered back-projection
        algorithm with a Ramp filter, and its equivalent for conebeam 3D, the
        Feldkamp-Davis-Kress algorithm. This is not the exact linear pseudo-inverse
        of the Ray Transform, but it is a good approximation which is robust to noise.

    .. note::

        In the default configuration, reconstruction cells and detector cells are
        set to have isotropic unit lengths. The geometry is set to 2D parallel
        and matches the default configuration of the :class:`deepinv.physics.Tomography` operator with
        ``circle=False``.

    .. warning::

        Due to computational efficiency reasons, the projector and backprojector
        implemented in ``astra`` are not matched. The projector is typically ray-driven,
        while the backprojector is pixel-driven. The adjoint of the forward Ray Transform
        is approximated by rescaling the backprojector.

    .. warning::

        The :class:`deepinv.physics.functional.XrayTransform` used in :class:`deepinv.physics.TomographyWithAstra` sequentially processes batch elements, which can make the 2D parallel beam operator significantly slower than its native torch counterpart with :class:`deepinv.physics.Tomography` (though still more memory-efficient).

    :param tuple[int, ...] img_size: Shape of the object grid, either a 2 or 3-element tuple, for respectively 2D or 3D.
    :param int num_angles: Number of angular positions sampled uniformly in ``angular_range``. (default: 180)
    :param int | tuple[int, ...], None num_detectors: In 2D, specify an integer for a single line of detector cells. In 3D, specify a 2-element tuple for (row,col) shape of the detector. (default: None)
    :param tuple[float, float] angular_range: Angular range, defaults to ``(0, torch.pi)``.
    :param float | tuple[float, float] detector_spacing: In 2D the width of a detector cell. In 3D a 2-element tuple specifying the (vertical, horizontal) dimensions of a detector cell. (default: 1.0)
    :param tuple[float, ...] object_spacing: In 2D, the (x,y) dimensions of a pixel in the reconstructed image. In 3D, the (x,y,z) dimensions of a voxel. (default: ``(1.,1.)``)
    :param tuple[float, ...], None bounding_box: Axis-aligned bounding-box of the reconstruction area [min_x, max_x, min_y, max_y, ...]. Optional argument, if specified, overrides argument ``object_spacing``. (default: None)
    :param torch.Tensor, None angles: Tensor containing angular positions in radii. Optional, if specified, overrides arguments ``num_angles`` and ``angular_range``. (default: None)
    :param str geometry_type: The type of geometry among ``'parallel'``, ``'fanbeam'`` in 2D and ``'parallel'`` and ``'conebeam'`` in 3D. (default: ``'parallel'``)
    :param dict[str, Any] geometry_parameters: Contains extra parameters specific to certain geometries. When ``geometry_type='fanbeam'`` or  ``'conebeam'``, the dictionnary should contains the keys

        - ``"source_radius"``: the distance between the x-ray source and the rotation axis, denoted :math:`D_{s0}`, (default: 80.),

        - ``"detector_radius"``: the distance between the x-ray detector and the rotation axis, denoted :math:`D_{0d}`. (default: 20.)

    :param torch.Tensor, None geometry_vectors: Alternative way to describe a 3D geometry. It is a torch.Tensor of shape [num_angles, 12], where for each angular position of index ``i`` the row consists of a vector of size (12,) with

        - ``(sx, sy, sz)``: the position of the source,

        - ``(dx, dy, dz)``: the center of the detector,

        - ``(ux, uy, uz)``: the horizontal unit vector of the detector,

        - ``(vx, vy, vz)``: the vertical unit vector of the detector.

        When specified, ``geometry_vectors`` overrides ``detector_spacing``, ``num_angles``/``angles`` and ``geometry_parameters``. It is particularly useful to build the geometry for the `Walnut-CBCT dataset <https://zenodo.org/records/2686726>`_, where the acquisition parameters are provided via such vectors.
    :param bool normalize: If ``True`` :func:`A` and :func:`A_adjoint` are normalized so that the operator has unit norm. (default: ``False``)
    :param torch.device | str device: The operator only supports CUDA computation. (default: ``torch.device('cuda')``)

    |sep|

    :Examples:

        Tomography operator with a 2D ``'fanbeam'`` geometry, 10 uniformly sampled angles in ``[0,2*torch.pi]``, a detector line of 5 cells with length 2., a source-radius of 20.0 and a detector_radius of 20.0 for 5x5 image:

        .. doctest::
           :skipif: astra is None or not cuda_available

            >>> from deepinv.physics import TomographyWithAstra
            >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
            >>> x = torch.randn(1, 1, 5, 5, device='cuda') # Define random 5x5 image
            >>> physics = TomographyWithAstra(
            ...        img_size=(5,5),
            ...        num_angles=10,
            ...        angular_range=(0, 2*torch.pi),
            ...        num_detectors=5,
            ...        detector_spacing=2.0,
            ...        geometry_type='fanbeam',
            ...        geometry_parameters={
            ...            'source_radius': 20.,
            ...            'detector_radius': 20.
            ...        }
            ...    )
            >>> sinogram = physics(x)
            >>> print(sinogram)
            tensor([[[[-2.4262, -0.3840, -2.1681, -1.1024,  1.8009],
                    [-2.4597, -0.0198, -1.6027,  0.1117,  1.0543],
                    [-3.8424, -2.5034,  1.8132,  2.4666, -1.0440],
                    [-3.0843, -2.0380,  2.2693,  2.4964, -2.7098],
                    [ 0.6441, -2.2355, -0.2281,  0.2533, -1.3641],
                    [ 1.7683, -0.9205, -2.1681, -0.2436, -2.5756],
                    [ 0.4655,  0.3250, -1.6027, -0.6839, -2.4529],
                    [-2.4195,  3.1875,  1.8132, -2.3952, -3.5968],
                    [-1.6350,  1.4374,  2.2693, -2.2185, -3.7328],
                    [-1.9789,  0.1986, -0.2281, -1.7952, -0.3667]]]], device='cuda:0')

        Tomography operator with a 3D ``'conebeam'`` geometry, 10 uniformly sampled angles in ``[0,2*torch.pi]``, a detector grid of 5x5 cells of size (2.,2.), a source-radius of 20.0 and a detector_radius of 20.0 for a 5x5x5 volume:

        .. doctest::
           :skipif: astra is None or not cuda_available

            >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
            >>> x = torch.randn(1, 1, 5, 5, 5, device='cuda')  # Define random 5x5x5 volume
            >>> angles = torch.linspace(0, 2*torch.pi, steps=4)[:-1]
            >>> physics = TomographyWithAstra(
            ...        img_size=(5,5,5),
            ...        angles = angles,
            ...        num_detectors=(5,5),
            ...        object_spacing=(1.0,1.0,1.0),
            ...        detector_spacing=(2.0,2.0),
            ...        geometry_type='conebeam',
            ...        geometry_parameters={
            ...            'source_radius': 20.,
            ...            'detector_radius': 20.
            ...       }
            ...    )
            >>> sinogram = physics(x)
            >>> print(sinogram)
            tensor([[[[[-2.0464,  0.4064, -1.5184, -0.9225,  1.5369],
                    [-2.3398, -0.9323,  2.0437,  0.5806, -1.5659],
                    [-1.0852,  2.0659,  1.1105, -1.7271, -2.6104]],
            <BLANKLINE>
                    [[ 1.4757, -0.2731,  0.9386,  0.5791,  0.2995],
                    [-0.8362,  2.5918,  1.0941,  1.0576, -1.4501],
                    [-1.1313,  3.8354, -0.9572, -2.3721,  3.5149]],
            <BLANKLINE>
                    [[ 0.6392,  0.1564, -0.8063, -3.8958,  1.2547],
                    [ 0.5294, -1.0241, -0.1792, -0.5054, -1.4253],
                    [-1.1961, -1.6911,  0.4279, -1.3608,  0.9488]],
            <BLANKLINE>
                    [[ 0.5134,  2.1534, -3.8697,  0.3571,  0.1060],
                    [ 0.4687, -3.0669,  1.5911,  1.5235, -0.8031],
                    [-1.1990,  0.2637,  2.0889, -0.8894,  0.2550]],
            <BLANKLINE>
                    [[-1.4643, -0.2128,  1.3425,  2.8803, -0.6605],
                    [ 0.9605,  1.1056,  4.2324, -3.5795, -0.1718],
                    [ 0.9207,  1.6948,  1.6556, -1.6624,  0.9960]]]]], device='cuda:0')
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        num_angles: int = 180,
        num_detectors: Optional[Union[int, tuple[int, ...]]] = None,
        angular_range: tuple[float, float] = (0, torch.pi),
        detector_spacing: Union[float, tuple[float, float]] = 1.0,
        object_spacing: tuple[float, ...] = (1.0, 1.0),
        bounding_box: Optional[tuple[float, ...]] = None,
        angles: Optional[torch.Tensor] = None,
        geometry_type: str = "parallel",
        geometry_parameters: dict[str, Any] = {
            "source_radius": 80.0,
            "detector_radius": 20.0,
        },
        geometry_vectors: Optional[torch.Tensor] = None,
        normalize: bool = False,
        device: Union[torch.device, str] = torch.device("cuda"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(img_size) in (
            2,
            3,
        ), f"len(img_size) is {len(img_size)}, must be either 2 or 3 (for 2D and 3D respectively)"

        if torch.device(device).type != "cuda":
            warn(
                f"TomographyWithAstra only supports CUDA Tensors and CUDA operations, got device={device}",
                RuntimeWarning,
            )

        self.img_size = img_size
        self.is_2d = len(img_size) == 2
        self.num_detectors = (
            math.ceil(math.sqrt(2) * img_size[0])
            if num_detectors is None
            else num_detectors
        )
        self.geometry_type = geometry_type
        self.normalize = False
        self.device = device

        if angles is None:
            angles = torch.linspace(*angular_range, steps=num_angles + 1)[:-1]

        self.object_geometry = create_object_geometry(
            *img_size,
            bounding_box=bounding_box,
            spacing=object_spacing,
            is_2d=self.is_2d,
        )

        self.projection_geometry = create_projection_geometry(
            geometry_type=geometry_type,
            detector_spacing=detector_spacing,
            n_detector_pixels=self.num_detectors,
            angles=angles,
            is_2d=self.is_2d,
            geometry_parameters=geometry_parameters,
            geometry_vectors=geometry_vectors,
        )

        self.xray_transform = XrayTransform(
            object_geometry=self.object_geometry,
            projection_geometry=self.projection_geometry,
            is_2d=self.is_2d,
        )

        self.filter = RampFilter(dtype=torch.float32, device=self.device)

        if normalize:
            self.operator_norm = self.compute_norm(
                torch.randn(
                    self.img_size,
                    generator=torch.Generator(self.device).manual_seed(0),
                    device=self.device,
                )[None, None]
            ).sqrt()
            self.normalize = normalize

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
        r"""Scales the computation by the inverse number of views and
        object-to-dector cell ratio.

        In conebeam 3D, compute FDK weights to correct inflated distances due to
        tilted rays. Given coordinate :math:`(x,y)`  of a detector cell, the corresponding
        weight is :math:`\omega(x,y) = \frac{D_{s0}}{\sqrt{D_{sd}^2 + x^2 + y^2}}`.

        :param torch.Tensor sinogram: Sinogram of shape [B,C,...,A,N].
        :return: Weighted sinogram.
        """
        sinogram_scaled = torch.clone(sinogram)
        is_3d = len(sinogram.shape) == 5

        if self.geometry_type == "conebeam" and is_3d:
            # dimensions (V,N) are (col,row) of the 2D detector
            # A is the number of angles
            B, C, V, A, N = sinogram.shape

            # vecs.shape = (num_angles, 12) with
            # (sx, sy, sz, dx, dy, dz, ux, uy, uz, vx, vy, vz) coordinates
            # - (sx, sy, sz) is the position of the source
            # - (dx, dy, dz) is the center of the detector
            # - (ux, uy, uz) is the horizontal unit vector of the detector
            # - (vx, vy, vz) is the vertical unit vector of the detector
            vecs = torch.from_numpy(
                astra.geom_2vec(self.projection_geometry)["Vectors"]
            )

            source_object_distance = torch.linalg.norm(
                vecs[:, [0, 1, 2]], axis=1, keepdims=False
            )

            v_range = torch.arange(V, dtype=torch.float64) - (V - 1) / 2
            u_range = torch.arange(N, dtype=torch.float64) - (N - 1) / 2

            v_grid, u_grid = torch.meshgrid(v_range, u_range, indexing="ij")
            weights = torch.ones((V, N), dtype=torch.float, device=sinogram.device)
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
        sinogram_scaled *= torch.pi / (2 * self.num_angles)

        return sinogram_scaled

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection.

        :param torch.Tensor x: input of shape [B,C,...,H,W]
        :return: projection of shape [B,C,...,A,N]
        """
        out = AutogradTransform.apply(x, self.xray_transform)
        if self.normalize:
            out /= self.operator_norm

        return out

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Pseudo-inverse estimated using filtered back-projection.

        :param torch.Tensor y: input of shape [B,C,...,A,N]
        :return: torch.Tensor filtered back-projection of shape [B,C,...,H,W]
        """

        filtered_y = self.filter(y, dim=-1)
        out = self.A_adjoint(self.fbp_weighting(filtered_y))
        if self.normalize:
            out *= self.operator_norm**2

        return out

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Approximation of the adjoint.

        :param torch.Tensor y: input of shape [B,C,...,A,N]
        :return: torch.Tensor filtered back-projection of shape [B,C,...,H,W]
        """
        out = AutogradTransform.apply(y, self.xray_transform.T)
        if self.normalize:
            out /= self.operator_norm

        return out
