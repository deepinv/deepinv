from typing import Any, Optional, Union

import torch
import numpy as np

try:
    import astra
except ImportError:  # pragma: no cover
    astra = ImportError(
        "The astra-toolbox package is not installed."
    )  # pragma: no cover


class XrayTransform:
    r"""X-ray Transform operator with ``astra-toolbox`` backend.

    Uses the `astra-toolbox <https://astra-toolbox.com/>`_ to implement a ray-driven forward projector
    and a pixel-driven backprojector (:attr:`XrayTransform.T`).
    This class leverages the GPULink functionality of ``astra`` to share the underlying
    CUDA memory between torch Tensors and CUDA-based arrays use in ``astra``. The
    functionality is only implemented for 3D arrays, thus the underlying transforms
    are all 3D operators. For 2D transforms, the object is set to a flat volume with only 1 voxel depth.

    .. note::

        This transform does not handle batched and multi-channel inputs. It is
        handled by a custom :class:`torch.autograd.Function` that wraps the :class:`XrayTransform`.
        To handle standard PyTorch pipelines, :class:`XrayTransform` is instanciated inside a :class:`deepinv.physics.TomographyWithAstra` operator.

    :param dict[str, Any] projection_geometry: Dictionnary containing the parameters of the projection geometry. It is passed to the ``astra.create_projector()`` function to instanciate the projector.
    :param dict[str, Any] object_geometry:  Dictionnary containing the parameters of the object geometry. It is passed to the ``astra.create_projector()`` function to instanciate the projector.
    :param bool is_2d: Specifies if the geometry is flat (2D) or describe a real 3D reconstruction setup.
    """

    def __init__(
        self,
        projection_geometry: dict[str, Any],
        object_geometry: dict[str, Any],
        is_2d: bool = False,
    ):
        self.projection_geometry = projection_geometry
        self.object_geometry = object_geometry
        self.is_2d = is_2d

        self._astra_projector_id = astra.create_projector(
            "cuda3d", self.projection_geometry, self.object_geometry
        )

    @property
    def domain_shape(self) -> tuple:
        """The shape of the input volume."""
        return astra.geom_size(self.object_geometry)

    @property
    def range_shape(self) -> tuple:
        """The shape of the output projection."""
        return astra.geom_size(self.projection_geometry)

    @property
    def object_cell_volume(self) -> float:
        """The volume in physical units of a voxel."""
        volume = 1.0
        volume *= (
            self.object_geometry["option"]["WindowMaxX"]
            - self.object_geometry["option"]["WindowMinX"]
        ) / self.object_geometry["GridColCount"]
        volume *= (
            self.object_geometry["option"]["WindowMaxY"]
            - self.object_geometry["option"]["WindowMinY"]
        ) / self.object_geometry["GridRowCount"]
        volume *= (
            self.object_geometry["option"]["WindowMaxZ"]
            - self.object_geometry["option"]["WindowMinZ"]
        ) / self.object_geometry["GridSliceCount"]

        return volume

    @property
    def detector_cell_v_length(self) -> float:
        """The vertical length of a detector cell."""
        if "vec" in self.projection_geometry["type"]:
            return np.sqrt(
                (self.projection_geometry["Vectors"][1, [6, 7, 8]] ** 2).sum()
            )
        else:
            return self.projection_geometry["DetectorSpacingY"]

    @property
    def detector_cell_u_length(self) -> float:
        """The horizontal length of a detector cell."""
        if "vec" in self.projection_geometry["type"]:
            return np.sqrt(
                (self.projection_geometry["Vectors"][0, [6, 7, 8]] ** 2).sum()
            )
        else:
            return self.projection_geometry["DetectorSpacingX"]

    @property
    def detector_cell_area(self) -> float:
        """The surface in physical units of a detector pixel."""
        return self.detector_cell_v_length * self.detector_cell_u_length

    @property
    def source_radius(self) -> float:
        """The distance between the source and the axis of rotation."""
        if not hasattr(self, "_source_radius"):
            if "vec" in self.projection_geometry["type"]:
                self._source_radius = np.sqrt(
                    (
                        astra.geom_2vec(self.projection_geometry)["Vectors"][
                            :, [0, 1, 2]
                        ]
                        ** 2
                    ).sum(axis=1)
                ).mean()
            else:
                self._source_radius = self.projection_geometry["DistanceOriginSource"]

        return self._source_radius

    @property
    def detector_radius(self) -> float:
        """The distance between the center of the detector and the axis of rotation."""
        if not hasattr(self, "_detector_radius"):
            if "vec" in self.projection_geometry["type"]:
                self._detector_radius = np.sqrt(
                    (
                        astra.geom_2vec(self.projection_geometry)["Vectors"][
                            :, [3, 4, 5]
                        ]
                        ** 2
                    ).sum(axis=1)
                ).mean()
            else:
                self._detector_radius = self.projection_geometry[
                    "DistanceOriginDetector"
                ]

        return self._detector_radius

    @property
    def magnitude(self) -> float:
        """The magnification factor induced by the fan/cone geometry."""
        if "cone" in self.projection_geometry["type"]:
            return (self.detector_radius + self.source_radius) / self.source_radius
        else:
            return 1.0

    def __call__(
        self, x: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Forward projection.

        :param torch.Tensor x: Tensor of shape [1,H,W] in 2D, or [D,H,W] in 3D.
        :param torch.Tensor, None out: To avoid unecessary copies, provide tensor of shape [...,A,N]
        to store output results
        :return: Sinogram of shape [1,A,N] in 2D or set of sinograms [V,A,N] in 3D.
        """

        assert (
            x.shape == self.domain_shape
        ), f"Input shape {x.shape} does not match expected shape {self.domain_shape}"

        if out is None:
            out = torch.zeros(self.range_shape, dtype=torch.float32, device=x.device)

        self._forward_projection(x, out)

        return out

    @property
    def T(self):
        """Implements and returns the adjoint of the transform operator."""
        parent = self

        class _Adjoint:
            def __init__(self):
                self.is_2d = parent.is_2d

            @property
            def domain_shape(self) -> tuple:
                """The shape of the input projection."""
                return parent.range_shape

            @property
            def range_shape(self) -> tuple:
                """The shape of the output volume."""
                return parent.domain_shape

            def __call__(
                self, x: torch.Tensor, out: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                r"""Backprojection.

                :param torch.Tensor x: Tensor of shape [1,A,N] in 2D, or [V,A,N] in 3D.
                :param torch.Tensor, None out: To avoid unecessary copies, provide tensor of shape [...,H,W]
                to store output results
                :return: Image of shape [1,H,W] in 2D or volume [D,H,W] in 3D.
                """
                assert (
                    x.shape == self.domain_shape
                ), f"Input shape {x.shape} does not match expected shape {self.domain_shape}"

                if out is None:
                    out = torch.zeros(
                        self.range_shape, dtype=torch.float32, device=x.device
                    )

                parent._backprojection(x, out)
                if self.is_2d:
                    # necessary scaling in fanbeam to obtain decent approximated adjoint
                    out /= parent.magnitude

                return out

            @property
            def T(self):
                return parent

        return _Adjoint()

    def _forward_projection(self, x: torch.Tensor, out: torch.Tensor) -> None:
        assert (
            x.shape == self.domain_shape
        ), f"Input shape {x.shape} does not match expected shape {self.domain_shape}"
        assert (
            out.shape == self.range_shape
        ), f"Output shape {out.shape} does not match expected shape {self.range_shape}"

        _astra_volume_link = _create_astra_link(x)
        _astra_projection_link = _create_astra_link(out)

        astra.experimental.direct_FPBP3D(
            self._astra_projector_id,
            _astra_volume_link,
            _astra_projection_link,
            1,
            "FP",
        )

    def _backprojection(self, y: torch.Tensor, out: torch.Tensor) -> None:
        assert (
            y.shape == self.range_shape
        ), f"Input shape {y.shape} does not match expected shape {self.range_shape}"
        assert (
            out.shape == self.domain_shape
        ), f"Output shape {out.shape} does not match expected shape {self.domain_shape}"

        _astra_projection_link = _create_astra_link(y)
        _astra_volume_link = _create_astra_link(out)

        astra.experimental.direct_FPBP3D(
            self._astra_projector_id,
            _astra_volume_link,
            _astra_projection_link,
            1,
            "BP",
        )


def _create_astra_link(data: torch.Tensor) -> object:
    """Return an `astra` GPULink.

    :param torch.Tensor data: CUDA torch.Tensor
    :return: GPULink, instance of a utility class which holds the pointer of the underlying CUDA array, its shape and the the stride of the data.
    """

    assert data.is_contiguous(), "Data must be contiguous"
    assert data.dtype == torch.float32, "Data must be of type float32"
    assert len(data.shape) == 3, "Data must be 3D"

    z, y, x = data.shape

    astra_link = astra.pythonutils.GPULink(
        data.data_ptr(),
        x,
        y,
        z,
        x * 4,  # stride in bytes (float32 = 4 bytes)
    )

    return astra_link


class AutogradTransform(torch.autograd.Function):
    r"""Custom torch.autograd.Function for XrayTransform

    See https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
    for more information.
    """

    @staticmethod
    def forward(input: torch.Tensor, op: XrayTransform) -> torch.Tensor:
        """Forward autograd.

        The ``astra-toolbox`` does not handle batched computation, the transform
        is applied sequantially by iterating over the batch and channel dimension
        (e.g. Learned Primal-Dual).

        :param torch.Tensor input: Batched with channel input Tensor of shape [B,C,...].
        :param XrayTransform op: XrayTransform operator which performs underlying computations.
        :return:
        """
        if op.is_2d:
            B, C, H, W = input.shape
            assert (1, H, W) == op.domain_shape, f"{(1,H,W)} != {op.domain_shape}"

            output = torch.empty(
                (B, C) + op.range_shape[1:], dtype=input.dtype, device=input.device
            )

            for i in range(B):
                for j in range(C):
                    op(input[i, j : j + 1], out=output[i, j : j + 1])

        else:
            B, C, D, H, W = input.shape
            assert (D, H, W) == op.domain_shape, f"{(D,H,W)} != {op.domain_shape}"

            output = torch.empty(
                (B, C) + op.range_shape, dtype=input.dtype, device=input.device
            )

            for i in range(B):
                for j in range(C):
                    op(input[i, j], out=output[i, j])

        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        input_tensor, operator = inputs

        ctx.operator = operator

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        r"""Backward autograd.

        It is simply the forward autograd applied with the adjoint kernel.
        """
        op = ctx.operator

        return AutogradTransform.apply(grad_output, op.T), None


def create_projection_geometry(
    geometry_type: str,
    detector_spacing: Union[int, tuple[int, int]],
    n_detector_pixels: Union[int, tuple[int, int]],
    angles: torch.Tensor,
    is_2d: bool = False,
    geometry_parameters: Optional[dict[str, Any]] = None,
    geometry_vectors: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Utility function that produces a "projection geometry", a dict of parameters
    used by ``astra`` to parametrize the geometry of the detector and the x-ray source.

    :param str geometry_type: The type of geometry among ``'parallel'``, ``'fanbeam'`` in 2D and ``'parallel'`` and ``'conebeam'`` in 3D.
    :param int | tuple[int, int]: In 2D the width of a detector cell. In 3D a 2-element tuple specifying the (vertical, horizontal) dimensions of a detector cell. (default: 1.0)
    :param torch.Tensor angles: Tensor containing angular positions in radii.
    :param bool is_2d: Boolean specifying if the parameters define a 2D slice or a 3D volume.
    :param dict[str, str], None geometry_parameters: Contains extra parameters specific to certain geometries. When ``geometry_type='fanbeam'`` or  ``'conebeam'``, the dictionnary should contains the keys

        - "source_radius" distance between the x-ray source and the rotation axis, denoted :math:`D_{s0}` (default: 80.)

        - "detector_radius" distance between the x-ray detector and the rotation axis, denoted :math:`D_{0d}` (default: 20.)
    :param torch.Tensor, None geometry_vectors: Alternative way to describe a 3D geometry. It is a torch.Tensor of shape [num_angles, 12], where for each angular position of index ``i`` the row consists of a vector of size (12,) with

        - ``(sx, sy, sz)``: the position of the source,

        - ``(dx, dy, dz)``: the center of the detector,

        - ``(ux, uy, uz)``: the horizontal unit vector of the detector,

        - ``(vx, vy, vz)``: the vertical unit vector of the detector.

        When specified, ``geometry_vectors`` overrides ``detector_spacing``, ``angles`` and ``geometry_parameters``. It is particularly useful to build the geometry for the `Walnut-CBCT dataset <https://zenodo.org/records/2686726>`_, where the acquisition parameters are provided via such vectors.
    """

    if is_2d:
        if geometry_vectors is None:
            if type(detector_spacing) is not float:
                raise ValueError(
                    f"For 2d geometry, argument `detector_spacing` should be a float specifying the width of a detector cell, got {type(detector_spacing)}"
                )
        if type(n_detector_pixels) is not int:
            raise ValueError(
                f"For 2d geometry, argument `n_detector_pixels` should be a int specifying the number of a cells in the detector line, got {type(n_detector_pixels)}"
            )
    else:
        if geometry_vectors is None:
            if len(detector_spacing) != 2:
                raise ValueError(
                    f"For 3D geometry, argument `detector_spacing` should be a tuple of 2 float the vertical and horizontal dimensions of a detector cell, got {len(detector_spacing)}"
                )
        if len(n_detector_pixels) != 2:
            raise ValueError(
                f"For 3D geometry, argument `n_detector_pixels` should be a tuple of 2 int specifying the number of (columns,rows) in the detector grid {len(n_detector_pixels)}"
            )

    if geometry_parameters is not None:
        source_radius = geometry_parameters.get("source_radius", 80.0)
        detector_radius = geometry_parameters.get("detector_radius", 20.0)

    angles = angles.tolist()

    # The astra-toolbox does not support GPU linking for 2D data. Thus, when creating a projection geometry in 2D, we actually create a flat 3D geometry, i.e. a 2D detector with only one row of cells.
    # GPULink python API for 2D data:  https://github.com/astra-toolbox/astra-toolbox/discussions/391
    if is_2d:
        detector_row_count = 1
        detector_col_count = n_detector_pixels

        if geometry_type == "parallel":
            projection_geometry = astra.create_proj_geom(
                "parallel3d",
                detector_spacing,  # horizontal spacing
                1.0,  # vertical spacing is unit length
                detector_row_count,
                detector_col_count,
                angles,
            )
        elif geometry_type == "fanbeam":
            projection_geometry = astra.create_proj_geom(
                "cone",
                detector_spacing,  # horizontal spacing
                1.0,  # vertical spacing is unit length
                detector_row_count,
                detector_col_count,
                angles,
                source_radius,
                detector_radius,
            )
        else:
            raise ValueError(
                f'got geometry_type="{geometry_type}", in 2D should be one of ["parallel","fanbeam"]'
            )
    else:
        detector_row_count, detector_col_count = n_detector_pixels

        if geometry_type == "parallel":
            if geometry_vectors is not None:
                projection_geometry = astra.create_proj_geom(
                    "parallel3d_vec",
                    detector_row_count,
                    detector_col_count,
                    geometry_vectors.numpy(force=True),
                )
            else:
                projection_geometry = astra.create_proj_geom(
                    "parallel3d",
                    detector_spacing[1],  # horizontal spacing
                    detector_spacing[0],  # vertical spacing
                    detector_row_count,
                    detector_col_count,
                    angles,
                )
        elif geometry_type == "fanbeam":
            raise NotImplementedError("fanbeam geometry is not implemented in 3D")

        elif geometry_type == "conebeam":
            if geometry_vectors is not None:
                projection_geometry = astra.create_proj_geom(
                    "cone_vec",
                    detector_row_count,
                    detector_col_count,
                    geometry_vectors.numpy(force=True),
                )
            else:
                projection_geometry = astra.create_proj_geom(
                    "cone",
                    detector_spacing[1],  # horizontal spacing
                    detector_spacing[0],  # vertical spacing
                    detector_row_count,
                    detector_col_count,
                    angles,
                    source_radius,
                    detector_radius,
                )
        else:
            raise ValueError(
                f'got geometry_type="{geometry_type}", in 3D should be one of ["parallel","conebeam"]'
            )

    return projection_geometry


def create_object_geometry(
    n_rows: int,
    n_cols: int,
    n_slices: int = 1,
    is_2d: bool = True,
    spacing: tuple[float, ...] = (1.0, 1.0),
    bounding_box: Optional[tuple[float, ...]] = None,
) -> dict[str, Any]:
    """Utility function that produces a "volume geometry", a dict of parameters
    used by ``astra`` to parametrize the reconstruction grid.

    :param int n_rows: Number of rows.
    :param int n_cols: Number of columns.
    :param int n_slices: Number of slices. It is automatically set to 1 when ``is_2d=True``.
    :param bool is_2D: Boolean specifying if the parameters define a 2D slice or a 3D volume.
    :param tuple[float, ...] spacing: Dimensions of reconstruction cell along the axis [x,y,...].
    :param tuple[float, ...] bounding_box: Extent of the reconstruction area [min_x, max_x, min_y, max_y, ...]
    """

    if is_2d:
        if bounding_box is not None:
            if len(bounding_box) != 4:
                raise ValueError(
                    f"For 2D geometry, argument `bounding_box` should be a tuple of size 4 for with (min_x,max_x,min_y,max_y), got len(bounding_box)={len(bounding_box)}"
                )
        else:
            if len(spacing) != 2:
                raise ValueError(
                    f"For 2D geometry, `spacing` should be a tuple of size 2 with dimensions (length_x,length_y) of a pixel, got len(spacing)={len(spacing)}"
                )
    else:
        if bounding_box is not None:
            if len(bounding_box) != 6:
                raise ValueError(
                    f"For 3D geometry, argument `bounding_box` should be a tuple of size 6 for with (min_x,max_x,min_y,max_y,min_z,max_z), got len(bounding_box)={len(bounding_box)}"
                )
        else:
            if len(spacing) != 3:
                raise ValueError(
                    f"For 23 geometry, `spacing` should be a tuple of size 3 with dimensions (length_x,length_y, length_z) of a voxel, got len(spacing)={len(spacing)}"
                )

    if is_2d:
        n_slices = 1
        if bounding_box is not None:
            min_x, max_x, min_y, max_y = bounding_box
        else:
            min_x, max_x = -n_cols / 2 * spacing[0], n_cols / 2 * spacing[0]
            min_y, max_y = -n_rows / 2 * spacing[1], n_rows / 2 * spacing[1]

        # assume the slice dimension is unit length to avoid scaling issues with ASTRA
        min_z, max_z = -0.5, 0.5

    else:
        if bounding_box is not None:
            min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
        else:
            min_x, max_x = -n_cols / 2 * spacing[0], n_cols / 2 * spacing[0]
            min_y, max_y = -n_rows / 2 * spacing[1], n_rows / 2 * spacing[1]
            min_z, max_z = -n_slices / 2 * spacing[2], n_slices / 2 * spacing[2]

    spacing = [
        (max_x - min_x) / n_cols,
        (max_y - min_y) / n_rows,
        (max_z - min_z) / n_slices,
    ]

    object_geometry = astra.create_vol_geom(
        n_rows, n_cols, n_slices, min_x, max_x, min_y, max_y, min_z, max_z
    )

    return object_geometry
