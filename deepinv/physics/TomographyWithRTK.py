import torch
from deepinv.physics.forward import LinearPhysics
from types import MappingProxyType

from deepinv.physics.functional.rtk import (
    XrayTransformRTK,
    import_itk_rtk,
    build_rtk_trajectory,
    build_rtk_info,
    AutogradTransformRTK as AutogradTransform,
)


class TomographyWithRTK(LinearPhysics):
    r"""Computed Tomography operator with RTK CUDA backend.

    This operator implements a ray transform :math:`\forw{x}` using the RTK (Reconstruction Toolkit) GPU-accelerated forward and backprojectors. This implementation requires ITK with CudaCommon v2.1 and RTK v3. See the `RTK documentation <https://docs.openrtk.org/>`_ for more details.

    Mathematically, it computes line integrals of an object :math:`x` along X-ray paths:

    .. math::

        y = \forw x

    where :math:`y` represents projection data.

    Supported geometries:

    * ``fanbeam`` (2D-like using the 3D geometry)
    * ``conebeam`` (3D trajectory)

    The adjoint is implemented using RTK's ray-cast adjoint, which is the exact adjoint of the discretized forward operator, unlike :class:`TomographyWithAstra`, which uses an approximate adjoint.

    .. warning::

        Batch elements are processed sequentially, not in parallel.

    .. warning::

        The pseudo-inverse is approximated using filtered backprojection (FBP) reconstruction, or the Feldkamp-Davis-Kress (FDK) algorithm in the cone-beam case. This is not the exact linear pseudo-inverse of the ray transform.

    .. note::

        If you need a custom acquisition trajectory (e.g. non-circular, or
        loaded from a real scanner), you can still supply your own pre-built
        RTK geometry object via ``geometry``. The detector grid and
        reconstruction volume are still described from ``img_size``, ``n_detector_pixels``,
        ``pixel_spacing``, ``detector_spacing`` and ``bounding_box``;
        only the source/detector trajectory itself comes from ``geometry`` in
        that case, and ``angles``/``angular_range``/``geometry_parameters`` are
        ignored.

    :param tuple[int, ...] img_size: Shape of the object grid: a 2-element
        ``(n_rows, n_cols)`` tuple for ``geometry_type='fanbeam'``, or a
        3-element ``(n_rows, n_slices, n_cols)`` tuple for
        ``geometry_type='conebeam'``.
    :param int | torch.Tensor angles: Number of angular positions sampled
        uniformly in ``angular_range``, or a Tensor of angular positions in
        degrees. Ignored if a custom ``geometry`` is supplied. (default: 180)
    :param int | tuple[int, int], None n_detector_pixels: In ``fanbeam``, an
        integer giving the number of detector cells. In ``conebeam``, a
        2-element ``(n_rows, n_cols)`` tuple for the detector grid (an int is
        broadcast to both). (default: None, inferred from ``img_size``)
    :param tuple[float, float] angular_range: Angular range in degrees.
        Ignored if a custom ``geometry`` is supplied. (default: ``(0, 360)``,
        a full circular trajectory)
    :param float | tuple[float, ...] pixel_spacing: Voxel size(s) of the
        reconstruction grid: scalar, or ``(x, z)`` in ``fanbeam``, ``(x, y, z)``
        in ``conebeam``. (default: 1.0)
    :param float | tuple[float, float] detector_spacing: Detector cell size(s):
        scalar, or ``(vertical, horizontal)`` in ``conebeam``. (default: 1.0)
    :param float | tuple[float, float], None detector_origin: Origin of the
        detector grid: scalar in ``fanbeam``, or ``(vertical, horizontal)`` in
        ``conebeam``. If ``None`` (default), the detector is centered (i.e.
        the origin is computed automatically from ``n_detector_pixels`` and
        ``detector_spacing`` so the detector grid is symmetric about 0).
    :param tuple[float, ...], None bounding_box: Axis-aligned bounding box of the
        reconstruction volume, ``[min_0, max_0, min_1, max_1, ...]``. If
        provided, overrides ``pixel_spacing``. (default: None)
    :param float | tuple[float, ...], None volume_origin: Origin of the
        reconstruction volume grid: scalar or ``(x, z)`` in ``fanbeam``,
        scalar or ``(x, y, z)`` in ``conebeam``. If ``None`` (default), the
        volume is centered (i.e. the origin is computed automatically from
        ``img_size`` and ``pixel_spacing``/``bounding_box`` so the volume is
        symmetric about 0). Takes precedence over the origin implied by
        ``bounding_box`` if both are given.
    :param str geometry_type: Either ``'fanbeam'`` or ``'conebeam'``.
    :param dict[str, float] geometry_parameters: ``"source_radius"`` (distance
        from the X-ray source to the rotation axis) and ``"detector_radius"``
        (distance from the rotation axis to the detector). Ignored if a custom
        ``geometry`` is supplied. (default: ``{"source_radius": 80.0,
        "detector_radius": 20.0}``)
    :param geometry: Optional pre-built ``rtk.ThreeDCircularProjectionGeometry``
        object describing the source/detector
        trajectory. If given, it is used as-is and ``angles``,
        ``angular_range``, ``geometry_parameters`` are ignored; the number of
        projections is read directly off ``geometry``. If left as ``None``
        (the default), the trajectory is built automatically from ``angles``,
        ``angular_range`` and ``geometry_parameters``.
    :param bool verbose: If True, print geometry configuration.
    :param bool normalize: If True, normalize operator to unit norm.
    :param float ray_step_size: Step size along the ray. If left at ``0.0``,
        defaults to the volume spacing along the stacking axis.

    |sep|

    :Examples:

        Fully automatic geometry (typical use case):

        .. doctest::

            >>> physics = TomographyWithRTK(
            ...     img_size=(64, 64, 64),
            ...     angles=600,
            ...     angular_range=(0, 360),
            ...     n_detector_pixels=(100, 100),
            ...     geometry_type="conebeam",
            ...     geometry_parameters={"source_radius": 300.0, "detector_radius": 200.0},
            ...     normalize=False,
            ...     ray_step_size=0.5,
            ... )

        Custom trajectory, everything else still inferred from ``img_size`` etc.:

        .. doctest::

            >>> geometry = rtk.ThreeDCircularProjectionGeometry.New()
            >>> for i in range(600):
            ...     geometry.AddProjection(300, 500, i * 360.0 / 600)
            >>> physics = TomographyWithRTK(
            ...     geometry=geometry,
            ...     img_size=(64, 64, 64),
            ...     n_detector_pixels=(100, 100),
            ...     geometry_type="conebeam",
            ...     normalize=False,
            ...     ray_step_size=0.5,
            ... )
    """

    def __init__(
        self,
        img_size: tuple[int, ...],
        angles: int | torch.Tensor = 180,
        n_detector_pixels: int | tuple[int, int] | None = None,
        angular_range: tuple[float, float] = (0, 360),
        pixel_spacing: float | tuple[float, ...] = 1.0,
        detector_spacing: float | tuple[float, float] = 1.0,
        detector_origin: float | tuple[float, float] | None = None,
        bounding_box: tuple[float, ...] | None = None,
        volume_origin: float | tuple[float, ...] | None = None,
        geometry_type: str | None = None,
        geometry_parameters: dict[str, float] = MappingProxyType(
            {
                "source_radius": 80.0,
                "detector_radius": 20.0,
            }
        ),
        geometry: any = None,
        verbose: bool = False,
        normalize: bool = False,
        ray_step_size: float = 0.0,
        *args,
        **kwargs,
    ):

        itk, rtk = import_itk_rtk()

        super().__init__(*args, **kwargs)

        self._NB_STACK_VOL = 2
        self._CUDA_IMAGE_TYPE = itk.CudaImage[itk.F, 3]

        if geometry_type is None:
            geometry_type = "conebeam"

        if geometry_type not in ("fanbeam", "conebeam"):
            raise ValueError(
                f"geometry_type {geometry_type!r} unrecognized (expected 'fanbeam' or 'conebeam')"
            )

        # ---------------------------------------------------------
        # Trajectory: either provided directly, or built from
        # angles / angular_range / geometry_parameters.
        # ---------------------------------------------------------
        if geometry is not None:
            n_angles = len(geometry.GetGantryAngles())
        else:
            geometry, n_angles = build_rtk_trajectory(
                rtk=rtk,
                angles=angles,
                angular_range=angular_range,
                geometry_parameters=dict(geometry_parameters),
            )

        # ---------------------------------------------------------
        # Projection-stack and volume information computed from the parameters.
        # ---------------------------------------------------------
        projection_stack_information, volume_information = build_rtk_info(
            geometry_type=geometry_type,
            img_size=img_size,
            n_angles=n_angles,
            n_detector_pixels=n_detector_pixels,
            pixel_spacing=pixel_spacing,
            detector_spacing=detector_spacing,
            detector_origin=detector_origin,
            bounding_box=bounding_box,
            volume_origin=volume_origin,
            geometry=geometry,
            nb_stack_vol=self._NB_STACK_VOL,
        )

        self.geometry = geometry
        self.geometry_type = geometry_type
        self.is_2d = geometry_type == "fanbeam"

        self.projection_stack_information = projection_stack_information
        self.volume_information = volume_information

        self.xray_transform = XrayTransformRTK(
            geometry,
            projection_stack_information,
            volume_information,
            is_2d=self.is_2d,
        )

        if ray_step_size == 0.0:
            self.ray_step_size = self.volume_information["spacing"][1]
        else:
            self.ray_step_size = ray_step_size

        self.normalize = False
        if normalize:
            vol_size = self.volume_information["size"]
            if geometry_type == "fanbeam":
                logical_size = [vol_size[0], vol_size[2]]
            else:
                logical_size = vol_size
            self.operator_norm = self.compute_norm(
                torch.randn((1, 1, *logical_size), device="cuda")
            ).sqrt()
            self.normalize = True
        else:
            self.operator_norm = None

        if verbose:
            print("Projection stack information: ")
            print(self.projection_stack_information["size"][::-1])
            print(self.projection_stack_information["origin"])
            print(self.projection_stack_information["spacing"])

            print("Output volume information: ")
            print(self.volume_information["size"][::-1])
            print(self.volume_information["origin"])
            print(self.volume_information["spacing"])

    def fbp(self, y: torch.Tensor, **kwargs) -> torch.Tensor:

        B, C = y.shape[0:2]

        output = torch.zeros(
            (B, C) + self.xray_transform.domain_shape, dtype=y.dtype, device=y.device
        )
        self.xray_transform.fdk_on_batch(y, output, **kwargs)

        if self.is_2d:
            return output[:, :, :, 0]
        else:
            return output

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection.

        In 2D, the output is a sinogram of shape [B,C,A,N],
        with A the number of angular positions, and N the number of detector cells.
        In 3D, the output is of shape [B,C,A,V,N], with A the
        number of angular positions, and (V,N) the shape of the 2D detector grid,
        where V is the number of rows of the detector and N the number of columns.

        :param torch.Tensor x: input of shape [B,C,H,D,W] for conebeam and [B,C,H,W] for fanbeam
        :return: projection of shape [B,C,A,V,N] for conebeam and [B,C,A,N] for fanbeam

        :note The measurement shape differs from the `TomographyWithAstra` forward operator . As in TomographyWithRTK is to [B,C,A,V,N] and `TomographyWithAstra` [B,C,V,A,N]
        """

        out = AutogradTransform.apply(x, self.xray_transform)
        if self.normalize:
            out /= self.operator_norm

        return out

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Back projection, adjoint of the forward projector.

        In 2D, the input is a sinogram of shape [B,C,A,N],
        with A the number of angular positions, and N the number of detector cells.
        In 3D, the input is of shape [B,C,A,V,N], with A the
        number of angular positions, and (V,N) the shape of the 2D detector grid,
        where V is the number of rows of the detector and N the number of columns.

        :param torch.Tensor y: input of shape [B,C,A,V,N] for conebeam and [B,C,A,N] for fanbeam
        :return: scaled back-projection of shape [B,C,H,D,W] for conebeam and [B,C,H,W] for fanbeam
        """
        out = AutogradTransform.apply(y, self.xray_transform.T)
        if self.normalize:
            out /= self.operator_norm

        return out

    def A_dagger(
        self,
        y: torch.Tensor,
        fbp: bool = False,
        parker_angular_gap_threshold: float = 0.0,
        truncation_correction_padding: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the solution in :math:`x` to :math:`y = Ax` using a least squares solver. A faster approximation can be obtained by setting ``fbp=True``, which computes the filtered back-projection of the measurements, or the Feldkamp-Davis-Kress algorithm (FDK) in cone-beam 3D.

        .. warning::

            The filtered back-projection algorithm is not the exact linear pseudo-inverse of the Radon transform, but it is a good approximation that is robust to noise.

        :param torch.Tensor y: input of shape [B,C,A,V,N] for conebeam and [B,C,A,N] for fanbeam
        :param bool fbp: compute the inverse through the FDK algorithm
        :param float parker_angular_gap_threshold: Angular gap threshold (in degrees) at which :footcite:t:`parker1982optimal` weighting is used. Only used if ``fbp=True``.
        :param float truncation_correction_padding: Padding ratio applied to reduce truncation artefacts via :footcite:t:`ohnesorge2000efficient`. Only used if ``fbp=True``.
        :return: reconstruction of shape [B,C,H,D,W] for conebeam and [B,C,H,W] for fanbeam
        """

        if fbp:
            return self.fbp(
                y, parker_angular_gap_threshold=parker_angular_gap_threshold, truncation_correction_padding=truncation_correction_padding, **kwargs
            )
        else:
            return super(TomographyWithRTK, self).A_dagger(y, **kwargs)
