from __future__ import annotations
from typing import Any
from types import ModuleType

import torch
import math


def import_itk_rtk() -> (ModuleType, ModuleType):
    try:
        import itk
        from itk import RTK as rtk

        return (itk, rtk)
    except ImportError:  # pragma: no cover
        raise ImportError(
            "itk-rtk is required to use TomographyWithRTK. It can be installed with itk-rtk-cuda128."
        )


class XrayTransformRTK:

    def __init__(
        self,
        geometry: any,
        projection_stack_information: dict[str, Any],
        volume_information: dict[str, Any],
        is_2d: bool = False,
        ray_step_size: float = 0.0,
    ):
        itk, rtk = import_itk_rtk()

        self._CUDA_IMAGE_TYPE = itk.CudaImage[itk.F, 3]
        self._NB_STACK_VOL = 2
        self.geometry = geometry

        self.projection_stack_information = projection_stack_information
        self.volume_information = volume_information
        self.is_2d = is_2d

        if ray_step_size == 0.0:
            self.ray_step_size = self.volume_information["spacing"][1]
        else:
            self.ray_step_size = ray_step_size

    @property
    def domain_shape(self) -> tuple:
        """The shape of the input volume."""
        domain_shape = tuple(self.volume_information["size"])

        return domain_shape

    @property
    def range_shape(self) -> tuple:
        """The shape of the output projection."""
        range_shape = tuple(self.projection_stack_information["size"])

        return range_shape

    @property
    def T(self):
        """Implements and returns the adjoint of the transform operator."""
        parent = self

        class _Adjoint:
            def __init__(self):
                itk, rtk = import_itk_rtk()
                self._CUDA_IMAGE_TYPE = itk.CudaImage[itk.F, 3]
                self._NB_STACK_VOL = parent._NB_STACK_VOL

                self.is_2d = parent.is_2d
                self.geometry = parent.geometry

                self.projection_stack_information = parent.projection_stack_information
                self.volume_information = parent.volume_information
                self.ray_step_size = parent.ray_step_size

            @property
            def domain_shape(self) -> tuple:
                """The shape of the input projection."""
                return parent.range_shape

            @property
            def range_shape(self) -> tuple:
                """The shape of the output volume."""
                return parent.domain_shape

            def __call__(
                self, x: torch.Tensor, out: torch.Tensor | None = None
            ) -> torch.Tensor:
                r"""Backprojection.

                :param torch.Tensor x: Tensor of shape [A,1,N] in 2D, or [A,V,N] in 3D.
                :param torch.Tensor, None out: To avoid unnecessary copies, provide tensor of shape [H, W] in 2D, or [H,D,W] in 3D
                to store output results
                :return: Image of shape [1,H,W] in 2D or volume [H,D,W] in 3D.
                """

                if parent.is_2d:
                    x = x.unsqueeze(1)

                assert (
                    x.shape == self.domain_shape
                ), f"Input shape {x.shape} does not match expected shape {self.domain_shape}"

                if out is None:  # pragma: no cover
                    out = torch.zeros(
                        self.range_shape, dtype=torch.float32, device=x.device
                    )

                parent._backprojection(x, out)
                # if self.is_2d:
                #    # necessary scaling in fanbeam to obtain decent approximated adjoint
                #    out /= parent.magnification_factor

                return out

            @property
            def T(self):  # pragma: no cover
                return parent

        return _Adjoint()

    def __call__(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        r"""Forward projection.

        :param torch.Tensor x: Tensor of shape [1,H,W] in 2D, or [H,D,W] in 3D.
        :param torch.Tensor, None out: To avoid unnecessary copies, provide tensor of shape [...,A,N]
        to store output results
        :return: Sinogram of shape [1,A,N] in 2D or set of sinograms [V,A,N] in 3D.
        """

        if self.is_2d:
            x = self.stack_volume(x)

        assert (
            x.shape == self.domain_shape
        ), f"Input shape {x.shape} does not match expected shape {self.domain_shape}"  # # The error may be unclear in 2D if the user provides an incorrect shape

        if out is None:  # pragma: no cover
            out = torch.zeros(self.range_shape, dtype=torch.float32, device=x.device)

        self._forward_projection(x, out)

        return out

    def _forward_projection(self, x: torch.Tensor, out: torch.Tensor) -> None:

        itk, rtk = import_itk_rtk()

        # Cast from tensor to ITK cuda image
        imageSource_cuda = itk.cuda_image_from_cuda_array(x)
        imageSource_cuda.SetOrigin(self.volume_information["origin"])
        imageSource_cuda.SetSpacing(self.volume_information["spacing"])

        out.mul_(0)  # RTK needs the source to be null
        fp_source = itk.cuda_image_from_cuda_array(out)
        fp_source.SetOrigin(self.projection_stack_information["origin"])
        fp_source.SetSpacing(self.projection_stack_information["spacing"])

        # Forward projection: Ax
        forward_projector = rtk.CudaForwardProjectionImageFilter[
            self._CUDA_IMAGE_TYPE
        ].New()
        forward_projector.SetGeometry(self.geometry)
        forward_projector.SetInput(fp_source)
        forward_projector.SetInput(1, imageSource_cuda)
        forward_projector.SetStepSize(self.ray_step_size)
        forward_projector.InPlaceOn()
        forward_projector.Update()
        Ax = forward_projector.GetOutput()
        Ax.DisconnectPipeline()

    def _backprojection(self, y: torch.Tensor, out: torch.Tensor) -> None:
        itk, rtk = import_itk_rtk()

        # https://discuss.pytorch.org/t/about-error-more-than-one-element-of-the-written-to-tensor-refers-to-a-single-memory-location/85526
        # https://docs.pytorch.org/docs/2.12/notes/extending.html#how-to-use
        y = y.contiguous()

        projection_cuda = itk.cuda_image_from_cuda_array(y)
        projection_cuda.SetOrigin(self.projection_stack_information["origin"])
        projection_cuda.SetSpacing(self.projection_stack_information["spacing"])

        out.mul_(0)  # RTK needs the source to be null
        fp_source = itk.cuda_image_from_cuda_array(out)
        fp_source.SetOrigin(self.volume_information["origin"])
        fp_source.SetSpacing(self.volume_information["spacing"])

        # Backprojection: A^T x
        back_projector = rtk.CudaRayCastBackProjectionImageFilter.New()
        back_projector.SetGeometry(self.geometry)
        back_projector.SetInput(0, fp_source)
        back_projector.SetInput(1, projection_cuda)
        back_projector.SetStepSize(self.ray_step_size)
        back_projector.InPlaceOn()
        back_projector.Update()
        Atx = back_projector.GetOutput()
        Atx.DisconnectPipeline()

    def _fdk(
        self,
        y: torch.Tensor,
        out: torch.Tensor,
        parker_angular_gap_threshold: float = 0,
        truncation_correction_padding: float = 0,
        hann_cut_frequency: float = 0,
    ) -> None:
        itk, rtk = import_itk_rtk()

        if self.is_2d:
            y_stacked = self.stack_volume(y)
        else:
            y_stacked = y

        projection_cuda = itk.cuda_image_from_cuda_array(y_stacked)
        origin = self.projection_stack_information["origin"].copy()

        projection_cuda.SetOrigin(origin)
        projection_cuda.SetSpacing(self.projection_stack_information["spacing"])

        # Initialize the source
        fp_source = itk.cuda_image_from_cuda_array(out)
        origin = self.volume_information["origin"].copy()
        if self.is_2d:  # This has to do with the backprojector used by rtkfdk
            origin[1] = 0.0

        fp_source.SetOrigin(origin)
        fp_source.SetSpacing(self.volume_information["spacing"])

        # Define the parker filter for short scan artefact correction
        parker = rtk.CudaParkerShortScanImageFilter.New(Geometry=self.geometry)
        parker.SetInput(projection_cuda)
        parker.SetAngularGapThreshold(parker_angular_gap_threshold)

        # FDK reconstruction
        feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
        feldkamp.SetInput(0, fp_source)
        feldkamp.SetInput(1, parker.GetOutput())
        feldkamp.SetGeometry(self.geometry)
        feldkamp.GetRampFilter().SetTruncationCorrection(truncation_correction_padding)
        feldkamp.GetRampFilter().SetHannCutFrequency(hann_cut_frequency)
        feldkamp.InPlaceOn()
        feldkamp.Update()

        itk_reco = feldkamp.GetOutput()
        itk_reco.DisconnectPipeline()

    def fdk_on_batch(
        self,
        input: torch.Tensor,
        out: torch.Tensor,
        parker_angular_gap_threshold: float = 0.0,
        truncation_correction_padding: float = 0.0,
        hann_cut_frequency: float = 0.0,
    ):

        if self.is_2d:
            B, C, A, N = input.shape
            assert (A, 1, N) == self.range_shape, f"{(A,1,N)} != {self.range_shape}"

            for i in range(B):
                for j in range(C):
                    self._fdk(
                        input[i, j],
                        out[i, j],
                        parker_angular_gap_threshold,
                        truncation_correction_padding,
                        hann_cut_frequency,
                    )

        else:
            B, C, A, V, N = input.shape
            assert (A, V, N) == self.range_shape, f"{(A,V,N)} != {self.range_shape}"

            for i in range(B):
                for j in range(C):
                    self._fdk(
                        input[i, j],
                        out[i, j],
                        parker_angular_gap_threshold,
                        truncation_correction_padding,
                        hann_cut_frequency,
                    )

    def stack_volume(self, x, dim=1):

        return torch.stack([x] * self._NB_STACK_VOL, dim=dim)


def build_rtk_trajectory(
    rtk,
    angles: int | torch.Tensor,
    angular_range: tuple[float, float],
    geometry_parameters: dict[str, float],
):
    """Build an ``rtk.ThreeDCircularProjectionGeometry`` trajectory from
     ``angles`` / ``angular_range`` / ``geometry_parameters``.
    Returns ``(geometry, n_angles)``.
    """
    if isinstance(angles, int):
        angles = torch.linspace(*angular_range, steps=angles + 1)[:-1]
    angles_deg = [float(a) for a in angles]
    n_angles = len(angles_deg)

    source_radius = geometry_parameters.get("source_radius", 80.0)
    detector_radius = geometry_parameters.get("detector_radius", 20.0)
    sid = source_radius
    sdd = source_radius + detector_radius

    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for angle in angles_deg:
        geometry.AddProjection(sid, sdd, angle)

    return geometry, n_angles


def build_rtk_info(
    geometry_type: str,
    img_size: tuple[int, ...],
    n_angles: int,
    n_detector_pixels: int | tuple[int, int] | None,
    pixel_spacing: float | tuple[float, ...],
    detector_spacing: float | tuple[float, float],
    detector_origin: float | tuple[float, float] | None,
    bounding_box: tuple[float, ...] | None,
    volume_origin: float | tuple[float, float] | None,
    geometry: any,
    nb_stack_vol: int,
):
    """Build the ``projection_stack_information`` / ``volume_information``
    dictionaries (already in the final 3D format expected by
    :class:`XrayTransformRTK`).

    For ``geometry_type='fanbeam'``, the 2D detector/volume
    description is embedded into 3D.
    """
    is_2d = geometry_type == "fanbeam"
    expected_ndim = 2 if is_2d else 3

    if len(img_size) != expected_ndim:
        raise ValueError(
            f"img_size must have {expected_ndim} elements for "
            f"geometry_type={geometry_type!r}, got {img_size!r}"
        )

    # --- volume grid ---
    if is_2d:
        n_rows, n_cols = img_size
    else:
        n_rows, n_slices, n_cols = img_size

    if isinstance(pixel_spacing, (int, float)):
        pixel_spacing = (float(pixel_spacing),) * expected_ndim

    if bounding_box is not None:
        if len(bounding_box) != 2 * expected_ndim:
            raise ValueError(
                f"bounding_box must have {2 * expected_ndim} elements for "
                f"geometry_type={geometry_type!r}, got {bounding_box!r}"
            )
        if is_2d:
            d0_min, d0_max, d1_min, d1_max = bounding_box
            s0 = (d0_max - d0_min) / n_cols
            s1 = (d1_max - d1_min) / n_rows
            vol_spacing = [s0, s1]
            vol_origin = [d0_min + s0 / 2, d1_min + s1 / 2]
        else:
            d0_min, d0_max, d1_min, d1_max, d2_min, d2_max = bounding_box
            s0 = (d0_max - d0_min) / n_cols
            s1 = (d1_max - d1_min) / n_rows
            s2 = (d2_max - d2_min) / n_slices
            vol_spacing = [s0, s1, s2]
            vol_origin = [d0_min + s0 / 2, d1_min + s1 / 2, d2_min + s2 / 2]
    else:
        if is_2d:
            s0, s1 = pixel_spacing
            vol_spacing = [s0, s1]
            vol_origin = [-0.5 * (n_cols - 1) * s0, -0.5 * (n_rows - 1) * s1]
        else:
            s0, s1, s2 = pixel_spacing
            vol_spacing = [s0, s1, s2]
            vol_origin = [
                -0.5 * (n_cols - 1) * s0,
                -0.5 * (n_rows - 1) * s1,
                -0.5 * (n_slices - 1) * s2,
            ]

    # takes precedence
    if volume_origin is not None:
        if isinstance(volume_origin, (int, float)):
            vol_origin = [float(volume_origin)] * expected_ndim
        else:
            if len(volume_origin) != expected_ndim:
                raise ValueError(
                    f"volume_origin must have {expected_ndim} elements for "
                    f"geometry_type={geometry_type!r}, got {volume_origin!r}"
                )
            vol_origin = [float(v) for v in volume_origin]

    volume_information = {
        "size": (
            [n_cols, nb_stack_vol, n_rows] if is_2d else [n_cols, n_slices, n_rows]
        )[::-1],
        "spacing": (
            [vol_spacing[0], 1.0, vol_spacing[1]]
            if is_2d
            else [
                vol_spacing[0],
                vol_spacing[1],
                vol_spacing[2],
            ]  # the 1.0 for the spacing might need to be changed
        )[::-1],
        "origin": (
            [vol_origin[0], -0.5 * (nb_stack_vol - 1), vol_origin[1]]
            if is_2d
            else [vol_origin[0], vol_origin[1], vol_origin[2]]
        )[::-1],
    }

    # --- detector / projections ---
    if is_2d:
        n_det = (
            n_detector_pixels
            if n_detector_pixels is not None
            else math.ceil(math.sqrt(2) * n_cols)
        )
        det_spacing = (
            float(detector_spacing)
            if isinstance(detector_spacing, (int, float))
            else float(detector_spacing[0])
        )
        proj_size = [n_det, 1, n_angles]
        proj_spacing = [det_spacing, 1.0, 1.0]
        if detector_origin is not None:
            origin_0 = (
                float(detector_origin)
                if isinstance(detector_origin, (int, float))
                else float(detector_origin[0])
            )
        else:
            origin_0 = -0.5 * (n_det - 1) * det_spacing
        proj_origin = [origin_0, -geometry.GetSourceOffsetsY()[0], 0.0]
    else:
        if n_detector_pixels is None:
            n_det_rows = n_det_cols = math.ceil(math.sqrt(2) * n_cols)
        elif isinstance(n_detector_pixels, int):
            n_det_rows = n_det_cols = n_detector_pixels
        else:
            n_det_rows, n_det_cols = n_detector_pixels

        if isinstance(detector_spacing, (int, float)):
            det_spacing_v = det_spacing_h = float(detector_spacing)
        else:
            det_spacing_v, det_spacing_h = detector_spacing

        proj_size = [n_det_cols, n_det_rows, n_angles]
        proj_spacing = [det_spacing_h, det_spacing_v, 1.0]
        if detector_origin is not None:
            if isinstance(detector_origin, (int, float)):
                origin_v = origin_h = float(detector_origin)
            else:
                origin_v, origin_h = detector_origin
            proj_origin = [origin_h, origin_v, 0.0]
        else:
            proj_origin = [
                -0.5 * (n_det_cols - 1) * det_spacing_h,
                -0.5 * (n_det_rows - 1) * det_spacing_v,
                0.0,
            ]

    projection_stack_information = {
        "size": proj_size[::-1],
        "spacing": proj_spacing,
        "origin": proj_origin,
    }

    return projection_stack_information, volume_information


class AutogradTransformRTK(torch.autograd.Function):
    r"""Custom torch.autograd.Function for XrayTransformRTK
    this class is based of the AutogradTransform from deepinv.physics.functional.astra

    See https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
    for more information.
    """

    @staticmethod
    def forward(input: torch.Tensor, op: XrayTransformRTK) -> torch.Tensor:
        """Forward autograd.

        ``RTK`` does not handle batched computation, the transform
        is applied sequentially by iterating over the batch and channel dimension
        (e.g. Learned Primal-Dual).

        For 2D operations, RTK requires a volume of at least two slices for 2D projections in order to compute interpolation.

        :param torch.Tensor input: Batched with channel input Tensor of shape [B,C,...].
        :param XrayTransformRTK op: XrayTransformRTK operator which performs underlying computations.
        :return:
        """
        if op.is_2d:
            B, C, H, W = input.shape

            assert (H, 1, W) == op.domain_shape or (
                H,
                op._NB_STACK_VOL,
                W,
            ) == op.domain_shape, f"{(H, 'X', W)} != {op.domain_shape}"

            output = torch.empty(
                (B, C) + op.range_shape, dtype=input.dtype, device=input.device
            )

            for i in range(B):
                for j in range(C):
                    op(input[i, j], out=output[i, j])

            output = output[
                :, :, :, 0
            ]  # select only one of the slices from op._NB_STACK_VOL

        else:
            B, C, H, D, W = input.shape
            assert (H, D, W) == op.domain_shape, f"{(H,D,W)} != {op.domain_shape}"

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

        return AutogradTransformRTK.apply(grad_output, op.T), None
