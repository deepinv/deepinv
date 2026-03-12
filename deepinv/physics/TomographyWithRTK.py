import torch
from deepinv.physics.forward import LinearPhysics
from types import ModuleType


def import_itk_rtk() -> (ModuleType, ModuleType):
    try:
        import itk
        from itk import RTK as rtk

        return (itk, rtk)
    except ImportError:  # pragma: no cover
        raise ImportError(
            "itk-rtk is required to use TomographyWithRTK. It can be installed with itk-rtk-cuda128."
        )


class TomographyWithRTK(LinearPhysics):
    r"""Computed Tomography operator with RTK CUDA backend.

    This operator implements a ray transform :math:`A` using the RTK (Reconstruction Toolkit) GPU-accelerated forward and backprojectors.

    Mathematically, it computes line integrals of an object :math:`x` along X-ray paths:

    .. math::
        y = Ax

    where :math:`y` represents projection data.

    Supported geometries:

    * ``fanbeam`` (2D-like using the 3D geometry)
    * ``conebeam`` (3D trajectory)

    The adjoint is implemented using RTK's ray-cast adjoint.
    The pseudo-inverse is approximated using filtered backprojection reconstruction.

    This implementation requires ITK with CudaCommon v2.1 and RTK v3.
    """

    def __init__(
        self,
        geometry: any,
        projection_stack_information: dict[str, list],
        volume_information: dict[str, list],
        mode: str,
        verbose: bool = False,
        normalize: bool = False,
        ray_step_size: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        :param geometry: RTK geometry object defining source/detector trajectory.
        :param projection_stack_information: Dictionary describing the projection stack information
            (size, spacing, origin).
        :param volume_information: Dictionary describing reconstruction
            volume grid (size, spacing, origin).
        :param str mode: Either ``"fanbeam"`` or ``"conebeam"``.
        :param bool verbose: If True, print geometry configuration.
        :param bool normalize: If True, normalize operator to unit norm.
        :param float ray_step_size: Step size along the ray.
        """

        itk, rtk = import_itk_rtk()

        super().__init__(*args, **kwargs)

        # A and A_adjoint use a ray-based approach with trilinear interpolation
        # of the volume so they need at least two volume slices. One projection
        # line is sufficient.
        self._NB_STACK_VOL = 2
        self._CUDA_IMAGE_TYPE = itk.CudaImage[itk.F, 3]

        self.geometry = geometry
        self.mode = mode

        self.projection_stack_information = projection_stack_information
        self.volume_information = volume_information

        if mode not in ("fanbeam", "conebeam"):
            raise ValueError(
                f"mode {mode!r} unrecognized (expected 'fanbeam' or 'conebeam')"
            )

        self._validate_info(
            mode, self.projection_stack_information, "projection_stack_information"
        )
        self._validate_info(mode, self.volume_information, "volume_information")

        # ---------------------------------------------------------
        # FANBEAM CONFIGURATION (2D embedded in 3D volume)
        # ---------------------------------------------------------
        if mode == "fanbeam":
            # Information for A and A_adjoint. Information for fbp is computed in the function
            self.projection_stack_information["spacing"] = [
                self.projection_stack_information["spacing"][0],
                1.0,
                self.projection_stack_information["spacing"][1],
            ]
            self.projection_stack_information["size"] = [
                self.projection_stack_information["size"][0],
                1,
                self.projection_stack_information["size"][1],
            ]
            self.projection_stack_information["origin"] = [
                self.projection_stack_information["origin"][0],
                -geometry.GetSourceOffsetsY()[0],
                self.projection_stack_information["origin"][1],
            ]

            self.volume_information["spacing"] = [
                self.volume_information["spacing"][0],
                1.0,
                self.volume_information["spacing"][1],
            ]
            self.volume_information["size"] = [
                self.volume_information["size"][0],
                self._NB_STACK_VOL,
                self.volume_information["size"][1],
            ]

            self.volume_information["origin"] = [
                self.volume_information["origin"][0],
                -0.5 * (self._NB_STACK_VOL - 1),
                self.volume_information["origin"][1],
            ]

        if ray_step_size is 0.0:
            self.ray_step_size = self.volume_information["spacing"][1]
        else:
            self.ray_step_size = ray_step_size

        self.normalize = False
        if normalize:
            vol_size = self.volume_information["size"]
            if mode == "fanbeam":
                logical_size = [vol_size[0], vol_size[2]]
            else:
                logical_size = vol_size
            self.norm_mat = self.compute_norm(
                torch.randn((1, 1, *logical_size), device="cuda")
            ).sqrt()
            self.normalize = True
        else:
            self.norm_mat = None

        if verbose:
            print("Projection stack information: ")
            print(self.projection_stack_information["size"])
            print(self.projection_stack_information["origin"])
            print(self.projection_stack_information["spacing"])

            print("Output volume information: ")
            print(self.volume_information["size"])
            print(self.volume_information["origin"])
            print(self.volume_information["spacing"])

    def _validate_info(
        self, mode: str, info: dict[str, int | float], name: str
    ) -> None:
        """Validate that all array-like values in ``info`` have the expected length for the given mode.

        :param mode: Either ``"fanbeam"`` or ``"conebeam"``.
        :param info: Dictionary describing a grid (size, spacing, origin) in the image space or in the observation space.
        :param name: Name for error throwing.
        """
        expected_len = 2 if mode == "fanbeam" else 3

        for key, element in info.items():
            # If the element is not array-like, skip
            if not hasattr(element, "__len__"):
                continue

            actual_len = len(element)
            if actual_len != expected_len:
                raise ValueError(
                    f"Expected element length {expected_len} for mode {mode!r}, "
                    f"got {actual_len} for key {key!r} in {name}"
                )

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward projection.

        :param torch.Tensor x: input of shape [B,C,...,H,W]
        :return: projection of shape [B,C,...,A,N]
        """

        itk, rtk = import_itk_rtk()

        x_stacked = x.squeeze(0).squeeze(0)

        # Add a dimension to simulate a 2D projection in fanbeam mode
        if self.mode == "fanbeam":
            x_stacked = torch.stack([x_stacked.clone()] * self._NB_STACK_VOL, dim=1).to(
                "cuda:0"
            )

        # Cast from tensor to ITK cuda image
        imageSource_cuda = itk.cuda_image_from_cuda_array(x_stacked)
        imageSource_cuda.SetOrigin(self.volume_information["origin"])
        imageSource_cuda.SetSpacing(self.volume_information["spacing"])

        fp_source = rtk.ConstantImageSource[self._CUDA_IMAGE_TYPE].New()
        fp_source.SetSize(self.projection_stack_information["size"])
        fp_source.SetOrigin(self.projection_stack_information["origin"])
        fp_source.SetSpacing(self.projection_stack_information["spacing"])

        # Forward projection: Ax
        forward_projector = rtk.CudaForwardProjectionImageFilter[
            self._CUDA_IMAGE_TYPE
        ].New()
        forward_projector.SetGeometry(self.geometry)
        forward_projector.SetInput(fp_source.GetOutput())
        forward_projector.SetInput(1, imageSource_cuda)
        forward_projector.SetStepSize(self.ray_step_size)
        forward_projector.Update()
        Ax = forward_projector.GetOutput()
        Ax.DisconnectPipeline()

        # Cast back from itk cuda image to tensor
        projections = torch.as_tensor(
            Ax, device=x.device
        ).clone()  # This may not be optimized for memory usage

        if self.mode == "fanbeam":
            projections = projections[:, 0, :]

        if self.normalize:
            projections /= self.norm_mat

        return projections.unsqueeze(0).unsqueeze(0)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Approximation of the adjoint.

        :param torch.Tensor y: input of shape [B,C,...,A,N]
        :return: scaled back-projection of shape [B,C,...,H,W]
        """

        itk, rtk = import_itk_rtk()

        y_stacked = y.squeeze(0).squeeze(0)

        # Add a dimension to simulate a 2D projection in fanbeam mode
        if self.mode == "fanbeam":
            y_stacked = torch.stack([y_stacked.clone()] * 1, dim=1).to("cuda:0")

        # Cast from tensor to ITK cuda image
        projection_cuda = itk.cuda_image_from_cuda_array(y_stacked)
        projection_cuda.SetOrigin(self.projection_stack_information["origin"])
        projection_cuda.SetSpacing(self.projection_stack_information["spacing"])

        fp_source = rtk.ConstantImageSource[self._CUDA_IMAGE_TYPE].New()

        fp_source.SetSize(self.volume_information["size"])
        fp_source.SetOrigin(self.volume_information["origin"])
        fp_source.SetSpacing(self.volume_information["spacing"])

        # Backprojection: A^T x
        back_projector = rtk.CudaRayCastBackProjectionImageFilter.New()
        back_projector.SetGeometry(self.geometry)
        back_projector.SetInput(0, fp_source.GetOutput())
        back_projector.SetInput(1, projection_cuda)
        back_projector.SetStepSize(self.ray_step_size)
        back_projector.Update()
        Atx = back_projector.GetOutput()
        Atx.DisconnectPipeline()

        # Cast back from itk cuda image to tensor
        backproj = torch.as_tensor(
            Atx, device=y.device
        ).clone()  # This may not be optimized for memory usage

        if self.mode == "fanbeam":
            backproj = backproj.sum(dim=1)
        else:
            backproj = backproj.permute(2, 1, 0)

        if self.normalize:
            backproj /= self.norm_mat

        return backproj.unsqueeze(0).unsqueeze(0)

    def fbp(
        self,
        y: torch.Tensor,
        parker_angular_gap_threshold: float = 0,
        truncation_correction_padding: float = 0,
        hann_cut_frequency: float = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Reconstruct an image from projection data using the FDK algorithm.

        :param torch.Tensor y: input of shape [B,C,...,A,N]
        :param parker_angular_gap_threshold: Angular gap threshold (in degrees) at which [Parker, Med Phys, 1982] weighting is used.
        :param truncation_correction_padding: Padding ratio applied to reduce truncation artefacts via [Ohnesorge, Eur Radiol, 1999].
        :param float hann_cut_frequency: Cut frequency for Hann windowing in ]0,1] (0.0 disables it, 1. is Nyquist frequency).
        :return: reconstruction using the FDK algorithm of shape [B,C,...,H,W]
        """

        itk, rtk = import_itk_rtk()

        y_stacked = y.squeeze(0).squeeze(0)

        # fbp uses a voxel based approach, i.e., a bilinear interpolation of
        # the projections so it needs at least two slices which is handled in
        # the fbp function. One volume slice is sufficient.
        nb_stack_proj = 2
        if self.mode == "fanbeam":
            y_stacked = torch.stack(
                [y_stacked.clone()] * nb_stack_proj, dim=1
            )  # stack n slices of x

        # Cast from tensor to ITK cuda image
        projection_cuda = itk.cuda_image_from_cuda_array(y_stacked)
        origin = self.projection_stack_information["origin"].copy()
        if self.mode == "fanbeam":
            origin[1] = -0.5 * (nb_stack_proj - 1)
        projection_cuda.SetOrigin(origin)
        projection_cuda.SetSpacing(self.projection_stack_information["spacing"])

        # Initialize the source
        fp_source = rtk.ConstantImageSource[self._CUDA_IMAGE_TYPE].New()
        fp_source.SetSize(self.volume_information["size"])
        origin = self.volume_information["origin"].copy()
        if self.mode == "fanbeam":
            origin[1] = 0.0
        fp_source.SetOrigin(origin)
        fp_source.SetSpacing(self.volume_information["spacing"])

        # Define the parker filter for short scan artefact correction
        parker = rtk.CudaParkerShortScanImageFilter.New(Geometry=self.geometry)
        parker.SetInput(projection_cuda)
        parker.SetAngularGapThreshold(parker_angular_gap_threshold)

        # FDK reconstruction
        feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
        feldkamp.SetInput(0, fp_source.GetOutput())
        feldkamp.SetInput(1, parker.GetOutput())
        feldkamp.SetGeometry(self.geometry)
        feldkamp.GetRampFilter().SetTruncationCorrection(truncation_correction_padding)
        feldkamp.GetRampFilter().SetHannCutFrequency(hann_cut_frequency)
        feldkamp.Update()

        itk_reco = feldkamp.GetOutput()
        itk_reco.DisconnectPipeline()

        # Cast back from itk cuda image to tensor
        reco = torch.as_tensor(
            itk_reco, device=y.device
        ).clone()  # This may not be optimized for memory usage

        if self.mode == "fanbeam":
            reco = reco.sum(dim=1)
        else:
            reco = reco.permute(2, 1, 0)

        return reco.unsqueeze(0).unsqueeze(0)
