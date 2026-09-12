from __future__ import annotations
from warnings import warn
import numpy as np
import torch
from torch import Tensor

from deepinv.physics.forward import DecomposablePhysics, LinearPhysics
from deepinv.physics.mri_motion import TimeVaryingMotion
from deepinv.utils.mixins import MRIMixin, TimeMixin


class MRI(MRIMixin, DecomposablePhysics):
    r"""
    Single-coil accelerated 2D or 3D magnetic resonance imaging.

    The linear operator operates in 2D slices or 3D volumes and is defined as

    .. math::

        y = MFx

    where :math:`M` applies a mask (subsampling operator), and :math:`F` is the 2D or 3D discrete Fourier Transform.
    This operator has a simple singular value decomposition, so it inherits the structure of
    :class:`deepinv.physics.DecomposablePhysics` and thus have a fast pseudo-inverse and prox operators.

    The complex images :math:`x` and measurements :math:`y` should be of size (B, C,..., H, W) with C=2, where the first channel corresponds to the real part
    and the second channel corresponds to the imaginary part. The ``...`` is an optional depth dimension for 3D MRI data.

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update``.

    .. note::

        We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. :class:`deepinv.physics.generator.mri.RandomMaskGenerator`
        If mask is not passed, a mask full of ones is used (i.e. no acceleration).

    .. note::

        This physics is directly compatible with FastMRI data using :class:`deepinv.datasets.FastMRISliceDataset`.
        The dataset loads pairs of magnitude images and kspace ``(x, y)`` where ``x = MRI().A_adjoint(y, mag=True, crop=True)``.

    :param torch.Tensor mask: binary mask, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (C,H,W), (B,C,H,W), (B,C,...,H,W) where H, W are the image height and width, C is channels (which should be 2) and B is batch size.
    :param tuple img_size: if mask not specified, flat mask of ones is created using ``img_size``, where ``img_size`` can be of any shape specified above. If mask provided, ``img_size`` is ignored.
    :param bool three_d: if ``True``, calculate Fourier transform in 3D for 3D data (i.e. data of shape (B,C,D,H,W) where D is depth).
    :param torch.device device: cpu or gpu.

    |sep|

    :Examples:

        Single-coil accelerated MRI operator with subsampling mask:

        >>> from deepinv.physics import MRI
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 2, 2, 2) # Define random 2x2 image
        >>> mask = 1 - torch.eye(2) # Define subsampling mask
        >>> physics = MRI(mask=mask) # Define mask at initialisation
        >>> physics(x)
        tensor([[[[ 0.0000, -1.4290],
                  [ 0.4564, -0.0000]],
        <BLANKLINE>
                 [[ 0.0000,  1.8622],
                  [ 0.0603, -0.0000]]]])
        >>> physics = MRI(img_size=x.shape) # No subsampling
        >>> physics(x)
        tensor([[[[ 2.2908, -1.4290],
                  [ 0.4564, -0.1814]],
        <BLANKLINE>
                 [[ 0.3744,  1.8622],
                  [ 0.0603, -0.6209]]]])
        >>> physics.update(mask=mask) # Update mask on the fly
        >>> physics(x)
        tensor([[[[ 0.0000, -1.4290],
                  [ 0.4564, -0.0000]],
        <BLANKLINE>
                 [[ 0.0000,  1.8622],
                  [ 0.0603, -0.0000]]]])

    """

    def __init__(
        self,
        mask: Tensor | None = None,
        img_size: tuple | None = (320, 320),
        three_d: bool = False,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.three_d = three_d
        self.img_size = img_size

        if mask is None:
            mask = torch.ones(*img_size, device=device)

        # Check and update mask
        self.register_buffer("mask", self.check_mask(mask))
        self.img_size = self.mask.shape[1:]
        self.to(device)

    def V_adjoint(self, x: Tensor) -> Tensor:
        return self.im_to_kspace(x, three_d=self.three_d)

    def V(self, x: Tensor) -> Tensor:
        return self.kspace_to_im(x, three_d=self.three_d)

    def A_adjoint(
        self,
        y: Tensor,
        mask: Tensor = None,
        mag: bool = False,
        crop: bool = False,
        **kwargs,
    ) -> Tensor:
        """Adjoint operator.

        Optionally perform crop and magnitude to match FastMRI data.

        By default, crop and magnitude are not performed.
        By setting ``mag=crop=True``, the outputs will be consistent with :class:`deepinv.datasets.FastMRISliceDataset`.

        :param torch.Tensor y: input kspace of shape (B,C,...,H,W)
        :param torch.Tensor mask: optionally set mask on-the-fly.
        :param bool mag: perform complex magnitude.
            This option is provided to match the original data of :class:`deepinv.datasets.FastMRISliceDataset`,
            such that ``x = MRI().A_adjoint(y, mag=True)``.
        :param bool crop: if ``True``, crop last 2 dims of x to last 2 dims of img_size.
            This option is provided to match the original data of :class:`deepinv.datasets.FastMRISliceDataset`,
            such that ``x = MRI().A_adjoint(y, crop=True)``.
        """
        x = super().A_adjoint(y, mask, **kwargs)
        if mag:
            x = self.rss(x, multicoil=False)
        if crop:
            x = self.crop(x, crop=crop)
        return x  # (B,C,...,H,W) where C=1 if mag else 2

    def noise(self, x, **kwargs):
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :return torch.Tensor: noisy measurements
        """
        noise = self.U(self.noise_model(x, **kwargs) * self.mask)
        return noise

    def update_parameters(self, mask: Tensor = None, check_mask: bool = True, **kwargs):
        """Update MRI subsampling mask.

        :param torch.nn.parameter.Parameter, torch.Tensor mask: MRI mask
        :param bool check_mask: check mask dimensions before updating
        """
        if mask is not None:
            mask = (
                self.check_mask(
                    mask=mask,
                    three_d=getattr(self, "three_d", False),
                )
                if check_mask
                else mask
            )

        super().update_parameters(mask=mask, **kwargs)


class MultiCoilMRI(MRIMixin, LinearPhysics):
    r"""
    Multi-coil 2D or 3D MRI operator.

    The linear operator operates in 2D slices or 3D volumes and is defined as:

    .. math::

        y_n = \text{diag}(p) F \text{diag}(s_n) x

    for :math:`n=1,\dots,N` coils, where :math:`y_n` are the measurements from the cth coil, :math:`\text{diag}(p)` is the acceleration mask, :math:`F` is the Fourier transform and :math:`\text{diag}(s_n)` is the nth coil sensitivity.

    The data ``x`` should be of shape (B,C,H,W) or (B,C,D,H,W) where C=2 is the channels (real and imaginary) and D is optional dimension for 3D MRI.
    Then, the resulting measurements ``y`` will be of shape (B,C,N,(D,)H,W) where N is the coils dimension.

    .. note::

        We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. :class:`deepinv.physics.generator.mri.RandomMaskGenerator`.
        If mask or coil maps are not passed, a mask and maps full of ones is used (i.e. no acceleration).

    .. note::

        You can also simulate basic `birdcage coil sensitivity maps <https://mriquestions.com/birdcage-coil.html>` by passing instead an integer to ``coil_maps``
        using ``MultiCoilMRI(coil_maps=N, img_size=x.shape)`` (note this requires installing the ``sigpy`` library).

    .. note::

        This physics is directly compatible with FastMRI data using :class:`deepinv.datasets.FastMRISliceDataset`.
        The dataset loads pairs of RSS images and multicoil kspace ``(x, y)`` where ``x = MultiCoilMRI().A_adjoint(y, rss=True, crop=True)``.

    :param torch.Tensor mask: binary sampling mask which should have shape (H,W), (C,H,W), (B,C,H,W), or (B,C,...,H,W). If None, generate mask of ones with ``img_size``.
    :param torch.Tensor, str coil_maps: either ``Tensor``, integer, or ``None``. If complex valued (i.e. of complex dtype) coil sensitivity maps which should have shape (H,W), (N,H,W), (B,N,H,W) or (B,N,...,H,W).
        If None, generate flat coil maps of ones with ``img_size``. If integer, simulate birdcage coil maps with integer number of coils (this requires ``sigpy`` installed).
    :param tuple img_size: if ``mask`` or ``coil_maps`` not specified, flat ``mask`` or ``coil_maps`` of ones are created using ``img_size``,
        where ``img_size`` can be of any shape specified above. If ``mask`` or ``coil_maps`` provided, ``img_size`` is ignored.
    :param bool three_d: if ``True``, calculate Fourier transform in 3D for 3D data (i.e. data of shape (B,C,D,H,W) where D is depth).
    :param torch.device, str device: specify which device you want to use (i.e, cpu or gpu).

    |sep|

    :Examples:

        Multi-coil MRI operator:

        >>> from deepinv.physics import MultiCoilMRI
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 2, 2, 2) # Define random 2x2 image B,C,H,W
        >>> physics = MultiCoilMRI(img_size=x.shape) # Define coil map of ones
        >>> physics(x).shape # B,C,N,H,W
        torch.Size([1, 2, 1, 2, 2])
        >>> coil_maps = torch.randn(1, 5, 2, 2, dtype=torch.complex64) # Define 5-coil sensitivity maps
        >>> physics.update(coil_maps=coil_maps) # Update coil maps on the fly
        >>> physics(x).shape
        torch.Size([1, 2, 5, 2, 2])

    """

    def __init__(
        self,
        mask: Tensor | None = None,
        coil_maps: Tensor | int | None = None,
        img_size: tuple | None = (320, 320),
        three_d: bool = False,
        device: torch.device | str = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.img_size = img_size
        self.three_d = three_d

        if mask is None:
            mask = torch.ones(*img_size, device=device)

        if coil_maps is None:
            coil_maps = torch.ones(
                (self.img_size[-2:] if not self.three_d else self.img_size[-3:]),
                dtype=torch.complex64,
                device=device,
            )
        elif isinstance(coil_maps, int):
            coil_maps = self.simulate_birdcage_csm(n_coils=coil_maps).to(device)

        self.register_buffer("mask", self.check_mask(mask, three_d=self.three_d))
        self.register_buffer(
            "coil_maps", self.check_coil_maps(coil_maps, three_d=self.three_d)
        )
        self.to(device)

    def A(
        self, x: Tensor, mask: Tensor = None, coil_maps: Tensor = None, **kwargs
    ) -> Tensor:
        r"""
        Applies linear operator.

        Optionally update MRI mask or coil sensitivity maps on the fly.

        :param torch.Tensor x: image with shape `(B,2,...,H,W)`.
        :param torch.Tensor mask: optionally set the mask on-the-fly.
        :param torch.Tensor coil_maps: optionally set the mask on-the-fly.
        :returns: (:class:`torch.Tensor`) multi-coil kspace measurements with shape `(B,2,N,...,H,W)` where `N` is coil dimension.
        """
        self.update_parameters(mask=mask, coil_maps=coil_maps, **kwargs)

        Sx = self.coil_maps * self.to_torch_complex(x)[:, None]  # [B,N,...,H,W]
        FSx = self.fft(Sx, dim=(-3, -2, -1) if self.three_d else (-2, -1))
        MFSx = self.mask[:, :, None] * self.from_torch_complex(FSx)  # [B,2,N,...,H,W]
        return MFSx

    def noise(self, x, **kwargs) -> Tensor:
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)` and takes the mask into account.

        :param torch.Tensor x:  clean measurements
        :param None, float noise_level: optional noise level parameter
        :return: noisy measurements
        """
        return self.mask[:, :, None] * self.noise_model(x, **kwargs)

    def A_adjoint(
        self,
        y: Tensor,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        rss: bool = False,
        crop: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""
        Applies adjoint linear operator.

        Optionally update MRI mask or coil sensitivity maps on the fly.

        :param torch.Tensor y: multi-coil kspace measurements with shape [B,2,N,...,H,W] where N is coil dimension.
        :param torch.Tensor mask: optionally set the mask on-the-fly.
        :param torch.Tensor coil_maps: optionally set the mask on-the-fly.
        :param bool rss: perform root-sum-square reconstruction.
            This option is provided to match the original data of :class:`deepinv.datasets.FastMRISliceDataset`,
            such that ``x = MultiCoilMRI().A_adjoint(y, rss=True)``.
        :param bool crop: if ``True``, crop last 2 dims of x to last 2 dims of img_size.
            This option is provided to match the original data of :class:`deepinv.datasets.FastMRISliceDataset`,
            such that ``x = MultiCoilMRI().A_adjoint(y, crop=True)``.
        :returns: (:class:`torch.Tensor`) image with shape `(B,2,...,H,W)` if not rss else `(B,1,...,H,W)`
        """
        if y.shape[1] != 2:  # pragma: no cover
            raise ValueError("y must be of shape (B,2,N,...,H,W)")
        self.update_parameters(mask=mask, coil_maps=coil_maps, **kwargs)

        My = self.to_torch_complex(self.mask[:, :, None] * y)  # [B,N,...,H,W]
        FiMy = self.ifft(My, dim=(-3, -2, -1) if self.three_d else (-2, -1))

        if rss:
            x = self.from_torch_complex(FiMy)
            x = self.rss(x, multicoil=True)  # [B,1,...,H,W]
        else:
            # Use conj as coil maps are elementwise multiplication
            SiFiMy = torch.sum(torch.conj(self.coil_maps) * FiMy, dim=1)  # [B,...,H,W]
            x = self.from_torch_complex(SiFiMy)  # [B,2,...,H,W]

        return self.crop(x, crop=crop)

    def A_dagger(
        self, y: Tensor, mask: Tensor = None, coil_maps: Tensor = None, **kwargs
    ) -> Tensor:
        r"""
        Computes least squares solution to the MRI inverse problem, as proposed in `SENSE: Sensitivity encoding for fast MRI <https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S>`_.

        By default uses conjugate gradient solver. Overwrite default solver arguments by passing `kwargs`. See :func:`deepinv.optim.linear.least_squares` for details.

        The MRI mask or coil sensitivity maps are updated if passed as inputs to the function.

        :param torch.Tensor y: multi-coil kspace measurements with shape [B,2,N,...,H,W] where N is coil dimension.
        :param torch.Tensor mask: optionally set the mask on-the-fly.
        :param torch.Tensor coil_maps: optionally set the mask on-the-fly.
        :param dict kwargs: kwargs to pass to base :meth:`deepinv.physics.LinearPhysics.A_dagger`.
        :returns: (:class:`torch.Tensor`) image with shape `(B,2,...,H,W)`
        """
        self.update_parameters(mask=mask, coil_maps=coil_maps)
        return super().A_dagger(y, **kwargs)

    def update_parameters(
        self,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        check_mask: bool = True,
        check_coil_maps: bool = True,
        **kwargs,
    ):
        """Update MRI subsampling mask and coil sensitivity maps.

        :param torch.nn.parameter.Parameter, torch.Tensor mask: MRI mask
        :param torch.nn.parameter.Parameter, torch.Tensor coil_maps: MRI coil sensitivity maps
        :param bool check_mask: check mask dimensions before updating
        :param bool check_coil_maps: check coil maps dimensions before updating
        """
        if mask is not None:
            mask = (
                self.check_mask(mask=mask, three_d=self.three_d) if check_mask else mask
            )

        if coil_maps is not None:
            coil_maps = (
                self.check_coil_maps(coil_maps, three_d=self.three_d)
                if check_coil_maps
                else coil_maps
            )

        super().update_parameters(mask=mask, coil_maps=coil_maps, **kwargs)

        # Update image size with latest mask shape
        self.img_size = self.mask.shape[1:]

        if self.coil_maps is not None and self.coil_maps.shape[2:] != self.img_size[1:]:
            warn(
                f"After updating parameters, img_size {self.img_size} in MultiCoilMRI is incompatible with coil_maps shape {self.coil_maps.shape} in the spatial dims."
            )

    @staticmethod
    def check_coil_maps(coil_maps: Tensor, three_d: bool) -> Tensor:
        """Check coil maps dimensions.

        :param torch.Tensor coil_maps: coil sensitivity maps
        :return torch.Tensor: checked coil sensitivity maps
        """
        while len(coil_maps.shape) < (
            4 if not three_d else 5
        ):  # to B,N,H,W or B,N,D,H,W
            coil_maps = coil_maps.unsqueeze(0)

        if not coil_maps.is_complex():
            raise ValueError("coil_maps should be of torch complex dtype.")

        return coil_maps

    def simulate_birdcage_csm(self, n_coils: int) -> Tensor:
        """Simulate birdcage coil sensitivity maps. Requires library ``sigpy``.

        :param int n_coils: number of coils N
        :return torch.Tensor: coil maps of complex dtype of shape (N,H,W)
        """
        try:
            from sigpy.mri import birdcage_maps
        except ImportError:  # pragma: no cover
            raise ImportError(
                "sigpy is required to simulate coil maps. Install it using pip install sigpy"
            )

        coil_maps = birdcage_maps(
            (n_coils,)
            + (self.img_size[-2:] if not self.three_d else self.img_size[-3:])
        )
        return torch.tensor(coil_maps).type(torch.complex64)

    @staticmethod
    def estimate_coil_maps(
        y: Tensor,
        calib_size: int = 24,
        use_cupy: bool = False,
        espirit_crop: float = 0.95,
    ) -> Tensor:
        """Estimate coil sensitivity maps using ESPIRiT.

        This was proposed in `ESPIRiT — An Eigenvalue Approach to Autocalibrating Parallel MRI: Where SENSE meets GRAPPA <https://onlinelibrary.wiley.com/doi/10.1002/mrm.24751>`_.

        Note this uses a suboptimal undifferentiable unbatched implementation provided by `sigpy`.

        Optionally use `cupy` to accelerate on GPU, only if `cupy` is installed and a GPU is available.

        :param torch.Tensor y: multi-coil kspace measurements with shape [B,2,N,...,H,W] where N is coil dimension.
        :param int calib_size: optional square auto-calibration size in pixels, used by `sigpy`.
        :param bool use_cupy: whether to attempt to use cupy for GPU acceleration.
        :param float espirit_crop: optionally set crop argument of ESPIRiT algorithm, defaults to 0.95.
        :return: torch.Tensor of coil maps of complex dtype and shape [B,N,...,H,W]
        """
        try:
            from sigpy.mri.app import EspiritCalib
            import sigpy as sp  # pragma: no cover
        except ImportError:  # pragma: no cover
            raise ImportError(
                "sigpy is required to estimate sens maps. Install it using pip install sigpy"
            )

        if use_cupy:  # pragma: no cover
            try:
                import cupy as cp

                use_cupy = cp.cuda.is_available()
            except ImportError:
                warn(
                    "cupy is not installed, using cpu for coil map estimation. Install cupy to speed up computation."
                )
                use_cupy = False

        complex_y = MRIMixin.to_torch_complex(y)
        if use_cupy:  # pragma: no cover
            if y.device.type == "cuda":
                cupy_y = cp.from_dlpack(complex_y)
            else:
                cupy_y = cp.from_dlpack(complex_y.to("cuda"))

            cupy_maps = cp.stack(
                [
                    EspiritCalib(
                        yb,
                        calib_size,
                        show_pbar=False,
                        crop=espirit_crop,
                        device=cupy_y.device,
                    ).run()
                    for yb in cupy_y
                ]
            )
            torch_maps = torch.from_dlpack(cupy_maps)

        else:
            device = sp.Device(-1)
            maps = np.stack(
                [
                    EspiritCalib(
                        yb,
                        calib_size,
                        show_pbar=False,
                        crop=espirit_crop,
                        device=device,
                    ).run()
                    for yb in complex_y.numpy(force=True)
                ]
            )

            torch_maps = torch.from_numpy(maps)

        return torch_maps


class DynamicMRI(MRI, TimeMixin):
    r"""
    Single-coil accelerated dynamic magnetic resonance imaging.

    The linear operator operates in 2D+t videos and is defined as

    .. math::

        y_t = M_t Fx_t

    where :math:`M_t` applies a time-varying mask, and :math:`F` is the 2D discrete Fourier Transform.
    This operator has a simple singular value decomposition, so it inherits the structure of
    :class:`deepinv.physics.DecomposablePhysics` and thus have a fast pseudo-inverse and prox operators.

    The complex images :math:`x` and measurements :math:`y` should be of size (B, 2, T, H, W) where the first channel corresponds to the real part
    and the second channel corresponds to the imaginary part.

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update``.

    .. note::

        We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. :class:`deepinv.physics.generator.mri.RandomMaskGenerator`

    :param torch.Tensor mask: binary mask, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (T,H,W), (C,T,H,W) or (B,C,T,H,W) where H, W are the image height and width, T is time-steps, C is channels (typically 2) and B is batch size.
    :param tuple img_size: if mask not specified, flat mask of ones is created using ``img_size``, where ``img_size`` can be of any shape specified above. If mask provided, ``img_size`` is ignored.
    :param torch.device device: cpu or gpu.

    |sep|

    :Examples:

        Single-coil accelerated 2D+t MRI operator:

        >>> from deepinv.physics import DynamicMRI
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 2, 2, 2, 2) # Define random video of shape (B,C,T,H,W)
        >>> mask = torch.rand_like(x) > 0.75 # Define random 4x subsampling mask
        >>> physics = DynamicMRI(mask=mask) # Physics with given mask
        >>> physics.update(mask=mask) # Alternatively set mask on-the-fly
        >>> physics(x)
        tensor([[[[[-0.0000,  0.7969],
                   [-0.0000, -0.0000]],
        <BLANKLINE>
                  [[-0.0000, -1.9860],
                   [-0.0000, -0.4453]]],
        <BLANKLINE>
        <BLANKLINE>
                 [[[ 0.0000,  0.0000],
                   [-0.8137, -0.0000]],
        <BLANKLINE>
                  [[-0.0000, -0.0000],
                   [-0.0000,  1.1135]]]]])

    """

    def A(self, x: Tensor, mask: Tensor = None, **kwargs) -> torch.Tensor:
        mask = self.check_mask(self.mask if mask is None else mask).to(x.device)
        mask_flatten = self.flatten(mask.expand(*x.shape))

        y = self.unflatten(
            super().A(self.flatten(x), mask_flatten, check_mask=False),
            batch_size=x.shape[0],
        )
        self.update_parameters(mask=mask, check_mask=False, **kwargs)
        return y

    def A_adjoint(
        self, y: Tensor, mask: Tensor = None, mag: bool = False, **kwargs
    ) -> Tensor:
        """Adjoint operator.

        Optionally perform magnitude to reduce channel dimension.

        :param torch.Tensor y: input kspace of shape `(B,2,T,H,W)`
        :param torch.Tensor mask: optionally set mask on-the-fly, see class docs for shapes allowed.
        :param bool mag: perform complex magnitude.
        """
        mask = self.check_mask(self.mask if mask is None else mask).to(y.device)
        mask_flatten = self.flatten(mask.expand(*y.shape))
        x = self.unflatten(
            super().A_adjoint(
                self.flatten(y), mask=mask_flatten, check_mask=False, mag=mag
            ),
            batch_size=y.shape[0],
        )
        self.update_parameters(mask=mask, check_mask=False, **kwargs)

        return x

    def A_dagger(self, y: Tensor, mask: Tensor = None, **kwargs) -> torch.Tensor:
        return self.A_adjoint(y, mask=mask, **kwargs)

    def check_mask(self, mask: torch.Tensor = None, **kwargs) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,T,H,W.

        :param torch.nn.parameter.Parameter, float MRI subsampling mask.
        """
        while mask is not None and len(mask.shape) < 5:  # to B,C,T,H,W
            mask = mask.unsqueeze(0)

        return super().check_mask(mask=mask, three_d=self.three_d)

    def noise(self, x, **kwargs):
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :return torch.Tensor: noisy measurements
        """
        return self.noise_model(x, **kwargs) * self.mask

    def to_static(
        self, mask: torch.Tensor | None = None, device: str | torch.device = "cpu"
    ) -> MRI:
        """Convert dynamic MRI to static MRI by removing time dimension.

        :param torch.Tensor mask: new static MRI mask. If None, existing mask is flattened (summed) along the time dimension.
        :return MRI: static MRI physics
        """
        return MRI(
            mask=torch.clip(self.mask.sum(2), 0.0, 1.0) if mask is None else mask,
            img_size=self.img_size,
            device=device,
        )


class SequentialMRI(DynamicMRI):
    r"""
    Single-coil accelerated magnetic resonance imaging using sequential sampling.

    Let :math:`M` be a subsampling mask with given acceleration.
    :math:`M_t` is a time-varying mask with the sequential sampling pattern e.g. non-overlapping lines or spokes, such that :math:`S=\bigcup_t S_t`.
    The sequential MRI operator then simulates a time sequence of k-space samples:

    .. math::

        y_t = M_t F x

    where :math:`F` is the 2D discrete Fourier Transform, the image :math:`x` is of shape (B, 2, H, W) and measurements :math:`y` is of shape (B, 2, T, H, W)
    where the first channel corresponds to the real part and the second channel corresponds to the imaginary part.

    This operator has a simple singular value decomposition, so it inherits the structure of :class:`deepinv.physics.DecomposablePhysics`
    and thus have a fast pseudo-inverse and prox operators.

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``)
    or using ``update``.

    .. note::

        We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. :class:`deepinv.physics.generator.mri.RandomMaskGenerator`

    :param torch.Tensor mask: binary mask :math:`S_t,t=1\ldots T`, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (T,H,W), (C,T,H,W) or (B,C,T,H,W) where H, W are the image height and width, T is time-steps, C is channels (typically 2) and B is batch size.
    :param tuple img_size: if mask not specified, flat mask of ones is created using ``img_size``, where ``img_size`` can be of any shape specified above. If mask provided, ``img_size`` is ignored.
    :param torch.device device: cpu or gpu.

    |sep|

    :Examples:

        Single-coil accelerated sequential MRI operator:

        >>> from deepinv.physics import SequentialMRI
        >>> x = torch.randn(1, 2, 2, 2) # Define random image of shape (B,C,H,W)
        >>> mask = torch.zeros(1, 2, 3, 2, 2) # Empty demo time-varying mask with 3 frames
        >>> physics = SequentialMRI(mask=mask) # Physics with given mask
        >>> physics.update(mask=mask) # Alternatively set mask on-the-fly
        >>> physics(x).shape # MRI sequential samples
        torch.Size([1, 2, 3, 2, 2])

    """

    def A(self, x: Tensor, mask: Tensor = None, **kwargs) -> torch.Tensor:
        return super().A(
            self.repeat(x, self.mask if mask is None else mask), mask, **kwargs
        )

    def A_adjoint(
        self, y: Tensor, mask: Tensor = None, keep_time_dim=False, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the adjoint of the forward operator :math:`\tilde{x} = A^{\top}y`.

        :param torch.Tensor y: input tensor
        :param torch.nn.parameter.Parameter, float mask: input mask
        :param bool keep_time_dim: if ``True``, adjoint is calculated frame-by-frame. Used for visualisation. If ``False``, flatten the time dimension before calculating.
        :return: (:class:`torch.Tensor`) output tensor
        """
        if keep_time_dim:
            return super().A_adjoint(y, mask, **kwargs)
        else:
            mask = mask if mask is not None else self.mask
            return self.to_static(device=y.device).A_adjoint(
                self.average(y, mask), mask=self.average(mask), **kwargs
            )


class DynamicMultiCoilMRI(MultiCoilMRI, TimeMixin):
    r"""Multi-coil MRI for dynamic 2D or 3D sequences.

    The linear operator operates in 2D slices or 3D volumes and is defined as:

    .. math::

        y_{n,t} = \operatorname{diag}(p_t) F \operatorname{diag}(s_n) x_t

    for :math:`n=1,\dots,N` coils and :math:`t=1,\dots,T` time steps, where
    :math:`p_t` is the sampling mask at time :math:`t`, :math:`F` is the spatial
    Fourier transform, and :math:`s_n` is the sensitivity of the nth coil.

    The input image has shape ``(B, 2, T, H, W)`` and the output k-space has
    shape ``(B, 2, N, T, H, W)``, where ``N`` is the number of coils. Static
    coil sensitivity maps are applied independently to every time frame.

    :param torch.Tensor mask: dynamic mask with shape ``(B, 2, T, H, W)`` or
        any broadcast-compatible shape accepted by :class:`MultiCoilMRI`.
    :param torch.Tensor coil_maps: complex coil maps with shape ``(B,N,H,W)``.
    :param tuple img_size: image size used when ``mask`` or ``coil_maps`` is not
        specified.
    :param bool three_d: if ``True``, apply a 3D spatial Fourier transform.
    :param torch.device, str device: computation device.

    |sep|

    :Example:

    >>> import torch
    >>> from deepinv.physics import DynamicMultiCoilMRI
    >>> x = torch.randn(1, 2, 3, 8, 8)  # (B,2,T,H,W)
    >>> mask = torch.ones_like(x)
    >>> coil_maps = torch.ones(1, 4, 8, 8, dtype=torch.complex64)
    >>> physics = DynamicMultiCoilMRI(mask=mask, coil_maps=coil_maps)
    >>> physics(x).shape
    torch.Size([1, 2, 4, 3, 8, 8])
    """

    def _flatten_coil_maps(self, batch_size: int, time_size: int) -> Tensor:
        if self.coil_maps.shape[0] not in (1, batch_size):
            raise ValueError(
                f"Coil-map batch size {self.coil_maps.shape[0]} is incompatible "
                f"with image batch size {batch_size}."
            )
        coil_maps = self.coil_maps.expand(batch_size, *self.coil_maps.shape[1:])
        return (
            coil_maps[:, None]
            .expand(batch_size, time_size, *coil_maps.shape[1:])
            .reshape(batch_size * time_size, *coil_maps.shape[1:])
        )

    def update_parameters(
        self,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        check_mask: bool = True,
        check_coil_maps: bool = True,
        **kwargs,
    ):
        r"""Update parameters.

        :param torch.Tensor mask: mask tensor with shape ``(B, 2, T, H, W)``.
        :param torch.Tensor coil_maps: coil maps tensor with shape ``(B, N, H, W)``.
        :param bool check_mask: if ``True``, check if ``mask`` is broadcast-compatible.
        :param bool check_coil_maps: if ``True``, check if ``coil_maps`` is broadcast-compatible.
        """
        if mask is not None and check_mask:
            mask = self.check_mask(mask)
        if coil_maps is not None and check_coil_maps:
            coil_maps = self.check_coil_maps(coil_maps, three_d=self.three_d)
        LinearPhysics.update_parameters(self, mask=mask, coil_maps=coil_maps, **kwargs)
        if self.mask is not None:
            self.img_size = self.mask.shape[1:]

    def A(
        self,
        x: Tensor,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Applies the linear forward operator.

        Optionally update MRI mask or coil sensitivity maps on the fly.

        :param torch.Tensor x: input tensor with shape ``(B, 2, T, H, W)`` or ``(B, 2, T, D, H, W)``
        :param torch.Tensor mask: input temporal mask with shape ``(B, 2, T, H, W)`` or ``(B, 2, T, D, H, W)``
        :param torch.Tensor coil_maps: complex coil maps with shape ``(B,N,H,W)``.
        :returns: (:class:`torch.Tensor`) output tensor with shape ``(B, 2, N, T, H, W)`` or ``(B, 2, N, T, D, H, W)``
        """
        mask = self.check_mask(self.mask if mask is None else mask).to(x.device)
        mask = mask.expand_as(x)
        coil_maps = self.coil_maps if coil_maps is None else coil_maps
        coil_maps = self.check_coil_maps(coil_maps, three_d=self.three_d).to(x.device)
        self.coil_maps = coil_maps
        flat_coil_maps = self._flatten_coil_maps(x.shape[0], x.shape[2])

        y = self.unflatten(
            super().A(
                self.flatten(x),
                mask=self.flatten(mask),
                coil_maps=flat_coil_maps,
                check_mask=False,
                check_coil_maps=False,
            ),
            batch_size=x.shape[0],
            time_dim=3,
        )
        self.update_parameters(
            mask=mask,
            coil_maps=coil_maps,
            check_mask=False,
            check_coil_maps=False,
            **kwargs,
        )
        return y

    def A_adjoint(
        self,
        y: Tensor,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Applies the adjoint forward operator.

        Mathematically, the operator writes as

        .. math::
            A^{\top}y_t = \sum_{n=1}^{N} s_n F^{\top} \diag(p) y_{n, t}

        Optionally update MRI mask or coil sensitivity maps on the fly.

        :param torch.Tensor y: input tensor with shape ``(B, 2, N, T, H, W)`` or ``(B, 2, N, T, D, H, W)``
        :param torch.Tensor mask: input temporal mask with shape ``(B, 2, T, H, W)``
        :param torch.Tensor coil_maps: complex coil maps with shape ``(B,N,H,W)``.
        :returns: (:class:`torch.Tensor`) output tensor with shape ``(B, 2, T, H, W)`` or ``(B, 2, T, D, H, W)``
        """
        mask = self.check_mask(self.mask if mask is None else mask).to(y.device)
        coil_maps = self.coil_maps if coil_maps is None else coil_maps
        coil_maps = self.check_coil_maps(coil_maps, three_d=self.three_d).to(y.device)
        self.coil_maps = coil_maps
        flat_coil_maps = self._flatten_coil_maps(y.shape[0], y.shape[3])

        x = self.unflatten(
            super().A_adjoint(
                self.flatten(y, time_dim=3),
                mask=self.flatten(mask),
                coil_maps=flat_coil_maps,
                check_mask=False,
                check_coil_maps=False,
                **kwargs,
            ),
            batch_size=y.shape[0],
        )
        self.update_parameters(
            mask=mask,
            coil_maps=coil_maps,
            check_mask=False,
            check_coil_maps=False,
        )
        return x

    def rss(
        self,
        x: Tensor,
        multicoil: bool = True,
        mag: bool = True,
        three_d: bool | None = None,
    ) -> Tensor:
        r"""Perform root-sum-square reconstruction frame by frame.

        .. math::

                \operatorname{RSS}(x)_{t} = \sqrt{\sum_{n=1}^N |x_{n, t}|^2}

        :param torch.Tensor x: dynamic coil images with shape
            ``(B,2,N,T,H,W)`` or ``(B,2,N,T,D,H,W)``.
        :param bool multicoil: reduce over the coil dimension, defaults to
            ``True``.
        :param bool mag: reduce over the real/imaginary dimension, defaults to
            ``True``.
        :param bool three_d: validate 3D spatial inputs. Defaults to the
            physics' ``three_d`` setting.
        :return: RSS images with shape ``(B,1,T,H,W)`` (or ``(B,1,T,D,H,W)``) when ``mag=True``.

        Internally, :meth:`MultiCoilMRI.A_adjoint` calls this method with time
        already flattened into the batch dimension. Those static-shaped inputs
        are delegated directly to the parent implementation.
        """
        three_d = self.three_d if three_d is None else three_d
        dynamic_ndim = 7 if three_d else 6
        if x.ndim != dynamic_ndim:
            return super().rss(x, multicoil=multicoil, mag=mag, three_d=three_d)

        batch_size = x.shape[0]
        return self.unflatten(
            super().rss(
                self.flatten(x, time_dim=3),
                multicoil=multicoil,
                mag=mag,
                three_d=three_d,
            ),
            batch_size=batch_size,
        )

    def check_mask(self, mask: Tensor = None, **kwargs) -> Tensor:
        r"""
        Checks that mask can be broadcast in the (B, 2, T, ...) convention.

        :param torch.Tensor mask: mask of shape (B, 2, T, ...)
        """
        while mask is not None and mask.ndim < 5:
            mask = mask.unsqueeze(0)
        return super().check_mask(mask=mask, three_d=self.three_d)

    def to_static(
        self, mask: Tensor = None, device: str | torch.device = "cpu"
    ) -> MultiCoilMRI:
        r"""
        Convert dynamic multi-coil MRI to static multi-coil MRI.

        This conversion is performed by removing time dimension. The mask is built by retaining all the sampled locations across time, as

        .. math::
            \tilde{M} = \bigcup_t M_t = \operatorname{max}_t M_t


        .. note::
            The new operator cannot handle dynamic multi-coil MRI tensors.

        :param torch.Tensor mask: mask of shape (B, 2, T, ...)
        :param str device: device to convert to
        :return MultiCoilMRI physucs: equivalent temporal dimension free MulicoilMRI physics
        """
        mask = self.mask.amax(dim=2) if mask is None else mask
        return MultiCoilMRI(
            mask=mask,
            img_size=mask.shape[-3:] if self.three_d else mask.shape[-2:],
            coil_maps=self.coil_maps,
            device=device,
            three_d=self.three_d,
        )


class SequentialMultiCoilMRI(DynamicMultiCoilMRI):
    r"""Sequential multi-coil MRI of a static image.

    The static image is repeated over time and can optionally undergo a
    different motion transform in every frame. The forward operator is
    modelled as

    .. math::
        y_{n,t} = \operatorname{diag}(p_t) F \operatorname{diag}(s_n) T_t( \operatorname{timecat}(x)),

    where :math:`x` is the input image of shape ``(B,2,H,W)``, :math:`\operatorname{timecat}` is a concatenation operator
    over the time dimension (producing an image of shape ``(B,2,T,H,W)``), `T_t` is a time-indexed transform,
    :math:`s_n` is the n-th coil map, :math:`F` the Fourier transform, :math:`p_t`
    the undersampling mask at time :math:`t`, and :math:`y_{n,t}` the sampled k-space data of the n-th coil
    at time :math:`t`.

    :param TimeVaryingMotion motion: optional deterministic motion operator.
    :param dict[str, torch.Tensor] motion_params: optional motion parameters with leading dimensions
        ``(B,T)``. Parameters are stored as buffers.
    :param torch.Tensor mask: sequential mask with shape ``(B,2,T,H,W)`` or
        ``(B,2,T,D,H,W)``.
    :param torch.Tensor coil_maps: complex coil maps with shape ``(B,N,H,W)``
        or ``(B,N,D,H,W)``.
    :param tuple img_size: image size used when ``mask`` or ``coil_maps`` is not
        specified.
    :param bool three_d: if ``True``, apply a 3D spatial Fourier transform.
    :param torch.device, str device: computation device.

    |sep|

    :Example:

    >>> import torch
    >>> from deepinv.physics import SequentialMultiCoilMRI
    >>> x = torch.randn(1, 2, 8, 8)  # Static image (B,2,H,W)
    >>> mask = torch.zeros(1, 2, 3, 8, 8)
    >>> mask[:, :, 0, :, 1] = mask[:, :, 1, :, 3] = mask[:, :, 2, :, 6] = 1
    >>> coil_maps = torch.ones(1, 4, 8, 8, dtype=torch.complex64)
    >>> physics = SequentialMultiCoilMRI(mask=mask, coil_maps=coil_maps)
    >>> physics(x).shape
    torch.Size([1, 2, 4, 3, 8, 8])
    """

    def __init__(
        self,
        *args,
        motion: TimeVaryingMotion = None,
        motion_params: dict[str, Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if motion is not None and not isinstance(motion, TimeVaryingMotion):
            raise TypeError("motion must be an instance of TimeVaryingMotion.")
        if motion is None and motion_params:
            raise ValueError("motion_params were provided without a motion operator.")
        self.motion = motion
        if self.motion is not None:
            self.motion.to(self.mask.device)
            if motion_params is not None:
                self.motion.update(motion_params=motion_params)

    def update_parameters(
        self,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        motion_params: dict[str, Tensor] = None,
        **kwargs,
    ):
        """Update MRI and stored motion parameters."""
        super().update_parameters(mask=mask, coil_maps=coil_maps, **kwargs)
        if motion_params is not None:
            if self.motion is None and motion_params:
                raise ValueError(
                    "motion_params were provided without a motion operator."
                )
            if self.motion is not None:
                self.motion.update(motion_params=motion_params)

    def A(
        self,
        x: Tensor,
        mask: Tensor = None,
        motion_params: dict[str, Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Apply sequential multi-coil sampling to a static image.

        :param torch.Tensor x: static image with shape ``(B,2,H,W)`` or
            ``(B,2,D,H,W)``.
        :param torch.Tensor mask: optional sequential mask with shape
            ``(B,2,T,H,W)`` or ``(B,2,T,D,H,W)``.
        :param dict[str, torch.Tensor] motion_params: optional per-call motion
            parameters with leading dimensions ``(B,T)``.
        :return: Temporal measurements with shape ``(B,2,N,T,H,W)`` or
            ``(B,2,N,T,D,H,W)``.
        """
        mask = self.mask if mask is None else self.check_mask(mask)
        x = self.repeat(x, mask)
        if self.motion is not None:
            x = self.motion.A(x, motion_params=motion_params)
        return super().A(x, mask=mask, **kwargs)

    def A_adjoint(
        self,
        y: Tensor,
        mask: Tensor = None,
        motion_params: dict[str, Tensor] = None,
        keep_time_dim: bool = False,
        blind: bool = False,
        **kwargs,
    ) -> Tensor:
        r"""Apply the adjoint and optionally retain its temporal decomposition.

        Motion correction is applied frame by frame before the temporal sum.
        Set ``blind=True`` to omit motion correction while retaining all other
        adjoint operations.

        :param torch.Tensor y: temporal measurements with shape
            ``(B,2,N,T,H,W)`` or ``(B,2,N,T,D,H,W)``.
        :param torch.Tensor mask: optional sequential mask with shape
            ``(B,2,T,H,W)`` or ``(B,2,T,D,H,W)``.
        :param dict[str, torch.Tensor] motion_params: optional per-call motion
            parameters with leading dimensions ``(B,T)``.
        :param bool keep_time_dim: if ``True``, return one adjoint image per
            frame instead of summing over time.
        :param bool blind: if ``True``, do not apply the adjoint motion transform.
        :return: Adjoint image with shape ``(B,2,H,W)`` or ``(B,2,D,H,W)``.
            If ``keep_time_dim=True``, retain the time dimension after channel.
        """
        x = super().A_adjoint(y, mask=mask, **kwargs)
        if self.motion is not None and not blind:
            x = self.motion.A_adjoint(x, motion_params=motion_params)
        return x if keep_time_dim else x.sum(dim=2)
