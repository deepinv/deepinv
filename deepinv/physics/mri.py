from typing import List, Optional, Union

import torch
from torch import Tensor
from torchvision.transforms import CenterCrop

from deepinv.physics.forward import DecomposablePhysics, LinearPhysics
from deepinv.physics.time import TimeMixin


class MRIMixin:
    r"""
    Mixin base class for MRI functionality.

    Base class that provides helper functions for FFT and mask checking.
    """

    def check_mask(
        self, mask: Tensor = None, three_d: bool = False, device: str = "cpu", **kwargs
    ) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,...,H,W where C=2.

        :param torch.nn.parameter.Parameter, torch.Tensor mask: MRI subsampling mask.
        :param bool three_d: If ``False`` the mask should be min 4 dimensions (B, C, H, W) for 2D data, otherwise if ``True`` the mask should have 5 dimensions (B, C, D, H, W) for 3D data.
        :param torch.device, str device: mask intended device.
        """
        if mask is not None:
            mask = mask.to(device)

            while len(mask.shape) < (
                4 if not three_d else 5
            ):  # to B,C,H,W or B,C,D,H,W
                mask = mask.unsqueeze(0)

            if mask.shape[1] == 1:  # make complex if real
                mask = torch.cat([mask, mask], dim=1)

        return mask

    @staticmethod
    def to_torch_complex(x: Tensor):
        """[B,2,...,H,W] real -> [B,...,H,W] complex"""
        return torch.view_as_complex(x.moveaxis(1, -1).contiguous())

    @staticmethod
    def from_torch_complex(x: Tensor):
        """[B,...,H,W] complex -> [B,2,...,H,W] real"""
        return torch.view_as_real(x).moveaxis(-1, 1)

    @staticmethod
    def ifft(x: Tensor, dim=(-2, -1), norm="ortho"):
        """Centered, orthogonal ifft

        :param torch.Tensor x: input kspace of complex dtype of shape [B,...] where ... is all dims to be transformed
        :param tuple dim: fft transform dims, defaults to (-2, -1)
        :param str norm: fft norm, see docs for :func:`torch.fft.fftn`, defaults to "ortho"
        """
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.ifftn(x, dim=dim, norm=norm)
        return torch.fft.fftshift(x, dim=dim)

    @staticmethod
    def fft(x: Tensor, dim=(-2, -1), norm="ortho"):
        """Centered, orthogonal fft

        :param torch.Tensor x: input image of complex dtype of shape [B,...] where ... is all dims to be transformed
        :param tuple dim: fft transform dims, defaults to (-2, -1)
        :param str norm: fft norm, see docs for :func:`torch.fft.fftn`, defaults to "ortho"
        """
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.fftn(x, dim=dim, norm=norm)
        return torch.fft.fftshift(x, dim=dim)

    def im_to_kspace(self, x: Tensor, three_d: bool = False) -> Tensor:
        """Convenience method that wraps fft.

        :param torch.Tensor x: input image of shape (B,2,...) of real dtype
        :param bool three_d: whether MRI data is 3D or not, defaults to False
        :return: Tensor: output measurements of shape (B,2,...) of real dtype
        """
        return self.from_torch_complex(
            self.fft(
                self.to_torch_complex(x), dim=(-3, -2, -1) if three_d else (-2, -1)
            )
        )

    def kspace_to_im(self, y: Tensor, three_d: bool = False) -> Tensor:
        """Convenience method that wraps inverse fft.

        :param torch.Tensor y: input measurements of shape (B,2,...) of real dtype
        :param bool three_d: whether MRI data is 3D or not, defaults to False
        :return: Tensor: output image of shape (B,2,...) of real dtype
        """
        return self.from_torch_complex(
            self.ifft(
                self.to_torch_complex(y), dim=(-3, -2, -1) if three_d else (-2, -1)
            )
        )

    def crop(self, x: Tensor, crop: bool = True) -> Tensor:
        """Center crop 2D image according to ``img_size``.

        This matches the RSS reconstructions of the original raw data in :class:`deepinv.datasets.FastMRISliceDataset`.

        If ``img_size`` has odd height, then adjust by one pixel to match FastMRI data.

        :param torch.Tensor x: input tensor of shape (...,H,W)
        :param bool crop: whether to perform crop, defaults to True
        """
        crop_size = self.img_size[-2:]
        odd_h = crop_size[0] % 2 == 1

        if odd_h:
            crop_size = (crop_size[0] + 1, crop_size[1])

        cropped = CenterCrop(crop_size)(x)

        if odd_h:
            cropped = cropped[..., :-1, :]

        return cropped if crop else x

    @staticmethod
    def rss(x: Tensor, multicoil: bool = True, three_d: bool = False) -> Tensor:
        """Perform root-sum-square reconstruction on multicoil data, defined as

        .. math::

                \operatorname{RSS}(x) = \sqrt{\sum_{n=1}^N |x_n|^2}

        where :math:`x_n` are the coil images of :math:`x`, :math:`|\cdot|` denotes the magnitude
        and :math:`N` is the number of coils. Note that the sum is performed voxel-wise.

        :param torch.Tensor x: input image of shape (B,2,...) where 2 represents
            real and imaginary channels
        :param bool multicoil: if ``True``, assume ``x`` is of shape (B,2,N,...),
            and reduce over coil dimension N too.
        """
        assert (
            x.shape[1] == 2 and not x.is_complex()
        ), "x should be of shape (B,2,...) and not of complex dtype."

        mc_dim = 1 if multicoil else 0
        th_dim = 1 if three_d else 0
        assert (
            len(x.shape) == 4 + mc_dim + th_dim
        ), "x should be of shape (B,2,...) for singlecoil data or (B,2,N,...) for multicoil data."

        ss = x.pow(2).sum(dim=1, keepdim=True)
        return ss.sum(dim=2).sqrt() if multicoil else ss.sqrt()


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

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update_parameters``.

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
        >>> physics.update_parameters(mask=mask) # Update mask on the fly
        >>> physics(x)
        tensor([[[[ 0.0000, -1.4290],
                  [ 0.4564, -0.0000]],
        <BLANKLINE>
                 [[ 0.0000,  1.8622],
                  [ 0.0603, -0.0000]]]])

    """

    def __init__(
        self,
        mask: Optional[Tensor] = None,
        img_size: Optional[tuple] = (320, 320),
        three_d: bool = False,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.three_d = three_d
        self.img_size = img_size

        if mask is None:
            mask = torch.ones(*img_size, device=device)

        # Check and update mask
        self.update_parameters(mask=mask.to(self.device))

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
    ):
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

    def update_parameters(self, mask: Tensor = None, check_mask: bool = True, **kwargs):
        """Update MRI subsampling mask.

        :param torch.nn.parameter.Parameter, torch.Tensor mask: MRI mask
        :param bool check_mask: check mask dimensions before updating
        """
        if mask is not None:
            self.mask = torch.nn.Parameter(
                (
                    self.check_mask(
                        mask=mask,
                        three_d=getattr(self, "three_d", False),
                        device=self.device,
                    )
                    if check_mask
                    else mask
                ),
                requires_grad=False,
            )


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
    :param torch.Tensor, str coil_maps: either ``Tensor``, integer, or ``None``. If complex valued (i.e. of complex dtype) coil sensitvity maps which should have shape (H,W), (N,H,W), (B,N,H,W) or (B,N,...,H,W).
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
        >>> physics.update_parameters(coil_maps=coil_maps) # Update coil maps on the fly
        >>> physics(x).shape
        torch.Size([1, 2, 5, 2, 2])

    """

    def __init__(
        self,
        mask: Optional[Tensor] = None,
        coil_maps: Optional[Union[Tensor, int]] = None,
        img_size: Optional[tuple] = (320, 320),
        three_d: bool = False,
        device=torch.device("cpu"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.device = device
        self.three_d = three_d

        if mask is None:
            mask = torch.ones(*img_size)

        if coil_maps is None:
            coil_maps = torch.ones(
                (self.img_size[-2:] if not self.three_d else self.img_size[-3:]),
                dtype=torch.complex64,
            )
        elif isinstance(coil_maps, int):
            coil_maps = self.simulate_birdcage_csm(n_coils=coil_maps)

        self.update_parameters(mask=mask.to(device), coil_maps=coil_maps.to(device))

    def A(self, x, mask=None, coil_maps=None, **kwargs):
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

    def A_adjoint(
        self,
        y,
        mask=None,
        coil_maps=None,
        rss: bool = False,
        crop: bool = False,
        **kwargs,
    ):
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
        assert y.shape[1] == 2, "y must be of shape (B,2,N,...,H,W)"
        self.update_parameters(mask=mask, coil_maps=coil_maps, **kwargs)

        My = self.to_torch_complex(self.mask[:, :, None] * y)  # [B,N,...,H,W]
        FiMy = self.ifft(My, dim=(-3, -2, -1) if self.three_d else (-2, -1))

        if rss:
            x = self.from_torch_complex(FiMy)
            x = self.rss(x, multicoil=True)  # [B,1,...,H,W]
        else:
            SiFiMy = torch.sum(torch.conj(self.coil_maps) * FiMy, dim=1)  # [B,...,H,W]
            x = self.from_torch_complex(SiFiMy)  # [B,2,...,H,W]

        return self.crop(x, crop=crop)

    def update_parameters(
        self,
        mask: Tensor = None,
        coil_maps: Tensor = None,
        check_mask: bool = True,
        **kwargs,
    ):
        """Update MRI subsampling mask and coil sensitivity maps.

        :param torch.nn.parameter.Parameter, torch.Tensor mask: MRI mask
        :param torch.nn.parameter.Parameter, torch.Tensor coil_maps: MRI coil sensitivity maps
        :param bool check_mask: check mask dimensions before updating
        """
        if mask is not None:
            self.mask = torch.nn.Parameter(
                (
                    self.check_mask(mask=mask, three_d=self.three_d, device=self.device)
                    if check_mask
                    else mask
                ),
                requires_grad=False,
            )

        if coil_maps is not None:
            while len(coil_maps.shape) < (
                4 if not self.three_d else 5
            ):  # to B,N,H,W or B,N,D,H,W
                coil_maps = coil_maps.unsqueeze(0)

            if not coil_maps.is_complex():
                raise ValueError("coil_maps should be of torch complex dtype.")

            self.coil_maps = torch.nn.Parameter(
                coil_maps.to(self.device), requires_grad=False
            )

    def simulate_birdcage_csm(self, n_coils: int):
        """Simulate birdcage coil sensitivity maps. Requires library ``sigpy``.

        :param int n_coils: number of coils N
        :return torch.Tensor: coil maps of complex dtype of shape (N,H,W)
        """
        try:
            from sigpy.mri import birdcage_maps
        except ImportError:
            raise ImportError(
                "sigpy is required to simulate coil maps. Install it using pip install sigpy"
            )

        coil_maps = birdcage_maps(
            (n_coils,)
            + (self.img_size[-2:] if not self.three_d else self.img_size[-3:])
        )
        return torch.tensor(coil_maps).type(torch.complex64)


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

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update_parameters``.

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
        >>> physics.update_parameters(mask=mask) # Alternatively set mask on-the-fly
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
        mask = self.check_mask(self.mask if mask is None else mask)

        mask_flatten = self.flatten(mask.expand(*x.shape)).to(x.device)
        y = self.unflatten(
            super().A(self.flatten(x), mask_flatten, check_mask=False),
            batch_size=x.shape[0],
        )

        self.update_parameters(mask=mask, **kwargs)
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
        mask = self.check_mask(self.mask if mask is None else mask)

        mask_flatten = self.flatten(mask.expand(*y.shape)).to(y.device)
        x = self.unflatten(
            super().A_adjoint(
                self.flatten(y), mask=mask_flatten, check_mask=False, mag=mag
            ),
            batch_size=y.shape[0],
        )

        self.update_parameters(mask=mask, **kwargs)
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

        return super().check_mask(mask=mask, device=self.device, three_d=self.three_d)

    def noise(self, x, **kwargs):
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :return torch.Tensor: noisy measurements
        """
        return self.noise_model(x, **kwargs) * self.mask

    def to_static(self, mask: Optional[torch.Tensor] = None) -> MRI:
        """Convert dynamic MRI to static MRI by removing time dimension.

        :param torch.Tensor mask: new static MRI mask. If None, existing mask is flattened (summed) along the time dimension.
        :return MRI: static MRI physics
        """
        return MRI(
            mask=torch.clip(self.mask.sum(2), 0.0, 1.0) if mask is None else mask,
            img_size=self.img_size,
            device=self.device,
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
    or using ``update_parameters``.

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
        >>> physics.update_parameters(mask=mask) # Alternatively set mask on-the-fly
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
            return self.to_static().A_adjoint(
                self.average(y, mask), mask=self.average(mask), **kwargs
            )
