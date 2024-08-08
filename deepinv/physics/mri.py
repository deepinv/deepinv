import torch
import torch.fft
from torch import Tensor
from typing import List, Optional, Union
import warnings
from deepinv.physics.forward import DecomposablePhysics


class MRI(DecomposablePhysics):
    r"""
    Single-coil accelerated magnetic resonance imaging.

    The linear operator operates in 2D slices and is defined as

    .. math::

        y = SFx

    where :math:`S` applies a mask (subsampling operator), and :math:`F` is the 2D discrete Fourier Transform.
    This operator has a simple singular value decomposition, so it inherits the structure of
    :meth:`deepinv.physics.DecomposablePhysics` and thus have a fast pseudo-inverse and prox operators.

    The complex images :math:`x` and measurements :math:`y` should be of size (B, 2, H, W) where the first channel corresponds to the real part
    and the second channel corresponds to the imaginary part.

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update_parameters``.

    .. note::

        We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. :class:`deepinv.physics.generator.mri.RandomMaskGenerator`

    :param torch.Tensor mask: binary mask, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (C,H,W), or (B,C,H,W) where H, W are the image height and width, C is channels (typically 2) and B is batch size.
    :param tuple img_size: if mask not specified, blank mask of ones is created using img_size, where img_size can be of any shape specified above.
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
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.img_size = img_size

        if mask is None:
            mask = torch.ones(*img_size)

        self.update_parameters(mask=mask.to(self.device))

    # def U(self, x):
    #     if self.mask.size(0) == 1:
    #         return x[:, self.mask[0, ...] > 0]
    #     elif x.size(0) == self.mask.size(0):
    #         return x[self.mask > 0]
    #     else:
    #         raise ValueError(
    #             "The batch size of the mask and the input should be the same."
    #         )
    #
    # def U_adjoint(self, x):
    #     _, c, h, w = self.mask.shape
    #     out = torch.zeros((x.shape[0], c, h, w), device=x.device)
    #
    #     if self.mask.size(0) == 1:
    #         out[:, self.mask[0, ...] > 0] = x
    #     elif x.size(0) == self.mask.size(0):
    #         out[self.mask > 0] = x
    #     else:
    #         raise ValueError(
    #             "The batch size of the mask and the input should be the same."
    #         )
    #     return out

    def V_adjoint(self, x: Tensor) -> Tensor:  # (B, 2, H, W) -> (B, H, W, 2)
        y = fft2c_new(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return y

    def V(self, x: Tensor) -> Tensor:  # (B, 2, H, W) -> (B, H, W, 2)
        x = x.permute(0, 2, 3, 1)
        return ifft2c_new(x).permute(0, 3, 1, 2)

    def update_parameters(self, mask=None, check_mask=True, **kwargs):
        return super().update_parameters(
            mask=self.check_mask(mask=mask) if check_mask else mask, **kwargs
        )

    def check_mask(self, mask: Tensor = None) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,H,W.

        :param torch.nn.Parameter, float MRI subsampling mask.
        """
        if mask is not None:
            mask = mask.to(self.device)

            while len(mask.shape) < 4:  # to B,C,H,W
                mask = mask.unsqueeze(0)

            if mask.shape[1] == 1:  # make complex if real
                mask = torch.cat([mask, mask], dim=1)

        return mask


class DynamicMRI(MRI):
    r"""
    Single-coil accelerated dynamic magnetic resonance imaging.

    The linear operator operates in 2D+t videos and is defined as

    .. math::

        y_t = S_t Fx_t

    where :math:`S` applies a time-varying mask, and :math:`F` is the 2D discrete Fourier Transform.
    This operator has a simple singular value decomposition, so it inherits the structure of
    :meth:`deepinv.physics.DecomposablePhysics` and thus have a fast pseudo-inverse and prox operators.

    The complex images :math:`x` and measurements :math:`y` should be of size (B, 2, T, H, W) where the first channel corresponds to the real part
    and the second channel corresponds to the imaginary part.

    A fixed mask can be set at initialisation, or a new mask can be set either at forward (using ``physics(x, mask=mask)``) or using ``update_parameters``.

    .. note::

        We provide various random mask generators (e.g. Cartesian undersampling) that can be used directly with this physics. See e.g. :class:`deepinv.physics.generator.mri.RandomMaskGenerator`

    :param torch.Tensor mask: binary mask, where 1s represent sampling locations, and 0s otherwise.
        The mask size can either be (H,W), (T,H,W), (C,T,H,W) or (B,C,T,H,W) where H, W are the image height and width, T is time-steps, C is channels (typically 2) and B is batch size.
    :param tuple img_size: if mask not specified, blank mask of ones is created using img_size, where img_size can be of any shape specified above.
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

    def flatten(self, a: Tensor) -> Tensor:
        B, C, T, H, W = a.shape
        return a.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    def unflatten(self, a: Tensor, batch_size=1) -> Tensor:
        BT, C, H, W = a.shape
        return a.reshape(batch_size, BT // batch_size, C, H, W).permute(0, 2, 1, 3, 4)

    def A(self, x: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        mask = self.check_mask(self.mask if mask is None else mask)

        mask_flatten = self.flatten(mask.expand(*x.shape)).to(x.device)
        y = self.unflatten(
            super().A(self.flatten(x), mask_flatten, check_mask=False),
            batch_size=x.shape[0],
        )

        self.update_parameters(mask=mask, **kwargs)
        return y

    def A_adjoint(self, y: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        mask = self.check_mask(self.mask if mask is None else mask)

        mask_flatten = self.flatten(mask.expand(*y.shape)).to(y.device)
        x = self.unflatten(
            super().A_adjoint(self.flatten(y), mask=mask_flatten, check_mask=False),
            batch_size=y.shape[0],
        )

        self.update_parameters(mask=mask, **kwargs)
        return x

    def A_dagger(self, y: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        return self.A_adjoint(y, mask=mask, **kwargs)

    def check_mask(
        self,
        mask: torch.Tensor = None,
    ) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,T,H,W.

        :param torch.nn.Parameter, float MRI subsampling mask.
        """
        while mask is not None and len(mask.shape) < 5:  # to B,C,T,H,W
            mask = mask.unsqueeze(0)

        return super().check_mask(mask=mask)

    def noise(self, x, **kwargs):
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :return torch.Tensor: noisy measurements
        """
        return self.noise_model(x, **kwargs) * self.mask

    def to_static_mri(self, mask: Optional[torch.Tensor] = None) -> MRI:
        """Convert dynamic MRI to static MRI by removing time dimension.

        :param torch.Tensor mask: new static MRI mask. If None, existing mask is flattened (summed) along the time dimension.
        :return MRI: static MRI physics
        """
        return MRI(
            mask=self.mask.sum(2) if mask is None else mask,
            img_size=self.img_size,
            device=self.device,
        )


#
# reference: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    r"""
    Apply centered 2 dimensional Fast Fourier Transform.
    :param torch.Tensor data: Complex valued input data containing at least 3 dimensions:
        dimensions -2 & -1 are spatial dimensions and dimension -3 has size
        2. All other dimensions are assumed to be batch dimensions.
    :param bool norm: Normalization mode. See ``torch.fft.fft``.
    :return: (torch.tensor) the FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -2 & -1 are spatial dimensions and dimension -3 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.
    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions
def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for s, d in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)
