from __future__ import annotations
import array_api_compat.torch as torch
import parallelproj
from array_api_compat import device as to_device
from typing import TYPE_CHECKING, Union
from .forward import LinearPhysics
import array_api_compat.numpy as np

if TYPE_CHECKING:
    import cupy as cp
    Array = Union[np.ndarray, cp.ndarray]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime

class PET(LinearPhysics):
    r"""
    Non time-of-flight Positron emission tomography (PET) physics model.

    This operator relies on the `parallelproj` library :cite:t:`schramm2024parallelproj`.

    The forward model is defined as

    .. math::

    .. note::

        This operator requires the `parallelproj` package to be installed. You can install it via conda:

        ::

            conda install -c conda-forge parallelproj

        Check the `parallelproj` documentation for more details: https://parallelproj.readthedocs.io/en/stable/.

    :param tuple img_shape: shape of the input 3D volumes, e.g. (D, H, W).
    :param float radius: radius of the regular polygon PET scanner.
    :param int num_sides: number of sides of the regular polygon PET scanner.
    :param int num_lor_endpoints_per_side: number of LOR endpoints per side of the regular polygon PET scanner.
    :param float lor_spacing: spacing between LOR endpoints in mm.
    :param torch.Tensor ring_positions: positions of the rings in mm.
    :param int symmetry_axis: symmetry axis of the scanner, e.g. 2 for z-axis.
    :param int radial_trim: radial trim in mm, i.e. maximum distance of LORs from the center of the scanner
    :param tuple voxel_size: voxel size in mm.
    :param float fwhm_data_mm: full width at half maximum (FWHM) of the data in mm, used for the Gaussian resolution model.
    :param torch.Tensor scatter: scatter sinogram, i.e. the expected number of scatter events in each LOR, with shape (num_lors,)
    :param torch.Tensor attenuation: attenuation map, i.e. the linear attenuation coefficient in each voxel, with shape (D, H, W)
    :param str | torch.device device: device to run the computations on, e.g. "cpu" or "cuda"
    """
    def __init__(self, img_shape : tuple, radius : float,
                 num_sides : int, num_lor_endpoints_per_side : int, lor_spacing : float,
                 ring_positions : torch.Tensor, symmetry_axis : int, radial_trim : int,
                 voxel_size: tuple, fwhm_data_mm : float,
                 scatter: torch.Tensor | None = None, attenuation: torch.Tensor | None = None,
                 normalize: bool = False,
                 device : str | torch.device = "cpu", **kwargs):
        super().__init__(**kwargs)
        if not isinstance(img_shape, tuple) or len(img_shape) != 3:
            raise ValueError("img_shape must be a tuple of length 3, e.g. (D, H, W)")
        scanner = parallelproj.RegularPolygonPETScannerGeometry(
            torch,
            device,
            radius=radius,
            num_sides=num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            ring_positions=ring_positions,
            symmetry_axis=symmetry_axis,
        )

        # setup the LOR descriptor that defines the sinogram
        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=radial_trim,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )

        self.proj = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=img_shape, voxel_size=voxel_size
        )

        scatter = self.proj(torch.zeros((1, 1,) + img_shape, device=device)) if scatter is None else scatter.to(device)
        attenuation = attenuation if attenuation is not None else torch.zeros((1, 1) + img_shape, device=device)

        self.res_model = parallelproj.GaussianFilterOperator(
            self.proj.in_shape, sigma=fwhm_data_mm / (2.35 * self.proj.voxel_size)
        )
        self.pet_lin_op = parallelproj.CompositeLinearOperator((self.proj, self.res_model))

        self.normalize = normalize
        self.norm = 1

        self.register_buffer("scatter", scatter)
        self.register_buffer("attenuation", attenuation)
        self.update_parameters(scatter=scatter, attenuation=attenuation)
        self.to(device)

    def A(self, x : torch.Tensor, add_scatter : bool = False, scatter : torch.Tensor | None =None,
          attenuation : torch.Tensor | None =None, **kwargs) -> torch.Tensor:
        self.update_parameters(attenuation=attenuation, scatter=scatter)
        out = LinearSingleChannelOperator.apply(x, self.pet_lin_op) * self.att_sino
        out /= self.norm
        if add_scatter:
            out = out + self.scatter
        return out

    def A_adjoint(self, y : torch.Tensor, attenuation=None, scatter=None,
                  **kwargs) -> torch.Tensor:
        self.update_parameters(attenuation=attenuation, scatter=scatter)
        return AdjointLinearSingleChannelOperator.apply(y * self.att_sino, self.pet_lin_op) / self.norm

    def forward(self, x: torch.Tensor, attenuation=None, scatter=None, **kwargs) -> torch.Tensor:
        self.update_parameters(attenuation=attenuation, scatter=scatter)
        return self.noise_model(self.A(x, **kwargs, add_scater=True))

    def update_parameters(self, attenuation: torch.Tensor | None = None,
                          scatter: torch.Tensor | None = None, **kwargs):
        if attenuation is not None:
            self.attenuation = attenuation
            proj_att = LinearSingleChannelOperator.apply(attenuation, self.proj)
            self.att_sino = torch.exp(-proj_att)
            if self.normalize:
                self.norm = 1
                self.norm = self.compute_norm(torch.ones_like(attenuation), squared=False)
        if scatter is not None:
            self.scatter = scatter



class LinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, num_voxels_x, num_voxels_y, num_voxels_z)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, operator.out_shape)
        """

        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use
        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size,) + operator.out_shape, dtype=x.dtype, device=to_device(x)
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, ...] = operator(x[i, 0, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, operator.out_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size, 1) + operator.in_shape,
                dtype=grad_output.dtype,
                device=to_device(grad_output),
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, 0, ...] = operator.adjoint(grad_output[i, ...].detach())

            return x, None


# %%
# Setup the back projection layer
# -------------------------------
#
# We subclass :class:`torch.autograd.Function` to define a custom pytorch layer
# that is compatible with pytorch's autograd engine.
# see also: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html


class AdjointLinearSingleChannelOperator(torch.autograd.Function):
    """
    Function representing the adjoint of a linear operator acting on a mini batch of single channel images
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, operator: parallelproj.LinearOperator
    ) -> torch.Tensor:
        """forward pass of the adjoint of the linear operator

        Parameters
        ----------
        ctx : context object
            that can be used to store information for the backward pass
        x : torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, operator.out_shape)
        operator : parallelproj.LinearOperator
            linear operator that can act on a single 3D image

        Returns
        -------
        torch.Tensor
            mini batch of 3D images with dimension (batch_size, 1, opertor.in_shape)
        """

        ctx.set_materialize_grads(False)
        ctx.operator = operator

        batch_size = x.shape[0]
        y = torch.zeros(
            (batch_size, 1) + operator.in_shape, dtype=x.dtype, device=to_device(x)
        )

        # loop over all samples in the batch and apply linear operator
        # to the first channel
        for i in range(batch_size):
            y[i, 0, ...] = operator.adjoint(x[i, ...].detach())

        return y

    @staticmethod
    def backward(ctx, grad_output):
        """backward pass of the forward pass

        Parameters
        ----------
        ctx : context object
            that can be used to obtain information from the forward pass
        grad_output : torch.Tensor
            mini batch of dimension (batch_size, 1, operator.in_shape)

        Returns
        -------
        torch.Tensor, None
            mini batch of 3D images with dimension (batch_size, 1, opertor.out_shape)
        """
        # For details on how to implement the backward pass, see
        # https://pytorch.org/docs/stable/notes/extending.html#how-to-use

        # since forward takes two input arguments (x, operator)
        # we have to return two arguments (the latter is None)
        if grad_output is None:
            return None, None
        else:
            operator = ctx.operator

            batch_size = grad_output.shape[0]
            x = torch.zeros(
                (batch_size,) + operator.out_shape,
                dtype=grad_output.dtype,
                device=to_device(grad_output),
            )

            # loop over all samples in the batch and apply linear operator
            # to the first channel
            for i in range(batch_size):
                x[i, ...] = operator(grad_output[i, 0, ...].detach())

            return x, None
