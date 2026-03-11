"""
Positron emission tomography (PET) with parallelproj
====================================================


"""
from __future__ import annotations
import array_api_compat.torch as torch
import parallelproj
from array_api_compat import device as to_device
import deepinv as dinv


class PositronEmissionTomography(dinv.physics.LinearPhysics):
    r"""
    Non time-of-flight Positron emission tomography (PET) physics model using the `parallelproj` package.

    .. note::

        This operator requires the `parallelproj` package to be installed. You can install it via conda:

        ::

            conda install -c conda-forge parallelproj

        Check the `parallelproj` documentation for more details: https://parallelproj.readthedocs.io/en/stable/.

    """
    def __init__(self, img_shape : tuple = (20, 5, 20), radius : float =35.0,
                 num_sides : int = 12, num_lor_endpoints_per_side : int = 6, lor_spacing : float = 3.0,
                 ring_positions : torch.Tensor = torch.linspace(-4, 4, 3),symmetry_axis : int =1, radial_trim : int =10,
                 max_ring_difference: int =1, scatter : torch.Tensor | None = None, attenuation : torch.Tensor | None = None,
                 voxel_size: tuple=(2.0, 2.0, 2.0), fwhm_data_mm: float = 5,
                 device : str | torch.device = "cpu", **kwargs):
        super().__init__(**kwargs)

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
            max_ring_difference=max_ring_difference,
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )

        self.proj = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=img_shape, voxel_size=voxel_size
        )

        scatter = self.proj(torch.zeros(img_shape, device=device)) if scatter is None else scatter.to(device)
        attenuation = attenuation if attenuation is not None else torch.zeros(img_shape, device=device)

        att_sino = torch.exp(-self.proj(attenuation))
        att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

        self.res_model = parallelproj.GaussianFilterOperator(
            self.proj.in_shape, sigma=fwhm_data_mm / (2.35 * self.proj.voxel_size)
        )
        self.pet_lin_op = parallelproj.CompositeLinearOperator((att_op, self.proj, self.res_model))

        self.register_buffer("scatter", scatter)
        self.register_buffer("attenuation", attenuation)
        self.update_parameters(scatter=scatter, attenuation=attenuation)
        self.to(device)

    def A(self, x : torch.Tensor, add_scatter=False, scatter=None,
          attenuation=None, **kwargs) -> torch.Tensor:
        self.update_parameters(attenuation=attenuation, scatter=scatter)
        out = LinearSingleChannelOperator.apply(x, self.pet_lin_op)

        if add_scatter:
            out = out + self.scatter
        return out

    def A_adjoint(self, y : torch.Tensor, attenuation=None, scatter=None,
                  **kwargs) -> torch.Tensor:
        self.update_parameters(attenuation=attenuation, scatter=scatter)
        return AdjointLinearSingleChannelOperator.apply(y, self.pet_lin_op)

    def forward(self, x: torch.Tensor, attenuation=None, scatter=None, **kwargs) -> torch.Tensor:
        self.update_parameters(attenuation=attenuation, scatter=scatter)
        return self.noise_model(self.A(x, **kwargs, add_scater=True))

    def update_parameters(self, attenuation: torch.Tensor | None = None,
                          scatter: torch.Tensor | None = None, **kwargs):
        if attenuation is not None:
            self.attenuation = attenuation
            att_sino = torch.exp(-self.proj(attenuation))
            att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)
            self.pet_lin_op = parallelproj.CompositeLinearOperator((att_op, self.proj, self.res_model))
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


# %%
# Setup a minimal non-TOF PET projector
# -------------------------------------
#
# We setup a minimal non-TOF PET projector of small scanner with
# three rings.

device = "cuda" if torch.cuda.is_available() else "cpu"
physics = PositronEmissionTomography(device=device)

x = torch.zeros((2, 1, 20, 5, 20), device=device)
x[:, 0, 8:12, 2:4, 8:12] = 1.0
y = physics.A(x)
x_adj = physics.A_dagger(y, 'lsqr')

# %%
# Visualize the scanner geometry and image FOV
# --------------------------------------------

dinv.utils.plot(x_adj[:, :, :, 0, :])