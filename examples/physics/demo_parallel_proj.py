"""
Positron emission tomography (PET) with parallelproj
====================================================


"""
from __future__ import annotations
import array_api_compat.torch as torch
import parallelproj
from array_api_compat import device as to_device
import deepinv as dinv
from typing import TYPE_CHECKING, Union
from types import ModuleType
import array_api_compat.numpy as np

if TYPE_CHECKING:
    import cupy as cp

    Array = Union[np.ndarray, cp.ndarray]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime


def pet_phantom(
    in_shape: tuple[int, int, int],
    xp: ModuleType,
    dev,
    mu_value: float = 0.01,
    add_spheres: bool = True,
    add_inner_cylinder: bool = True,
    r0=0.45,
    r1=0.28,
) -> tuple[Array, Array]:
    """
    Generate a 3D PET phantom.

    Parameters
    ----------
    in_shape : tuple
        Shape of the phantom.
    xp : module
        Array API module to use.
    dev : str
        Device to use.
    mu_value : float
        Attenuation coefficient.
    Returns
    -------
    tuple
        Emission and attenuation images.

    Note
    ----

    The activity in the background should be 1.
    """
    if dev == "cpu":
        dev = None

    oversampling_factor = 4

    # elliptical phantom on oversampled grid
    oversampled_shape = tuple(oversampling_factor * x for x in in_shape)
    x_em_oversampled = xp.zeros(oversampled_shape, device=dev, dtype=xp.float32)
    x_att_oversampled = xp.zeros(oversampled_shape, device=dev, dtype=xp.float32)
    c0 = oversampled_shape[0] / 2
    c1 = oversampled_shape[1] / 2
    c2 = oversampled_shape[2] / 2
    a = r0 * oversampled_shape[0]  # semi-major axis
    b = r1 * oversampled_shape[1]  # semi-minor axis

    rix = oversampled_shape[0] / 25
    riy = oversampled_shape[1] / 25

    y, x = xp.ogrid[: oversampled_shape[0], : oversampled_shape[1]]

    outer_mask = ((x - c0) / a) ** 2 + ((y - c1) / b) ** 2 <= 1
    inner_mask = ((x - c0) / rix) ** 2 + ((y - c1) / riy) ** 2 <= 1

    for z in range(oversampled_shape[2]):
        x_em_oversampled[:, :, z][outer_mask] = 1.0
        x_att_oversampled[:, :, z][outer_mask] = mu_value

        if add_inner_cylinder:
            x_em_oversampled[:, :, z][inner_mask] = 0.25
            x_att_oversampled[:, :, z][inner_mask] = mu_value / 3

    # add a few spheres to the emission image

    if add_spheres:
        x, y, z = xp.ogrid[
            : oversampled_shape[0], : oversampled_shape[1], : oversampled_shape[2]
        ]

        r_sp = 3 * [oversampled_shape[2] / 9]
        r_sp2 = 3 * [oversampled_shape[2] / 17]

        for z_offset in [c2, 0.45 * c2]:
            sp_mask = ((x - c0) / r_sp[0]) ** 2 + ((y - 1.4 * c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask] = 2.5

            sp_mask2 = ((x - 1.3 * c0) / r_sp[0]) ** 2 + ((y - c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask2] = 0.25

            sp_mask = ((x - c0) / r_sp2[0]) ** 2 + ((y - 0.6 * c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask] = 2.5

            sp_mask2 = ((x - 0.7 * c0) / r_sp2[0]) ** 2 + ((y - c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em_oversampled[sp_mask2] = 0.25

    # downsample to original grid size by averaging
    x_em_oversampled = (
        x_em_oversampled[::4, :, :]
        + x_em_oversampled[1::4, :, :]
        + x_em_oversampled[2::4, :, :]
        + x_em_oversampled[3::4, :, :]
    )
    x_em_oversampled = (
        x_em_oversampled[:, ::4, :]
        + x_em_oversampled[:, 1::4, :]
        + x_em_oversampled[:, 2::4, :]
        + x_em_oversampled[:, 3::4, :]
    )
    x_em_oversampled = (
        x_em_oversampled[:, :, ::4]
        + x_em_oversampled[:, :, 1::4]
        + x_em_oversampled[:, :, 2::4]
        + x_em_oversampled[:, :, 3::4]
    )

    x_att_oversampled = (
        x_att_oversampled[::4, :, :]
        + x_att_oversampled[1::4, :, :]
        + x_att_oversampled[2::4, :, :]
        + x_att_oversampled[3::4, :, :]
    )
    x_att_oversampled = (
        x_att_oversampled[:, ::4, :]
        + x_att_oversampled[:, 1::4, :]
        + x_att_oversampled[:, 2::4, :]
        + x_att_oversampled[:, 3::4, :]
    )
    x_att_oversampled = (
        x_att_oversampled[:, :, ::4]
        + x_att_oversampled[:, :, 1::4]
        + x_att_oversampled[:, :, 2::4]
        + x_att_oversampled[:, :, 3::4]
    )

    x_em = x_em_oversampled.copy() / (oversampling_factor**3)
    x_att = x_att_oversampled.copy() / (oversampling_factor**3)

    x_em[:, :, :3] = 0
    x_em[:, :, -3:] = 0

    # make the attenuation a bit wider in z (plastic wall)
    x_att[:, :, :2] = 0
    x_att[:, :, -2:] = 0

    return x_em, x_att



class PositronEmissionTomography(dinv.physics.LinearPhysics):
    r"""
    Non time-of-flight Positron emission tomography (PET) physics model using the `parallelproj` package.

    .. note::

        This operator requires the `parallelproj` package to be installed. You can install it via conda:

        ::

            conda install -c conda-forge parallelproj

        Check the `parallelproj` documentation for more details: https://parallelproj.readthedocs.io/en/stable/.

    """
    def __init__(self, img_shape : tuple, radius : float,
                 num_sides : int, num_lor_endpoints_per_side : int, lor_spacing : float,
                 ring_positions : torch.Tensor, symmetry_axis : int, radial_trim : int,
                 max_ring_difference: int, scatter : torch.Tensor | None, attenuation : torch.Tensor | None,
                 voxel_size: tuple, fwhm_data_mm : float,
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
            #max_ring_difference=max_ring_difference,
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

    def A(self, x : torch.Tensor, add_scatter : bool = False, scatter : torch.Tensor | None =None,
          attenuation : torch.Tensor | None =None, **kwargs) -> torch.Tensor:
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
num_rings = 17
radius = 300.0
num_sides = 36
num_lor_endpoints_per_side = 12
lor_spacing = 4.0
fmw_data_mm = 4.0
ring_positions = torch.linspace(
    -5 * (num_rings - 1) / 2, 5 * (num_rings - 1) / 2, num_rings
)
symmetry_axis = 2
radial_trim = 40
max_ring_difference = 5
img_shape = (161, 161, 2 * num_rings - 1)
voxel_size = (2.5, 2.5, 2.5)

x, attenuation = pet_phantom(img_shape, np, "cpu")
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
attenuation = torch.from_numpy(attenuation).to(device)
attenuation = attenuation
physics = PositronEmissionTomography(device=device, img_shape=img_shape, radius=radius, radial_trim=radial_trim, max_ring_difference=max_ring_difference,
                                     num_sides=num_sides, num_lor_endpoints_per_side=num_lor_endpoints_per_side,
                                     lor_spacing=lor_spacing, ring_positions=ring_positions, symmetry_axis=symmetry_axis,
                                     voxel_size=voxel_size, fwhm_data_mm=fmw_data_mm,
                                     attenuation=attenuation, scatter=None,
                                     noise_model=dinv.physics.PoissonNoise(gain=10.))


y = physics(x, attenuation=attenuation)
x_adj = physics.A_dagger(y)

# %%
# Visualize the scanner geometry and image FOV
# --------------------------------------------


sensitivities = physics.A_adjoint(torch.ones_like(y))

dinv.utils.plot([y[..., 3].unsqueeze(0), x[...,15], attenuation[...,15].unsqueeze(0).unsqueeze(0),
                 sensitivities[..., 15], x_adj[..., 15]],
                titles=["measuremnets", "Emission image",
                        "Attenuation image", "sensitivities", "Backprojection of the data"],)