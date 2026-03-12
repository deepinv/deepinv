"""
Positron emission tomography (PET) with parallelproj
====================================================


"""

import deepinv as dinv
from deepinv.physics import PET
import torch

# %%
# Define a phantom and attenuation map
# ------------------------------------
#
# We define a 3D PET phantom with an outer elliptical cylinder, an inner cylindrical region, and two pairs of spheres.
# The attenuation map is defined as a constant value in the outer cylinder and a lower value in the inner cylinder.
# The phantom is oversampled by a factor of 4 in each dimension and then downsampled by averaging to reduce aliasing artifacts.


def pet_phantom(
    in_shape: tuple[int, int, int],
    device: str = "cpu",
    mu_value: float = 0.01,
    add_spheres: bool = True,
    add_inner_cylinder: bool = True,
    r0: float = 0.45,
    r1: float = 0.28,
):
    """
    Generate a 3D PET phantom (torch native).

    Returns
    -------
    x_em : torch.Tensor
        Emission image (1,1,D,H,W)
    x_att : torch.Tensor
        Attenuation image (1,1,D,H,W)
    """

    oversampling_factor = 4
    D, H, W = in_shape
    oD, oH, oW = [oversampling_factor * x for x in in_shape]

    x_em = torch.zeros((oD, oH, oW), dtype=torch.float32, device=device)
    x_att = torch.zeros_like(x_em)

    c0 = oD / 2
    c1 = oH / 2
    c2 = oW / 2

    a = r0 * oD
    b = r1 * oH

    rix = oD / 25
    riy = oH / 25

    y, x = torch.meshgrid(
        torch.arange(oD, device=device),
        torch.arange(oH, device=device),
        indexing="ij",
    )

    outer_mask = ((x - c0) / a) ** 2 + ((y - c1) / b) ** 2 <= 1
    inner_mask = ((x - c0) / rix) ** 2 + ((y - c1) / riy) ** 2 <= 1

    for z in range(oW):
        x_em[:, :, z][outer_mask] = 1.0
        x_att[:, :, z][outer_mask] = mu_value

        if add_inner_cylinder:
            x_em[:, :, z][inner_mask] = 0.25
            x_att[:, :, z][inner_mask] = mu_value / 3

    if add_spheres:
        x, y, z = torch.meshgrid(
            torch.arange(oD, device=device),
            torch.arange(oH, device=device),
            torch.arange(oW, device=device),
            indexing="ij",
        )

        r_sp = [oW / 9] * 3
        r_sp2 = [oW / 17] * 3

        for z_offset in [c2, 0.45 * c2]:

            sp_mask = ((x - c0) / r_sp[0]) ** 2 + ((y - 1.4 * c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em[sp_mask] = 2.5

            sp_mask2 = ((x - 1.3 * c0) / r_sp[0]) ** 2 + ((y - c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em[sp_mask2] = 0.25

            sp_mask = ((x - c0) / r_sp2[0]) ** 2 + ((y - 0.6 * c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em[sp_mask] = 2.5

            sp_mask2 = ((x - 0.7 * c0) / r_sp2[0]) ** 2 + ((y - c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em[sp_mask2] = 0.25

    # downsample by averaging
    f = oversampling_factor

    def downsample(v):
        v = v.view(D, f, H, f, W, f)
        v = v.sum(dim=(1, 3, 5))
        return v / (f**3)

    x_em = downsample(x_em)
    x_att = downsample(x_att)

    x_em[:, :, :3] = 0
    x_em[:, :, -3:] = 0

    x_att[:, :, :2] = 0
    x_att[:, :, -2:] = 0

    # add batch + channel
    x_em = x_em.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    x_att = x_att.unsqueeze(0).unsqueeze(0)

    return x_em, x_att


# %%
# Setup a minimal non-TOF PET projector
# -------------------------------------
#
# We setup a minimal non-TOF PET projector of small scanner with
# three rings.

device = "cuda" if torch.cuda.is_available() else "cpu"
img_shape = (161, 161, 33)
physics = PET(device=device, img_shape=img_shape, normalize=True, gain=0.01)

# %%
# Forward projection and backprojection
# --------------------------------------------
# We forward project the phantom and then backproject the data to visualize the sensitivity map of the scanner.
# The sensitivity map is defined as the back-projection of a sinogram of ones, which corresponds to the number of LORs intersecting each voxel.

x, attenuation = pet_phantom(img_shape, device=device)
scatter = torch.ones_like(physics.A(x)) * 0.1
physics.update(attenuation=attenuation, scatter=scatter)
y = physics(x)
x_adj = physics.A_dagger(y)
sensitivities = physics.A_adjoint(torch.ones_like(y))

print(f"Norm operator: {physics.compute_norm(x):.2f}")
dinv.utils.plot(
    [
        y[..., 3].unsqueeze(0),
        x[..., 15],
        attenuation[..., 15],
        sensitivities[..., 15],
        x_adj[..., 15],
    ],
    titles=[
        "measurements",
        "Emission image",
        "Attenuation image",
        "sensitivities",
        "Backprojection of the data",
    ],
)

# %%
# Visualize the scanner geometry and image FOV
# --------------------------------------------


class Denoiser3D(dinv.models.Denoiser):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma=None):
        B, C, D, H, W = x.shape
        # reshape to (B * W, C, D, H)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * W, C, D, H)
        x = self.denoiser(x, sigma)
        # reshape back to (B, C, D, H, W)
        x = x.reshape(B, W, C, D, H).permute(0, 2, 3, 4, 1)
        return x


denoiser = Denoiser3D(dinv.models.DRUNet(in_channels=1, out_channels=1, device=device))

data_fidelity = dinv.optim.PoissonLikelihood(
    bkg=physics.scatter * physics.noise_model.gain,
    gain=physics.noise_model.gain,
    denormalize=True,
)
prior = dinv.optim.PnP(denoiser=denoiser)
stepsize = 1.0
x_pnp = torch.ones_like(x)
gain = physics.noise_model.gain
with torch.no_grad():
    for i in range(50):
        # grad_hand =  physics.A_adjoint((1 - (y / gain) / physics.A(x_pnp / gain, add_scatter=True)))
        grad = data_fidelity.grad(x=x_pnp, y=y, physics=physics) / gain
        x_pnp = x_pnp - stepsize * (x_pnp + 1e-9) / (sensitivities) * grad
        x_pnp = x_pnp * 0.9 + 0.1 * prior.prox(x_pnp, sigma_denoiser=0.03)
        x_pnp = torch.clamp(x_pnp, min=0)

dinv.utils.plot(
    [x_pnp[..., 15], x[..., 15]], titles=["DPIR reconstruction", "Ground truth"]
)
