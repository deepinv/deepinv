r"""
Cryo-electron tomography of experimental data
=============================================

Reconstruct an experimental cryo-electron tomography (cryo-ET) tilt series with the weighted back-projection (WBP), the pretrained network CryoLithe :footcite:p:`kishore2025cryolithe` and the self-supervised method Icecream :footcite:p:`kishore2026icecream`.

The aligned tilt series (EMPIAR-11830) has 41 tilts of size :math:`1024 \times 1024` with pixel size 7.84 Å, acquired in :math:`[-72^\circ, 48^\circ]`.
It is downloaded (172 MB) from the `CryoLithe training dataset <https://huggingface.co/datasets/sada-group/CryoLithe-training-dataset>`_.
See :ref:`sphx_glr_auto_examples_physics_demo_cryo_et.py` for the simulation of tilt series.
"""

# %%

# sphinx_gallery_tags = ["Tomography", "Microscopy", "Self-supervised"]

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import deepinv as dinv

device = dinv.utils.get_device()
# 3D grid sampling backward is not implemented on MPS
physics_device = torch.device("cpu") if device.type == "mps" else device
torch.manual_seed(0)

# %%
# Data
# ----
#
# The tilt series is stored as an MRC file: a 1024-byte header, containing the size and the data type, followed by an extended header and the data.
# We read it with NumPy; general MRC files can be read with `mrcfile <https://mrcfile.readthedocs.io>`_.

repo = dict(repo_id="sada-group/CryoLithe-training-dataset", repo_type="dataset")
name = "empiar-11830/tomo_001/{}_06022023_BrnoKrios_Arctis_xe_Position_70.{}"
tilt_path = hf_hub_download(filename=name.format("tilt_series", "mrc"), **repo)
angles_path = hf_hub_download(filename=name.format("angles", "tlt"), **repo)

header = np.fromfile(tilt_path, dtype=np.int32, count=256)
nx, ny, nz, mode = header[:4]
dtype = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}[mode]
data = np.fromfile(tilt_path, dtype=dtype, offset=1024 + header[23])
tilts = torch.from_numpy(data.reshape(nz, ny, nx).astype(np.float32))
angles = torch.from_numpy(np.loadtxt(angles_path).astype(np.float32))
pixel_size = 7.84  # Å

# %%
# Under the weak-phase approximation, tilt :math:`i` is :math:`I_i = s_i (1 + h_i * P_{\theta_i} x)`. We use the relative contrast :math:`I_i / \bar I_i - 1`, which removes the factors :math:`s_i` (hence `ice_thickness=0`).
# We use a nominal defocus of 4 µm. At the pixel sizes below, the CTF is negative at all frequencies, so that phase flipping inverts the contrast, and radiation damage only affects frequencies above Nyquist, so we disable dose weighting.
# Tilts are stored as `(N, H, W)` with the tilt axis along `H`, as in :class:`deepinv.physics.CryoET`.

y_full = tilts / tilts.mean(dim=(-1, -2), keepdim=True) - 1
i0 = int(torch.argmin(angles.abs()))
dinv.utils.plot(
    [y_full[None, None, i] for i in (0, i0, -1)],
    titles=[f"{angles[i]:.0f}°" for i in (0, i0, -1)],
    suptitle="Tilts (relative contrast)",
)


def cryo_et(factor, angles):
    r"""Operator for the tilt series binned by `factor`, with a volume of depth half the tilt width."""
    return dinv.physics.CryoET(
        img_size=(nx // (2 * factor), ny // factor, nx // factor),
        angles=angles.to(physics_device),
        pixel_size=pixel_size * factor,
        defocus=4.0,
        dose_weighting=False,
        ice_thickness=0.0,
        device=physics_device,
    )


# %%
# Weighted back-projection
# ------------------------
#
# :meth:`deepinv.physics.CryoET.fbp` phase-flips the tilts, ramp-filters them perpendicularly to the tilt axis, weights them by their angular spacing and back-projects them.
# We bin the tilts by 8 (pixel size 62.7 Å), giving a :math:`64 \times 128 \times 128` volume. The missing wedge elongates structures along the beam (`D`).

x_wbp = cryo_et(8, angles).fbp(F.avg_pool2d(y_full, 8)[None, None].to(physics_device))
dinv.utils.plot_ortho3D(x_wbp, titles="WBP (binned by 8)", figsize=(5, 5))

# %%
# Deep-learning reconstruction with CryoLithe
# -------------------------------------------
#
# :class:`deepinv.models.CryoLithe` is a supervised network estimating a denoised, missing-wedge-corrected volume from an aligned tilt series (see the `original code <https://github.com/swing-research/CryoLithe>`_).
# Each wavelet coefficient of the volume is estimated by an MLP from :math:`21 \times 21` patches of the ramp-filtered tilts around its projections. Pretrained weights are downloaded from Hugging Face.
#
# As patches are local, the network is applied at its training pixel size (about 8 Å), i.e. on the raw tilts.
# As voxels are estimated independently, we reconstruct a :math:`96 \times 192 \times 192` region of interest only. The output keeps the contrast of the tilts, which we invert.
#
# .. warning::
#
#     This tilt series is part of the CryoLithe training set, so this result is not representative of the performance on unseen data.

nn_device = dinv.utils.get_device(verbose=False)
cryolithe = dinv.models.CryoLithe(
    pretrained="download",
    variant="cryolithe",
    dtype=torch.float16 if nn_device.type == "cuda" else torch.float32,
    device=nn_device,
)
roi = (slice(208, 304), slice(416, 608), slice(416, 608))
x_cryolithe = -cryolithe(
    tilts[None, None].to(nn_device),
    angles=angles.to(nn_device),
    depth=nx // 2,
    roi=roi,
).to(device)

# %%
# Self-supervised reconstruction with Icecream
# --------------------------------------------
#
# Icecream trains a 3D network :math:`f` on sub-tomograms of two tomograms :math:`y_i = Mx + n_i` with independent noise, with :math:`M` the :class:`missing-wedge operator <deepinv.physics.MissingWedge>`.
# The :class:`deepinv.loss.IcecreamLoss` combines a Noise2Noise term in the measured wedge, for denoising, and an equivariance term with :class:`cube symmetries <deepinv.transform.CubeSymmetry>`, which move reliable frequencies of the estimates into the missing wedge, where the network must recover them.
# See the `original code <https://github.com/swing-research/Icecream>`_.
#
# We reconstruct the even and odd tilts, binned by 4 (pixel size 31.4 Å), with the WBP, and standardize the two tomograms.

y4 = F.avg_pool2d(y_full, 4)[None, None].to(physics_device)
standardize = lambda v: (v - v.mean()) / v.std()
tomograms = torch.cat(
    [
        standardize(cryo_et(4, angles[s]).fbp(y4[:, :, s]))
        for s in (slice(0, None, 2), slice(1, None, 2))
    ],
    dim=1,
).to(device)

# %%
# We train :class:`deepinv.models.Icecream` on random pairs of sub-tomograms at the same location in the two halves, stacked along channels.
# The paper trains a larger network on :math:`72^3` sub-tomograms for 50,000 iterations. To keep this example short, we use a smaller network and few iterations. Increase them for better results.


class SubTomogramPairs(torch.utils.data.Dataset):
    r"""Random pairs of sub-tomograms at the same location in two half tomograms of shape `(2, D, H, W)`."""

    def __init__(self, tomograms, size, n_samples):
        self.tomograms, self.size, self.n_samples = tomograms, size, n_samples
        self.generator = torch.Generator().manual_seed(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        i, j, k = (
            int(torch.randint(s - self.size + 1, (1,), generator=self.generator))
            for s in self.tomograms.shape[1:]
        )
        n = self.size
        return {"y": self.tomograms[:, i : i + n, j : j + n, k : k + n]}


if device.type == "cuda":
    crop_size, channels, batch_size, n_iterations, lr = (
        64,
        [32, 64, 128, 256],
        8,
        4000,
        1e-4,
    )
else:
    crop_size, channels, batch_size, n_iterations, lr = 32, [16, 32, 64], 4, 600, 3e-4

angle_range = (angles.min().item(), angles.max().item())
physics_mw = dinv.physics.MissingWedge(
    img_size=(crop_size,) * 3, angle_range=angle_range, device=device
)
model = dinv.models.Icecream(channels_per_scale=channels, device=device)
trainer = dinv.Trainer(
    model=model,
    physics=physics_mw,
    optimizer=torch.optim.Adam(model.parameters(), lr=lr),
    train_dataloader=torch.utils.data.DataLoader(
        SubTomogramPairs(tomograms[0].cpu(), crop_size, n_iterations * batch_size),
        batch_size=batch_size,
    ),
    losses=dinv.loss.IcecreamLoss(
        transform=dinv.transform.CubeSymmetry(rng=torch.Generator().manual_seed(0))
    ),
    epochs=1,
    online_measurements=False,
    metrics=None,
    save_path=None,
    plot_images=False,
    device=device,
)
model = trainer.train()

x_icecream = model.reconstruct(tomograms, physics_mw, batch_size=batch_size)
x_avg = tomograms.mean(1, keepdim=True)
dinv.utils.plot_ortho3D(
    [x_avg, x_icecream],
    titles=["WBP (average of halves)", f"Icecream ({n_iterations} iterations)"],
    figsize=(8, 5),
)

# %%
# The power spectrum in the :math:`(k_x, k_z)` plane, perpendicular to the tilt axis, shows the missing wedge of the WBP, outside the measured wedge (red), which Icecream starts filling.

mask = dinv.physics.functional.missing_wedge_mask(
    x_avg.shape[-3:], angle_range=angle_range, spherical_support=False
)
fig, axs = plt.subplots(1, 2, figsize=(8, 2.6))
for ax, vol, title in zip(axs, (x_avg, x_icecream), ("WBP", "Icecream")):
    power = torch.fft.fftshift(torch.fft.fftn(vol[0, 0].float().cpu()).abs() ** 2)
    ax.imshow((power.mean(dim=1) + 1e-6).log(), cmap="gray", aspect="auto")
    ax.contour(torch.fft.fftshift(mask).mean(dim=1), levels=[0.5], colors="r")
    ax.set(title=title, xlabel="$k_x$", ylabel="$k_z$", xticks=[], yticks=[])
plt.tight_layout()
plt.show()

# %%
# Comparison
# ----------
#
# We compare the three reconstructions in the region of interest, interpolated to the full resolution.

volumes = [
    F.interpolate(
        vol[(..., *(slice(s.start // f, s.stop // f) for s in roi))].to(device),
        scale_factor=f,
        mode="trilinear",
    )
    for vol, f in ((x_wbp, 8), (x_icecream, 4))
]
dinv.utils.plot_ortho3D(
    volumes + [x_cryolithe],
    titles=["WBP (binned by 8)", "Icecream (binned by 4)", "CryoLithe"],
    figsize=(12, 5),
)

# %%
# :References:
#
# .. footbibliography::
