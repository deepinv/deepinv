r"""
Cryo-electron tomography
========================

Simulate cryo-electron tomography (cryo-ET) tilt series with :class:`deepinv.physics.CryoET`, reconstruct them with the weighted back-projection (WBP), and train Icecream :footcite:p:`kishore2026icecream` to denoise the reconstruction and correct its missing wedge without ground truth.

See :ref:`sphx_glr_auto_examples_physics_demo_cryo_et_real_data.py` for the reconstruction of experimental data.
"""

# %%

# sphinx_gallery_tags = ["Tomography", "Microscopy", "Self-supervised"]

import matplotlib.pyplot as plt

plt.ion()
import torch

import deepinv as dinv
from deepinv.physics.functional import ctf, dose_filter

device = dinv.utils.get_device()
# 3D grid sampling backward is not implemented on MPS
physics_device = torch.device("cpu") if device.type == "mps" else device
torch.manual_seed(0)
generator = torch.Generator(physics_device).manual_seed(0)

# %%
# Synthetic specimen
# ------------------
#
# The phantom contains membrane vesicles (Gaussian shells) and macromolecules (Gaussian blobs) in a :math:`64 \times 128 \times 128` volume with pixel size 8 Å.
# Volumes have shape `(B, C, D, H, W)`, with `D` the beam direction at zero tilt and `H` the tilt axis.

vol_shape = (64, 128, 128)
pixel_size = 8.0  # Å

g = torch.Generator().manual_seed(0)
grid = torch.stack(
    torch.meshgrid(
        *[torch.arange(n, dtype=torch.float32) for n in vol_shape], indexing="ij"
    ),
    -1,
)
x = torch.zeros(vol_shape)
for i in range(208):
    dist = (grid - torch.rand(3, generator=g) * torch.tensor(vol_shape)).norm(dim=-1)
    r = torch.rand(1, generator=g)
    if i < 8:
        x += torch.exp(-((dist - 10 - 15 * r) ** 2) / 2)
    else:
        x += 0.7 * torch.exp(-(dist**2) / (2 * (1 + 1.5 * r) ** 2))
x = x[None, None].to(physics_device)

dinv.utils.plot_ortho3D(x, titles="phantom", figsize=(5, 5))

# %%
# Tilt series with different parameters
# -------------------------------------
#
# The tilt series contains 41 tilts in :math:`[-60^\circ, 60^\circ]`. We scale the specimen to a projected contrast of 20%, higher than in typical data (about 5%), so that structures remain visible in this small noisy example.
# The :func:`CTF <deepinv.physics.functional.ctf>` oscillates faster for larger defocus, attenuating and inverting high frequencies, and the :func:`dose filter <deepinv.physics.functional.dose_filter>` attenuates high frequencies as the accumulated exposure increases.

angles = torch.arange(-60.0, 61.0, 3.0, device=physics_device)
common = dict(
    img_size=vol_shape, angles=angles, pixel_size=pixel_size, device=physics_device
)
ideal = dinv.physics.CryoET(
    ctf=False, dose_weighting=False, ice_thickness=0.0, **common
)
x = x * 0.2 / ideal.A(x).std()

k = torch.linspace(0, 1 / (2 * pixel_size), 500)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
for df in (2.0, 5.0):
    axs[0].plot(k, ctf(k, torch.zeros_like(k), defocus=df), label=f"defocus {df} µm")
for exposure in (0.0, 20.0, 60.0, 120.0):
    axs[1].plot(k, dose_filter(k, exposure), label=f"{exposure:.0f} e/Å²")
axs[0].set(xlabel="spatial frequency (1/Å)", title="CTF")
axs[1].set(xlabel="spatial frequency (1/Å)", title="Dose filter")
axs[0].legend()
axs[1].legend()
plt.tight_layout()
plt.show()

# %%
# We compare the tilt at :math:`-60^\circ`, acquired last in the dose-symmetric scheme, for:
#
# * an ideal projection;
# * the CTF for two defocus values;
# * dose weighting, with 3 e/Å² per tilt;
# * per-tilt deformations: a misalignment of the projections and a smooth beam-induced displacement of the volume growing with the exposure;
# * shot noise: with `background=1`, measurements are in units of the mean number of electrons per pixel, and :class:`deepinv.physics.PoissonNoise` with gain `1 / physics.electrons_per_pixel` models the counting statistics.

order = ideal.acquisition_order
progress = (order.float() / order.max())[:, None, None, None, None]
deformations = dict(
    projection_shift=2.0
    * torch.randn(len(angles), 2, generator=generator, device=physics_device),
    volume_displacement=progress
    * 3.0
    * torch.randn(3, 4, 4, 4, generator=generator, device=physics_device),
)

physics = dinv.physics.CryoET(defocus=5.0, background=1.0, **common)
physics.noise_model = dinv.physics.PoissonNoise(
    gain=1 / physics.electrons_per_pixel, clip_positive=True, rng=generator
)
print(f"Electrons per pixel and per tilt: {physics.electrons_per_pixel.item():.0f}")

tilts = {
    "ideal": ideal.A(x),
    "CTF (2 µm)": dinv.physics.CryoET(
        defocus=2.0, dose_weighting=False, ice_thickness=0.0, **common
    ).A(x),
    "CTF (5 µm)": dinv.physics.CryoET(
        defocus=5.0, dose_weighting=False, ice_thickness=0.0, **common
    ).A(x),
    "CTF + dose": dinv.physics.CryoET(defocus=5.0, **common).A(x),
    "CTF + dose + deformations": dinv.physics.CryoET(
        defocus=5.0, **deformations, **common
    ).A(x),
    "CTF + dose + noise": physics(x),
}
dinv.utils.plot(
    [y[:, :, 0] for y in tilts.values()],
    titles=list(tilts),
    suptitle=f"Tilt at {angles[0]:.0f}°",
    figsize=(18, 3.5),
)

# %%
# Weighted back-projection
# ------------------------
#
# :meth:`deepinv.physics.CryoET.fbp` phase-flips the tilts, ramp-filters them perpendicularly to the tilt axis, weights them by their angular spacing and back-projects them.
# The missing wedge of the limited angular range elongates structures along the beam (`D`), and noise dominates the reconstruction of noisy tilt series.

y = physics(x)
x_wbp = physics.fbp(physics.remove_background(y))
dinv.utils.plot_ortho3D(
    [x, physics.fbp(physics.A(x)), x_wbp],
    titles=["ground truth", "WBP (noiseless)", "WBP (noisy)"],
    figsize=(12, 5),
)

# %%
# Self-supervised reconstruction with Icecream
# --------------------------------------------
#
# Icecream trains a 3D network :math:`f` on sub-tomograms of two tomograms :math:`y_i = Mx + n_i` reconstructed from independent halves of the dose, with :math:`M` the :class:`missing-wedge operator <deepinv.physics.MissingWedge>`.
# The :class:`deepinv.loss.IcecreamLoss` combines a Noise2Noise term in the measured wedge, for denoising, and an equivariance term with :class:`cube symmetries <deepinv.transform.CubeSymmetry>`, which move reliable frequencies of the estimates into the missing wedge, where the network must recover them.
# See the `original code <https://github.com/swing-research/Icecream>`_.
#
# We simulate two tilt series with half the dose (1.5 e/Å² per tilt), without deformations, and reconstruct them with the WBP.

half = dinv.physics.CryoET(defocus=5.0, dose_per_tilt=1.5, background=1.0, **common)
half.noise_model = dinv.physics.PoissonNoise(
    gain=1 / half.electrons_per_pixel, clip_positive=True, rng=generator
)
standardize = lambda v: (v - v.mean()) / v.std()
tomograms = torch.cat(
    [standardize(half.fbp(half.remove_background(half(x)))) for _ in range(2)], dim=1
).to(device)

# %%
# We train :class:`deepinv.models.Icecream` on random pairs of :math:`32^3` sub-tomograms at the same location in the two halves, stacked along channels.
# The paper trains a larger network on larger sub-tomograms for 50,000 iterations. Increase `n_iterations` for better results.


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


crop_size, batch_size = 32, 4
n_iterations = 200 if device.type == "cuda" else 500

physics_mw = dinv.physics.MissingWedge(
    img_size=(crop_size,) * 3, angle_range=(-60.0, 60.0), device=device
)
model = dinv.models.Icecream(channels_per_scale=[16, 32, 64], device=device)
trainer = dinv.Trainer(
    model=model,
    physics=physics_mw,
    optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
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

# %%
# :meth:`deepinv.models.Icecream.reconstruct` applies the network on overlapping sub-tomograms and averages the two halves.
# Icecream removes the noise and reduces the elongation along the beam.

x_icecream = model.reconstruct(tomograms, physics_mw, batch_size=batch_size)
x_avg = tomograms.mean(1, keepdim=True)

for name, vol in (("WBP (average of halves)", x_avg), ("Icecream", x_icecream)):
    corr = torch.corrcoef(torch.stack([vol.flatten(), x.to(device).flatten()]))[0, 1]
    print(f"Correlation with the ground truth, {name}: {corr:.3f}")

dinv.utils.plot_ortho3D(
    [x_avg, x_icecream, x],
    titles=["WBP (average of halves)", "Icecream", "ground truth"],
    figsize=(12, 5),
)

# %%
# :References:
#
# .. footbibliography::
