r"""
Self-supervised cryo-ET reconstruction with Icecream
====================================================

Train Icecream :footcite:p:`kishore2026icecream` to denoise a cryo-electron tomography (cryo-ET) tomogram and correct its missing wedge, without ground truth.

Tomograms reconstructed from tilt series in a limited angular range are very noisy and lack a *missing wedge* of Fourier coefficients, which elongates structures along the beam.
Icecream trains a 3D network :math:`f` on sub-tomograms of two tomograms :math:`y_i = Mx + n_i` reconstructed from independent halves of the dose, with :math:`M` the :class:`missing-wedge operator <deepinv.physics.MissingWedge>`.
The :class:`deepinv.loss.IcecreamLoss` combines:

* a Noise2Noise term in the measured wedge, for denoising;
* an equivariance term with :class:`cube symmetries <deepinv.transform.CubeSymmetry>`, which move reliable frequencies of the estimates into the missing wedge, where the network must recover them.

See the `original code <https://github.com/swing-research/Icecream>`_. The half tomograms are reconstructed from half-dose tilt series with :meth:`deepinv.physics.CryoET.fbp`.
The data is read from the folder `CRYOET_DATA_DIR` with `mrcfile <https://mrcfile.readthedocs.io>`_ (see :ref:`sphx_glr_auto_examples_physics_demo_cryo_et.py`). If unavailable, a synthetic phantom is used.
"""

# %%

# sphinx_gallery_tags = ["Tomography", "Microscopy", "Self-supervised"]

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import deepinv as dinv

torch.manual_seed(0)
device = dinv.utils.get_device()
# 3D grid sampling backward is not implemented on MPS
physics_device = torch.device("cpu") if device.type == "mps" else device

DATA_DIR = Path(os.environ.get("CRYOET_DATA_DIR", "tomo_001"))
PREFIX = "06022023_BrnoKrios_Arctis_xe_Position_70"
PIXEL_SIZE = 7.84  # Å

try:
    import mrcfile

    use_real_data = (DATA_DIR / f"tilt_series_{PREFIX}_split1.mrc").exists()
except ImportError:
    use_real_data = False


def read_mrc(name, factor, bin_depth=True):
    r"""Reads a volume or tilt series from a memory-mapped MRC file, binned by `factor`."""
    step = factor if bin_depth else 1
    with mrcfile.mmap(DATA_DIR / f"{name}.mrc", permissive=True) as mrc:
        slabs = [
            torch.from_numpy(mrc.data[i : i + step].astype("float32")).mean(0)
            for i in range(0, mrc.data.shape[0] - step + 1, step)
        ]
    return F.avg_pool2d(torch.stack(slabs), factor)


def make_phantom(shape, seed):
    r"""Synthetic phantom of membrane vesicles (Gaussian shells) and macromolecules (Gaussian blobs)."""
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack(
        torch.meshgrid(
            *[torch.arange(n, dtype=torch.float32) for n in shape], indexing="ij"
        ),
        -1,
    )
    vol = torch.zeros(shape)
    for i in range(310):
        dist = (grid - torch.rand(3, generator=g) * torch.tensor(shape)).norm(dim=-1)
        r = torch.rand(1, generator=g)
        if i < 10:
            vol += torch.exp(-((dist - (0.06 + 0.1 * r) * shape[-1]) ** 2) / 2)
        else:
            vol += 0.7 * torch.exp(-(dist**2) / (2 * (1 + r) ** 2))
    return vol


# %%
# Training configuration
# ----------------------
#
# The paper trains a bias-free 3D U-Net with 64 to 512 feature maps on :math:`72^3` sub-tomograms for 50,000 iterations.
# To keep this example short, we use binned data, a smaller network and sub-tomograms, and few iterations. Increase them for real applications.

bin_factor = 4
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

# %%
# Half tomograms
# --------------
#
# We reconstruct each half-dose tilt series (binned by 4, pixel size 31 Å), normalized by the mean intensity of each tilt, with the weighted back-projection, and standardize the tomograms.

standardize = lambda v: (v - v.mean()) / v.std()

if use_real_data:
    halves = [
        read_mrc(f"tilt_series_{PREFIX}_split{i}", bin_factor, bin_depth=False)
        for i in (1, 2)
    ]
    tilts = torch.stack([t / t.mean(dim=(-1, -2), keepdim=True) - 1 for t in halves])
    angles = torch.tensor(
        [float(a) for a in (DATA_DIR / f"angles_{PREFIX}.tlt").read_text().split()]
    )
    vol_shape = (512 // bin_factor, 1024 // bin_factor, 1024 // bin_factor)
    reference = standardize(read_mrc(f"vol_{PREFIX}_Icecream", bin_factor))
    reference_title = "Icecream (reference, full training)"
else:
    vol_shape = (48, 96, 96)
    angles = torch.arange(-60.0, 61.0, 3.0)
    phantom = make_phantom(vol_shape, seed=0)
    sim = dinv.physics.CryoET(
        img_size=vol_shape,
        angles=angles,
        pixel_size=PIXEL_SIZE * bin_factor,
        ctf=False,
        dose_per_tilt=1.5,
        background=1.0,
    )
    sim.noise_model = dinv.physics.PoissonNoise(
        gain=1 / sim.electrons_per_pixel, clip_positive=True
    )
    x0 = phantom[None, None] * 0.05 / sim.A(phantom[None, None]).std()
    tilts = torch.cat([sim(x0) - 1 for _ in range(2)], dim=1)[0]
    reference = standardize(phantom)
    reference_title = "Ground truth"

physics_cryoet = dinv.physics.CryoET(
    img_size=vol_shape,
    angles=angles,
    pixel_size=PIXEL_SIZE * bin_factor,
    ctf=False,
    dose_weighting=False,
    ice_thickness=0.0,
    device=physics_device,
)
tomograms = torch.cat(
    [standardize(physics_cryoet.fbp(t[None, None].to(physics_device))) for t in tilts],
    dim=1,
).to(device)

dinv.utils.plot_ortho3D(
    [tomograms[:, :1], tomograms[:, 1:], tomograms.mean(1, keepdim=True)],
    titles=["half tomogram 1", "half tomogram 2", "average (WBP)"],
    figsize=(12, 5),
)

# %%
# The missing wedge
# -----------------
#
# The power spectrum in the :math:`(k_x, k_z)` plane, perpendicular to the tilt axis, shows the missing wedge, whose boundary matches the :func:`mask <deepinv.physics.functional.missing_wedge_mask>` built from the tilt range (red).

angle_range = (angles.min().item(), angles.max().item())
mask = dinv.physics.functional.missing_wedge_mask(
    vol_shape, angle_range=angle_range, spherical_support=False
)


def plot_spectra(volumes, titles):
    r"""Plots the log power spectra of volumes averaged along the tilt axis, with the outline of the measured wedge."""
    fig, axs = plt.subplots(
        1, len(volumes), figsize=(4 * len(volumes), 2.6), squeeze=False
    )
    for ax, vol, title in zip(axs[0], volumes, titles, strict=True):
        power = torch.fft.fftshift(torch.fft.fftn(vol[0, 0].float().cpu()).abs() ** 2)
        ax.imshow((power.mean(dim=1) + 1e-6).log(), cmap="gray", aspect="auto")
        ax.contour(
            torch.fft.fftshift(mask).mean(dim=1), levels=[0.5], colors="r", linewidths=1
        )
        ax.set(title=title, xlabel="$k_x$", ylabel="$k_z$", xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


plot_spectra([tomograms[:, :1]], ["half tomogram"])

# %%
# Training
# --------
#
# We train :class:`deepinv.models.Icecream` with :class:`deepinv.loss.IcecreamLoss` on random pairs of sub-tomograms at the same location in the two halves, stacked along channels.


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


physics = dinv.physics.MissingWedge(
    img_size=(crop_size,) * 3, angle_range=angle_range, device=device
)
model = dinv.models.Icecream(channels_per_scale=channels, device=device)
loss = dinv.loss.IcecreamLoss(
    transform=dinv.transform.CubeSymmetry(rng=torch.Generator().manual_seed(0))
)
dataset = SubTomogramPairs(tomograms[0].cpu(), crop_size, n_iterations * batch_size)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=torch.optim.Adam(model.parameters(), lr=lr),
    train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size),
    losses=loss,
    epochs=1,
    online_measurements=False,
    metrics=None,
    save_path=None,
    plot_images=False,
    device=device,
)
model = trainer.train()

# %%
# Reconstruction
# --------------
#
# :meth:`deepinv.models.Icecream.reconstruct` applies the network on overlapping sub-tomograms :math:`v` as :math:`f(Mf(v))` and averages the two halves.

x_icecream = model.reconstruct(tomograms, physics, batch_size=batch_size)

volumes = [
    tomograms.mean(1, keepdim=True),
    x_icecream,
    reference[None, None].to(device),
]
dinv.utils.plot_ortho3D(
    volumes,
    titles=["WBP", f"Icecream ({n_iterations} iterations)", reference_title],
    figsize=(12, 5),
)
plot_spectra(volumes, ["WBP", "Icecream", "reference"])

# %%
# Icecream denoises the tomogram and starts filling the missing wedge. Longer training with the full-size network gets closer to the reference.
#
# :References:
#
# .. footbibliography::
