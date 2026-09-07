r"""
Reconstruct real CT sinograms with the 2DeteCT benchmark
========================================================
We demonstrate image reconstruction of acquired CT projection data in sparse-view, limited-angle
and low-dose CT acquisition scenarios.

The data is taken from the 2DeteCT benchmark :footcite:p:`kiss2025benchmarking` and dataset :footcite:p:`kiss20232detect`,
which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image).
The setup is matched exactly to :footcite:t:`kiss2025benchmarking`, such that
you can compare DeepInverse image reconstruction methods with the values reported in :footcite:t:`kiss2025benchmarking`.

.. note::
  This example requires `astra`. Install it with instructions from `their docs <https://astra-toolbox.com>`_ using `pip`, `conda` or `conda-forge`, e.g. `pip install astra-toolbox`.
  Note that `astra` only supports CUDA.

  This example also requires `tifffile`. Install it with `pip install tifffile`.
"""

import deepinv as dinv
import torch
from torch.utils.data import DataLoader, Subset
from matplotlib.colors import Normalize

try:
    import astra
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        "This example requires astra. Install it on a CUDA-compatible machine following https://astra-toolbox.com"
    )

device = dinv.utils.get_device()

if torch.device(device).type != "cuda":
    raise RuntimeError("The TomographyWithAstra operator only supports CUDA device.")

# %%
# Model acquisition physics
# -------------------------
# Construct Astra geometry for fan-beam CT using values from `LION <https://github.com/CambridgeCIA/LION>`_.
# First construct object geometry (single-slice):

obj_geom = astra.create_vol_geom(1024, 1024, 1, -513, 511, -513, 511, -0.5, 0.5)

# %%
# Then construct CT projection geometry (= conebeam with one detector row):

det_pix = 2 * 0.0748  # binned detector pixel in mm
fov = det_pix * 956 * 431.019989 / 529.000488  # field-of-view width in mm
scale = 1024 / fov  # rescale such that recon grid has unit voxels
sod = 431.019989 * scale  # source-origin distance
sdd = 529.000488 * scale  # source-detector distance
det_pix *= scale

angles = -torch.linspace(0, 2 * torch.pi, 3600 + 1)[:-1] + torch.pi

# %%
# For sparse-view projection geometry, simply downsample angles. Here, we use 360 angles i.e. 10x acceleration;
# you can decrease the number of angles to make the problem more challenging.

n_angles = 360
proj_geom = astra.create_proj_geom(
    "cone",
    det_pix,
    det_pix,
    1,
    956,
    angles[:: 3600 // n_angles].numpy(),
    sod,
    sdd - sod,
)

# %%
# Finally, we use :class:`deepinv.physics.TomographyWithAstra` to instantiate the forward/backward projectors:
#
# .. tip::
#     You can also use :func:`deepinv.datasets.DeteCTDataset.get_astra_geometry` to get `obj_geom, proj_geom` directly, passing `problem='sparse_view', n_angles=n_angles`.

physics = dinv.physics.TomographyWithAstra(
    object_geometry=obj_geom,
    projection_geometry=proj_geom,
    is_2d=True,
    normalize=True,
    device=device,
    noise_model=dinv.physics.PoissonGaussianNoise(),
)

# %%
# Load projection data
# --------------------
# Load sparse-view sinograms, which are stored as `tif` files from the 2DeteCT test set.
# We follow the preprocessing steps performed in `LION <https://github.com/CambridgeCIA/LION>`_,
# which include detector binning, flat/dark-field correction, and log transform (Beer-Lambert).
# We subsample 360 angles out of the total 3600 angles (i.e. 10x acceleration).
#
# .. tip::
#     We download a sample test dataset slice (ID 4531), originally hosted at `Zenodo <https://zenodo.org/records/8014874>`_ and rehosted on HuggingFace for the demo.
#     to demonstrate reconstructing a single sample. See below for processing the test dataset of multiple samples.

# root = dinv.utils.get_cache_home() / "2DeteCT"
from pathlib import Path
root = Path("/lustre/fsn1/projects/rech/nyd/commun/ram_project/datasets/2DeteCT")# Path("/Volumes/E/ram-experiments/data/2DeteCT")

# dinv.datasets.download_archive(
#     dinv.utils.get_image_url("2DeteCT_slices_4001-5000_slice04531.zip"),
#     root / "2DeteCT_slices_4001-5000_slice04531.zip",
#     extract=True,
# )

data_dir = root / "2DeteCT_slices4001-5000/slice04531/mode2"

sino = dinv.io.load_tiff(data_dir / "sinogram.tif")[:, :, :-1]  # (1, 1, 3600, 1912)
dark = dinv.io.load_tiff(data_dir / "dark.tif")  # (1, 1, 1, 1912)
flat = 0.5 * (
    dinv.io.load_tiff(data_dir / "flat1.tif")
    + dinv.io.load_tiff(data_dir / "flat2.tif")
)

# sum adjacent binned detector pixels
sino = sino[..., 0::2] + sino[..., 1::2]  # (1, 1, 3600, 956)
dark = dark[..., 0::2] + dark[..., 1::2]  # (1, 1, 1, 956)
flat = flat[..., 0::2] + flat[..., 1::2]

# Detector corrections:
sino = (sino - dark) / (flat - dark)  # flat/dark-field correction
sino = -sino.clip(min=1e-6).log()  # Beer-Lambert
sino = sino.flip(dims=(-1,))  # flip detector

# Processed projections of shape (1, 1, n_angles, 956)
y = sino[:, :, :: 3600 // n_angles].float().contiguous().to(device)

dinv.utils.plot({"Sparse-view sino": y}, subtitles=[f"Shape: {tuple(y.shape)}"])

# %%
# Reconstruct with FBP and RAM
# ----------------------------
# The `A_dagger` method of :class:`deepinv.physics.TomographyWithAstra` uses an approximate pseudo-inverse when `fbp=True`.
# When computed on the full benchmark test set, the performance matches the FBP values reported in :footcite:t:`kiss2025benchmarking`.
#
# .. note::
#     The FBP below is computed with the normalised operator, so we divide by `physics.operator_norm` to obtain quantitative output.
#

with torch.no_grad():
    x_fbp = physics.A_dagger(y / physics.operator_norm, fbp=True)

# %%
# RAM is a model not trained on any 2DeteCT data, so this eaxmple tests its generalisability.
#
# .. tip::
#     Tune the sigma and gain parameters to tune the denoising strength.
#
# Since RAM expects the reconstruction to be in [0, 1], we rescale the model input by some reference value.
# Similarly, for quantitative comparions, divide the output by `physics.operator_norm`.

# model = dinv.models.RAM(pretrained=True, device=device)
model = dinv.models.RAM(pretrained=False, device=device)
model.load_state_dict(torch.load("/lustre/fsn1/projects/rech/nyd/commun/ram_project/models/ram.pth.tar", map_location=device, weights_only=True), strict=False)

# use estimated noise params
physics.update(sigma=0.01 / physics.operator_norm, gain=0.003 / physics.operator_norm)

with torch.no_grad():
    x_ram = model(y / physics.operator_norm, physics)

# %%
# Plot. First rescale by FBP max , and clip 0-1, such that image intensities are visualised on same scale.
dinv.utils.plot(
    {
        "FBP": x_fbp,
        "RAM": x_ram,
    },
    rescale_mode=None,
    figsize=(12,3),
    save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp0.png",
    vmax=x_fbp.max() * 0.4,
    norm=Normalize(vmax=x_fbp.max() * 0.4),
)


# %%
# The :class:`deepinv.datasets.DeteCTDataset` class can be used to load `x, y`, where `y` are the already-preprocessed sinograms as above,
# and `x` is a ground truth i.e. a proprietary reconstruction using all angles. Since `x` is on an arbitrary scale, we use PSNR after standardizing to its scale.
# The dataset finds all samples in the local directory matching the test slice IDs. Here, since only downloaded one slice, it uses the same as above.
#

dataset = dinv.datasets.DeteCTDataset(
    root, problem="sparse_view", n_angles=n_angles, slice_ids="test"
)

metric = dinv.metric.PSNR(max_pixel=None)

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
x, y = x.to(device), y.to(device)
with torch.no_grad():
    x_fbp = physics.A_dagger(y / physics.operator_norm, fbp=True)
    x_ram = model(y / physics.operator_norm, physics)

print(x.max(), x_fbp.max(), x_ram.max())

dinv.utils.plot(
    {
        "All angles recon": x,
        "FBP": x_fbp,
        "RAM": x_ram,
    },
    subtitles=["",
        f"PSNR: {metric(x_fbp, x).item():.2f}",
        f"PSNR: {metric(x_ram, x).item():.2f}"
    ],
    rescale_mode=None,
    figsize=(12,3),
    save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp.png",
    vmax=x_fbp.max() * 0.4,
    norm=Normalize(vmax=x_fbp.max() * 0.4),
)

# %%
# Use the full benchmark
# ----------------------
# For the full benchmark, use :class:`deepinv.datasets.DeteCTDataset` and process them using :meth:`deepinv.Trainer.test`.
# This tests the algorithm on all samples of the test set, and the PSNR and SSIM results should be comparable to those reported in the benchmark in :footcite:t:`kiss2025benchmarking`.
#
# .. tip::
#     For the demo, we do not download anymore data. For the official benchmark, download the full test set yourself by downloading and extracting
#     from `Zenodo <https://zenodo.org/records/8014874>`_ (and reference reconstructions `here <https://zenodo.org/records/8017624>`_).
#
# .. note::
#     :meth:`deepinv.Trainer.test` does not do any manual rescaling, so we use `min_max` rescale mode for plotting. Therefore, "no learning recon" and RAM appear with different visual intensities.
#
# .. note::
#     The no learning reconstruction compared here is the least-squares using conjugate gradient, which performs better than FBP, which is merely a fast approximation.
#

class ScaledTrainer(dinv.Trainer):
    def get_samples(self, iterators, g):
        x, y, physics = super().get_samples(iterators, g)
        return x, y / physics.operator_norm, physics

ScaledTrainer(
    model,
    physics,
    metrics=metric,
    optimizer=None,
    train_dataloader=None,
    device=device,
    plot_images=True,
    rescale_mode="min_max",
    no_learning_method="A_dagger",
).test(DataLoader(Subset(dataset, range(1))))


# %%
# Limited-angle CT reconstruction
# -------------------------------
# The same 2DeteCT dataset can be used also for limited-angle CT reconstruction.
# The physics reuses all other parameters, except different angles: we take here the first 1200 angles,
# defining a limited angle (120 degrees) wedge. You can decrease the number of angles to define smaller wedges,
# which makes the problem more challenging.
#
# Like before, we'll show how to reconstruct a single acquisition vs. test a full dataset.
#
n_angles = 1200
proj_geom = astra.create_proj_geom(
    "cone", det_pix, det_pix, 1, 956, angles[:n_angles].numpy(), sod, sdd - sod
)

physics = dinv.physics.TomographyWithAstra(
    object_geometry=obj_geom,
    projection_geometry=proj_geom,
    is_2d=True,
    normalize=True,
    device=device,
    noise_model=dinv.physics.PoissonGaussianNoise(),
)

physics.update(sigma=0.01 / physics.operator_norm, gain=0.003 / physics.operator_norm)

dataset = dinv.datasets.DeteCTDataset(
    root, problem="limited_angle", n_angles=n_angles, slice_ids="test"
)

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
x, y = x.to(device), y.to(device)
with torch.no_grad():
    x_fbp = physics.A_dagger(y / physics.operator_norm, fbp=True)
    x_ram = model(y / physics.operator_norm, physics)

dinv.utils.plot(
    {
        "All angles recon": x,
        "FBP": x_fbp,
        "RAM": x_ram,
    },
    subtitles=["",
        f"PSNR: {metric(x_fbp, x).item():.2f}",
        f"PSNR: {metric(x_ram, x).item():.2f}"
    ],
    rescale_mode=None,
    figsize=(12,3),
    save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp1.png",
    vmax=x_fbp.max() * 0.4,
    norm=Normalize(vmax=x_fbp.max() * 0.4),
)

# %%
# Note that, similar above, you can also use :meth:`deepinv.Trainer.test` to test the model on the full test dataset.
# The results on the test set should then be comparable to those reported in the benchmark in :footcite:t:`kiss2025benchmarking`.

# %%
# Low-dose CT reconstruction
# --------------------------
# Each sample in 2DeteCT is scanned 3 times: `mode2` was used above, and `mode1` corresponds to a low-dose
# acquisition (3W instead of 90W, i.e. 30x lower dose).
# `mode3` is a beam-hardened acquisition, you can also try on this.
#
# .. tip::
#     Tune the sigma and gain parameters to tune the denoising strength.
#

proj_geom = astra.create_proj_geom(
    "cone", det_pix, det_pix, 1, 956, angles.numpy(), sod, sdd - sod
)

physics = dinv.physics.TomographyWithAstra(
    object_geometry=obj_geom,
    projection_geometry=proj_geom,
    is_2d=True,
    normalize=True,
    device=device,
    noise_model=dinv.physics.PoissonGaussianNoise(),
)

# use estimated higher noise params
physics.update(sigma=0.01 / physics.operator_norm, gain=0.1 / physics.operator_norm)

dataset = dinv.datasets.DeteCTDataset(root, problem="low_dose", slice_ids="test")

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
x, y = x.to(device), y.to(device)
with torch.no_grad():
    x_fbp = physics.A_dagger(y / physics.operator_norm, fbp=True)
    x_ram = model(y / physics.operator_norm, physics)

dinv.utils.plot(
    {
        "All angles recon": x,
        "FBP": x_fbp,
        "RAM": x_ram,
    },
    subtitles=["",
        f"PSNR: {metric(x_fbp, x).item():.2f}",
        f"PSNR: {metric(x_ram, x).item():.2f}"
    ],
    rescale_mode=None,
    figsize=(12,3),
    save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp2.png",
    vmax=x_fbp.max() * 0.4,
    norm=Normalize(vmax=x_fbp.max() * 0.4),
)
# %%
# Similarly you can also use :meth:`deepinv.Trainer.test` to test the model on the full low-dose test dataset.
# The results on the test set should then be comparable to those reported in the benchmark in :footcite:t:`kiss2025benchmarking`.

# %%
# :References:
#
# .. footbibliography::
