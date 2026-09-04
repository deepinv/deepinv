# Reconstruct real CT sinograms with the 2DeteCT benchmark
# ========================================================
# We demonstrate image reconstruction of acquired CT projection data in sparse-view, limited-angle
# and low-dose CT acquisition scenarios.
#
# The data is taken from the 2DeteCT benchmark :footcite:p:`kiss2025benchmarking` and dataset :footcite:p:`kiss20232detect`,
# which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image).
# The setup is matched exactly to :footcite:t:`kiss2025benchmarking`, such that
# you can compare DeepInverse image reconstruction methods with the values reported in :footcite:t:`kiss2025benchmarking`.
#
# .. note::
#   This example requires `astra`. Install it with instructions from `here <https://astra-toolbox.com>`_ using `pip`, `conda` or `conda-forge`, e.g. `pip install astra-toolbox`.
#   Note that `astra` only supports CUDA.
#
#   This example also requires `tifffile`. Install it with `pip install tifffile`.

import deepinv as dinv
import torch

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

# For sparse-view projection geometry, simply downsample angles:
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
# Load sparse-view sinograms, which are stored as `.tif`s from the 2DeteCT test set.
# We follow the preprocessing steps performed in `LION <https://github.com/CambridgeCIA/LION>`_,
# which include detector binning, flat/dark-field correction, and log transform (Beer-Lambert).
# We subsample 360 angles out of the total 3600 angles (i.e. 10x acceleration).
#
# .. tip::
#     We download a sample test dataset slice (ID 4531), originally hosted at `Zenodo <https://zenodo.org/records/8014874>`_ and rehosted on HuggingFace for the demo.
#     to demonstrate reconstructing a single sample. See below for processing the test dataset of multiple samples.

root = dinv.utils.get_cache_home() / "2DeteCT"

dinv.datasets.download_archive(
    dinv.utils.get_image_url("2DeteCT_slices_4001-5000_slice04531.zip"),
    root / "2DeteCT_slices_4001-5000_slice04531.zip",
    extract=True,
)

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

# %%
# Reconstruct with FBP and RAM
# ----------------------------
# The `A_dagger` method of :class:`deepinv.physics.TomographyWithAstra` uses an approximate pseudo-inverse when `fbp=True`.
# When computed on the full benchmark test set, the performance matches the values reported in :footcite:t:`kiss2025benchmarking`.
#
# RAM is a model not trained on any 2DeteCT data, so this eaxmple tests its generalisability.
#
# .. tip::
#     Tune the sigma and gain parameters to tune the denoising strength.

model = dinv.models.RAM(pretrained=True, device=device)


# Optional FBP wrapper to rescale FBP output by operator norm for quantitative comparisons
class FBPWrapper(dinv.models.Reconstructor):
    def forward(self, y, physics, **kwargs):
        return physics.A_dagger(y, fbp=True) * physics.operator_norm


fbp = FBPWrapper()

with torch.no_grad():
    x_fbp = fbp(y, physics)


# Optional model wrapper to scale input and output
class ModelWrapper(dinv.models.Reconstructor):
    def __init__(self, model, scaling):
        super().__init__()
        self.model = model
        self.scaling = scaling

    def forward(self, y, physics, **kwargs):
        return (
            self.model(y / self.scaling, physics) * self.scaling * physics.operator_norm
        )


model = ModelWrapper(model, scaling=x_fbp.max())

# use estimated noise params
physics.update(sigma=0.01 / model.scaling, gain=0.003 / model.scaling)

with torch.no_grad():
    x_ram = model(y, physics)

# Plot (rescale by FBP max to visualise intensities on same scale, and clip 0-1.)
dinv.utils.plot(
    {
        "Sparse-view sino": y / y.max(),
        "FBP": x_fbp / x_fbp.max(),
        "RAM": x_ram / x_fbp.max(),
    },
    rescale_mode="clip",
)

# %%
# For the full benchmark, use the dataset to load these measurements and process them using `deepinv.test`.
#
# .. tip::
#     For the demo, we do not download anymore data. For the official benchmark, download the full test set yourself by downloading and extracting
#     from `Zenodo <https://zenodo.org/records/8014874>`_ (and reference reconstructions `here <https://zenodo.org/records/8017624>`_).
#
# .. note::
#     Here, ground truth = a proprietary reconstruction using all angles. Since it's arbitrary scale, we use PSNR after standardizing to its scale.
#     Similarly, FBP and RAM have different visual intensities since `min_max` rescale mode is used for plotting.
#

dataset = dinv.datasets.DeteCTDataset(
    root, problem="sparse_view", n_angles=n_angles, slice_ids="test"
)

metric = dinv.metric.PSNR(max_pixel=None, norm_inputs="standardize")

dinv.test(
    model,
    torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(1))),
    physics,
    metrics=metric,
    device=device,
    plot_images=True,
    rescale_mode="min_max",
    no_learning_method="A_dagger",
)


# %%
# Limited-angle CT reconstruction
# -------------------------------
# The same 2DeteCT dataset can be used also for limited-angle CT reconstruction.
# The physics reuses all other parameters, except different angles: we take the first 30% of angles,
# defining a limited angle wedge.
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

physics.update(sigma=0.01 / model.scaling, gain=0.003 / model.scaling)

dataset = dinv.datasets.DeteCTDataset(
    root, problem="limited_angle", n_angles=n_angles, slice_ids="test"
)

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
x, y = x.to(device), y.to(device)
with torch.no_grad():
    x_fbp = fbp(y, physics)
    x_ram = model(y, physics)

dinv.utils.plot(
    {
        "All angles recon": x / x_fbp.max(),
        "Limited-angle sino": y / y.max(),
        "FBP": x_fbp / x_fbp.max(),
        "RAM": x_ram / x_fbp.max(),
    },
    rescale_mode="clip",
)

dinv.test(
    model,
    torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(1))),
    physics,
    metrics=metric,
    device=device,
    plot_images=True,
    rescale_mode="min_max",
    no_learning_method="A_dagger",
)


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
physics.update(sigma=0.01 / model.scaling, gain=0.1 / model.scaling)

dataset = dinv.datasets.DeteCTDataset(root, problem="low_dose", slice_ids="test")

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
x, y = x.to(device), y.to(device)
with torch.no_grad():
    x_fbp = fbp(y, physics)
    x_ram = model(y, physics)

dinv.utils.plot(
    {
        "All angles recon": x / x_fbp.max(),
        "Low-dose sino": y / y.max(),
        "FBP": x_fbp / x_fbp.max(),
        "RAM": x_ram / x_fbp.max(),
    },
    rescale_mode="clip",
)

dinv.test(
    model,
    torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(1))),
    physics,
    metrics=metric,
    device=device,
    plot_images=True,
    rescale_mode="min_max",
    no_learning_method="A_dagger",
)
