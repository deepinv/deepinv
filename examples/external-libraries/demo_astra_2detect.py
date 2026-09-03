# Reconstruct real CT sinograms with the 2DeteCT benchmark
# ========================================================
# We demonstrate image reconstruction of acquired CT projection data in sparse-view, limited-angle
# and low-dose CT acquisition scenarios.
# The data is taken from the 2DeteCT benchmark and dataset TODO CITE https://www.aimsciences.org/article/doi/10.3934/ammc.2025001 https://www.nature.com/articles/s41597-023-02484-6
# which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI.
#
# .. note::
#   This example requires tifffile and astra. Install them with TODO

import deepinv as dinv
import torch
import astra
from pathlib import Path

device = dinv.utils.get_device()

# %% 
# Model acquisition physics
# -------------------------
# Construct Astra geometry for fan-beam CT using values from LION. https://github.com/CambridgeCIA/LION
# First construct object geometry (single-slice):
obj_geom = astra.create_vol_geom(1024, 1024, 1, -513, 511, -513, 511, -0.5, 0.5)

# Then construct CT projection geometry (= conebeam with one detector row):
det_pix = 2 * 0.0748 # binned detector pixel in mm
fov = det_pix * 956 * 431.019989 / 529.000488 # field-of-view width in mm
scale = 1024 / fov # rescale such that recon grid has unit voxels
sod, sdd, det_pix = 431.019989 * scale, 529.000488 * scale, det_pix * scale # source-origin, source-detector

angles = -torch.linspace(0, 2 * torch.pi, 3600 + 1)[:-1] + torch.pi

# For sparse-view projection geometry, simply downsample angles:
n_angles = 360
proj_geom = astra.create_proj_geom("cone", det_pix, det_pix, 1, 956, angles[::3600 // n_angles].numpy(), sod, sdd - sod)

physics = dinv.physics.TomographyWithAstra(object_geometry=obj_geom, projection_geometry=proj_geom, is_2d=True, normalize=True, device=device, noise_model=dinv.physics.PoissonGaussianNoise())

# %%
# Load projection data
# --------------------
# Load sparse-view sinograms, which are stored as `.tif`s.
# We subsample 360 angles out of the total 3600 angles (i.e. 10x acceleration).

root = Path("/lustre/fsn1/projects/rech/nyd/commun/ram_project/datasets/2DeteCT")# Path("/Volumes/E/ram-experiments/data/2DeteCT")
data_dir = root / "2DeteCT_slices4001-5000/slice04001/mode2"

sino = dinv.io.load_tiff(data_dir / "sinogram.tif")[:, :, :-1] # (1, 1, 3600, 1912)
dark = dinv.io.load_tiff(data_dir / "dark.tif") # (1, 1, 1, 1912)
flat = 0.5 * (dinv.io.load_tiff(data_dir / "flat1.tif") + dinv.io.load_tiff(data_dir / "flat2.tif"))

# sum adjacent binned detector pixels
sino = sino[..., 0::2] + sino[..., 1::2] # (1, 1, 3600, 956)
dark = dark[..., 0::2] + dark[..., 1::2] # (1, 1, 1, 956)
flat = flat[..., 0::2] + flat[..., 1::2]

# Detector corrections:
sino = (sino - dark) / (flat - dark) # flat/dark-field correction
sino = -sino.clip(min=1e-6).log() # Beer-Lambert
sino = sino.flip(dims=(-1,))

y = sino[:, :, ::3600 // n_angles].float().contiguous().to(device) # (1, 1, n_angles, 956)

# %%
# Reconstruct with FBP and RAM
# ----------------------------
model = dinv.models.RAM(pretrained=False, device=device)
model.load_state_dict(torch.load("/lustre/fsn1/projects/rech/nyd/commun/ram_project/models/ram.pth.tar", map_location=device, weights_only=True), strict=False)

# FBP wrapper to rescale FBP output by operator norm for quantitative comparisons
class FBPWrapper(dinv.models.Reconstructor):
    def forward(self, y, physics, **kwargs):
        return physics.A_dagger(y, fbp=True) * physics.operator_norm

fbp = FBPWrapper()

with torch.no_grad():
    x_fbp = fbp(y, physics)

class ModelWrapper(dinv.models.Reconstructor):
    def __init__(self, model, scaling):
        super().__init__()
        self.model = model
        self.scaling = scaling

    def forward(self, y, physics, **kwargs):
        return model(y / self.scaling, physics) * self.scaling * physics.operator_norm

# Wrap model to scale input and output
model = ModelWrapper(model, scaling=x_fbp.max())
    
physics.update(sigma=0.001 / model.scaling, gain=0.001 / model.scaling) # use estimated noise params

with torch.no_grad():
    x_ram = model(y, physics)

# Plot (rescale by FBP max to visualise intensities on same scale, and clip 0-1.)
dinv.utils.plot({"Sparse-view sinogram": y / y.max(), "FBP": x_fbp / x_fbp.max(), "RAM": x_ram / x_fbp.max()}, rescale_mode="clip", save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp.png")

# %%
# For the full benchmark, use the dataset to load these measurements.
# Note ground truth here = their proprietary reconstruction with all angles.
# Note: for the full benchmark, make sure all test slices are in the folder. For the purposes of the demo, only a few slices are shown.
# Note: FBP and RAM have different visual intensities since min_max rescale mode is used for plotting.
dataset = dinv.datasets.DeteCTDataset(root, problem="sparse_view", n_angles=n_angles, slice_ids='test')

dinv.test(model, torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(2))), physics, metrics=[dinv.metric.PSNR(max_pixel=None)], device=device, plot_images=True, rescale_mode='min_max', no_learning_method="A_dagger")




# %% 
# Limited-angle CT reconstruction
# -------------------------------
# The physics reuses all other parameters, except different angles: we take the first 30% of angles,
# defining a limited angle wedge.
# Like before, we'll show how to reconstruct a single acquisition vs. test a full dataset.
#
n_angles = 1200
proj_geom = astra.create_proj_geom("cone", det_pix, det_pix, 1, 956, angles[:n_angles].numpy(), sod, sdd - sod)
physics = dinv.physics.TomographyWithAstra(object_geometry=obj_geom, projection_geometry=proj_geom, is_2d=True, normalize=False, device=device, noise_model=dinv.physics.PoissonGaussianNoise())
physics.update(sigma=0.001 / model.scaling, gain=0.001 / model.scaling) # use estimated noise params

dataset = dinv.datasets.DeteCTDataset(root, problem="limited_angle", n_angles=n_angles, slice_ids='test')

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
with torch.no_grad():
    x_fbp = fbp(y, physics)
    x_ram = model(y, physics)

dinv.utils.plot({"All angles recon": x / x_fbp.max(), "Limited-angle sino": y / y.max(), "FBP": x_fbp / x_fbp.max(), "RAM": x_ram / x_fbp.max()}, rescale_mode="clip", save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp1.png")


dinv.test(model, torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(2))), physics, metrics=[dinv.metric.PSNR(max_pixel=None)], device=device, plot_images=True, rescale_mode='min_max', no_learning_method="A_dagger")




# %%
# Low-dose CT reconstruction
# --------------------------
# This corresponds to a different acquisition at a lower dose (3W instead of 90W).
#

proj_geom = astra.create_proj_geom("cone", det_pix, det_pix, 1, 956, angles.numpy(), sod, sdd - sod)
physics = dinv.physics.TomographyWithAstra(object_geometry=obj_geom, projection_geometry=proj_geom, is_2d=True, normalize=False, device=device, noise_model=dinv.physics.PoissonGaussianNoise(sigma=27, gain=0.001))
physics.update(sigma=0.001 / model.scaling, gain=0.05 / model.scaling) # use estimated higher noise params

dataset = dinv.datasets.DeteCTDataset(root, problem="low_dose", slice_ids='test')

x, y = next(iter(torch.utils.data.DataLoader(dataset)))
with torch.no_grad():
    x_fbp = fbp(y, physics)
    x_ram = model(y, physics)

dinv.utils.plot({"All angles recon": x / x_fbp.max(), "Low-dose sino": y / y.max(), "FBP": x_fbp / x_fbp.max(), "RAM": x_ram / x_fbp.max()}, rescale_mode="clip", save_fn="/lustre/fswork/projects/rech/nyd/ubk23eb/Repos/ram-experiments/temp2.png")

dinv.test(model, torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(2))), physics, metrics=[dinv.metric.PSNR(max_pixel=None)], device=device, plot_images=True, rescale_mode='min_max', no_learning_method="A_dagger")
