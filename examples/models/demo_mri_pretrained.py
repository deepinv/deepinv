r"""
Reconstruct undersampled k-space for cardiac and brain MRI
==========================================================

This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:

* single-coil cardiac cine MRI from `CMRxRecon <https://cmrxrecon.github.io>`_ train set (:class:`deepinv.datasets.CMRxReconSliceDataset`) TODO CITE;
* 12-coil brain k-space from `Calgary-Campinas <https://sites.google.com/view/calgary-campinas-dataset/>`_ test set (:class:`deepinv.datasets.CalgarySliceDataset`) TODO CITE;
* 16-coil brain k-space from `FastMRI <https://fastmri.med.nyu.edu>`_ test set (:class:`deepinv.datasets.FastMRISliceDataset`) TODO CITE.

We demonstrate pretrained models:

* Joint-ICNet TODO CITE, pretrained on Calgary data, from DIRECT;
* vSHARP TODO CITE, pretrained on fastMRI brain, knee, prostate, and CMRxRecon cardiac data; https://huggingface.co/NKI-AI/direct-uniform https://openreview.net/forum?id=I13Y1nU6gs, from DIRECT;
* :class:`RAM <deepinv.models.RAM>` :footcite:t:`terris2025reconstruct`, pretrained on natural images, abdominal CT and knee MRI.

.. note::
    This example requires `DIRECT <https://docs.aiforoncology.nl/direct/>`_ (Netherlands Cancer Institute) and Python >=3.12. Install with `pip install deepinv[direct]`.

"""

# %%
import torch
import deepinv as dinv
from torch.utils.data import DataLoader

device = dinv.utils.get_device()
metric = dinv.metric.SharpnessIndex()

# %%
# Cardiac MRI reconstruction
# --------------------------
#
# Use a sample cardiac cine volume from :class:`deepinv.datasets.CMRxReconSliceDataset`, which loads ground-truth fully-sampled recon, undersampled y, and mask.
# Take the middle time-frame and a single slice for the demo, and construct a 2D :class:`deepinv.physics.MRI` physics.
#
# .. note::
#     For dynamic MRI reconstruction, use :class:`deepinv.physics.DynamicMRI` along with a model that can reconstruct temporal data.
#

dinv.datasets.download_archive(
    dinv.utils.get_image_url("CMRxRecon.zip"),
    dinv.utils.get_cache_home() / "CMRxRecon.zip",
    extract=True,
)

dataset = dinv.datasets.CMRxReconSliceDataset(dinv.utils.get_cache_home() / "CMRxRecon")

x, y, params = next(iter(DataLoader(dataset)))
x, y = x[:, :, x.shape[-2] // 2], y[:, :, y.shape[-2] // 2].to(device)
mask = params["mask"].squeeze(2).to(device)

dinv.utils.plot(
    {
        f"y of shape {tuple(y.shape)}": y,
        f"mask of acc {1 / mask.mean().item():.2f}": mask,
    },
    figsize=(6, 8),
    suptitle="CMRxRecon data",
)

physics = dinv.physics.MRI(img_size=mask.shape[-2:], mask=mask, device=device)

# %%
# Perform reconstruction with pretrained models.
# We use the vSHARP 2D model from DIRECT, and RAM from :footcite:t:`terris2025reconstruct`.
# We compare to the zero-filled reconstruction using the sharpness metric.

vsharp = dinv.models.DIRECTModel(
    model_name="vsharp_cardiac", pretrained=True, device=device
)
ram = dinv.models.RAM(device=device, pretrained=True)

with torch.no_grad():
    x_zf = physics.A_adjoint(y).cpu()
    x_vsharp = vsharp(y, physics).cpu()
    x_ram = ram(y / x_zf.max(), physics).cpu() * x_zf.max()

dinv.utils.plot(
    {"Fully-sampled": x, "Zero-filled": x_zf, "vSHARP": x_vsharp, "RAM": x_ram},
    subtitles=[
        f"{metric(x).item():.1f}",
        f"{metric(x_zf).item():.1f}",
        f"{metric(x_vsharp).item():.1f}",
        f"{metric(x_ram).item():.1f}",
    ],
)

# %%
# Multicoil brain MRI reconstruction
# ----------------------------------
#
# We use a 5x accelerated multicoil Calgary-Campinas brain test volume with no ground truth,
# with :class:`deepinv.datasets.CalgarySliceDataset`, which loads undersampled y, mask (Poisson-disk), and estimated coil maps using ESPIRiT.
# Take a single slice for the demo, and construct a 2D :class:`deepinv.physics.MultiCoilMRI` physics.

dinv.datasets.download_archive(
    dinv.utils.get_image_url("demo_calgary_test_e13991s3_P01536.7.h5"),
    dinv.utils.get_cache_home() / "calgary_test_5" / "e13991s3_P01536.7.h5",
)

dataset = dinv.datasets.CalgarySliceDataset(
    dinv.utils.get_cache_home() / "calgary_test_5",
    slice_index="middle",
    transform=dinv.datasets.CalgarySliceTransform(
        estimate_coil_maps=True, acs=24, espirit_crop=0.85
    ),
)

_, y, params = next(iter(DataLoader(dataset)))
y = y.to(device)

physics = dinv.physics.MultiCoilMRI(img_size=y.shape[-2:], **params, device=device)

dinv.utils.plot(
    {"Mask": physics.mask, "0th coil map": physics.coil_maps[:, [0]]}, figsize=(6, 8)
)

# %%
# Perform reconstruction with pretrained models.
# Note that vSHARP and Joint-ICNet (TODO CITE) estimate coil maps internally, whereas RAM uses the ESPIRiT maps.

vsharp = dinv.models.DIRECTModel(
    model_name="vsharp_brain", pretrained=True, device=device
)
jointicnet = dinv.models.DIRECTModel(
    model_name="jointicnet_5x", pretrained=True, device=device
)
ram = dinv.models.RAM(device=device, pretrained=True)

with torch.no_grad():
    x_zf = physics.A_adjoint(y).cpu()
    x_sense = physics.A_dagger(y).cpu()
    x_vsharp = vsharp(y, physics).cpu()
    x_jointicnet = jointicnet(y, physics).cpu()

    physics.phase_correct_maps(x_zf)
    x_ram = ram(y / x_zf.max(), physics).cpu() * x_zf.max()

dinv.utils.plot(
    {
        "Zero-filled": x_zf,
        "SENSE": x_sense,
        "vSHARP": x_vsharp,
        "Joint-ICNet": x_jointicnet,
        "RAM": x_ram,
    },
    subtitles=[
        f"{metric(x_zf).item():.1f}",
        f"{metric(x_sense).item():.1f}",
        f"{metric(x_vsharp).item():.1f}",
        f"{metric(x_jointicnet).item():.1f}",
        f"{metric(x_ram).item():.1f}",
    ],
)

# %%
# FastMRI brain test set
# ----------------------

# We use a volume from the :class:`deepinv.datasets.FastMRISliceDataset` test set where GT was provided by the organisers for computing metrics.

dinv.datasets.download_archive(
    dinv.utils.get_image_url(
        "demo_fastmri_brain_multicoil_test_file_brain_AXT2_200_2000341.h5"
    ),
    dinv.utils.get_cache_home()
    / "fastmri_brain_multicoil_test"
    / "file_brain_AXT2_200_2000341.h5",
)
dinv.datasets.download_archive(
    dinv.utils.get_image_url(
        "demo_fastmri_brain_multicoil_test_full_file_brain_AXT2_200_2000341.h5"
    ),
    dinv.utils.get_cache_home()
    / "fastmri_brain_multicoil_test_full"
    / "file_brain_AXT2_200_2000341.h5",
)

dataset = dinv.datasets.FastMRISliceDataset(
    dinv.utils.get_cache_home() / "fastmri_brain_multicoil_test",
    target_root=dinv.utils.get_cache_home() / "fastmri_brain_multicoil_test_full",
    slice_index="middle",
    transform=dinv.datasets.MRISliceTransform(
        estimate_coil_maps=True, espirit_crop=0.85
    ),
)

x, y, params = next(iter(DataLoader(dataset)))
y = y.to(device)

physics = dinv.physics.MultiCoilMRI(img_size=y.shape[-2:], **params, device=device)

dinv.utils.plot(
    {"Mask": physics.mask, "0th coil map": physics.coil_maps[:, [0]]}, figsize=(6, 8)
)


# %%
# Perform reconstruction with pretrained models.
# Note that vSHARP estimates coil maps internally, whereas RAM uses the ESPIRiT maps.

vsharp = dinv.models.DIRECTModel(
    model_name="vsharp_brain", pretrained=True, device=device
)
ram = dinv.models.RAM(device=device, pretrained=True)

with torch.no_grad():
    x_zf = physics.A_adjoint(y).cpu()
    x_sense = physics.A_dagger(y).cpu()
    x_vsharp = vsharp(y, physics).cpu()

    physics.phase_correct_maps(x_zf)
    x_ram = ram(y / x_zf.max(), physics).cpu() * x_zf.max()

# Crop to FastMRI FOV
x_zf = physics.crop(x_zf, shape=x.shape[-2:])
x_vsharp = physics.crop(x_vsharp, shape=x.shape[-2:])
x_ram = physics.crop(x_ram, shape=x.shape[-2:])

dinv.utils.plot(
    {
        "Fully-sampled": x,
        "Zero-filled": x_zf,
        "SENSE": x_sense,
        "vSHARP": x_vsharp,
        "RAM": x_ram,
    },
    subtitles=[
        f"{metric(x).item():.1f}",
        f"{metric(x_zf).item():.1f}",
        f"{metric(x_sense).item():.1f}",
        f"{metric(x_vsharp).item():.1f}",
        f"{metric(x_ram).item():.1f}",
    ],
)
