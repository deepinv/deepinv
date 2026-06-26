r"""
Low-intensity STED fluorescence microscopy denoising
=====================================================

This example shows how to denoise low-intensity STED fluorescence microscopy
images of live-cell mitochondria using the pretrained foundation model
:class:`deepinv.models.RAM`. We load real Abberior STED microscopy data
from :footcite:t:`osunavargas2025denoising`, process it in batches, and
visualize the results both with :func:`deepinv.utils.plot` and with the
interactive 3D viewer :func:`deepinv.utils.plot_napari`.

Acquiring fluorescence microscopy images at low excitation laser power reduces
photobleaching and phototoxicity, which lets biologists image living samples for
longer, but it produces noisier images. Here we recover clean images from
acquisitions taken at 1.5 microwatt excitation, using images taken at 8 microwatt
as the ground truth.

This example requires `tifffile`, `rarfile` and `napari`. Install them with
``pip install tifffile rarfile "napari[all]"``.
"""

# %%
import deepinv as dinv
import torch

device = dinv.utils.get_device()

# %%
# Download the dataset
# --------------------
#
# We download the live-cell mitochondria STED dataset from
# `Zenodo <https://zenodo.org/records/14215838>`_. The data is distributed as a
# RAR archive, which :func:`deepinv.datasets.download_archive` extracts for us.
# Each acquisition is a pair of TIFF images: a low-intensity (1.5 microwatt)
# measurement and a high-intensity (8 microwatt) ground truth.

data_dir = dinv.utils.get_cache_home() / "datasets" / "sted"
archive_name = "live_cell_mitochondria_u2os_tom20_halotag7_dm_sir"

dinv.datasets.download_archive(
    f"https://zenodo.org/records/14215838/files/{archive_name}.rar?download=1",
    data_dir / f"{archive_name}.rar",
    extract=True,
)

# %%
# Build the dataset
# -----------------
#
# We use :class:`deepinv.datasets.ImageFolder` to pair the ground truth and
# low-intensity images. TIFF files are loaded with :func:`deepinv.utils.load_tiff`.
dataset = dinv.datasets.ImageFolder(
    data_dir / archive_name / "test_and_training_data_1",
    x_path="ground_truth_images/*.tif",
    y_path="low_intensity_images/*.tif",
    loader=lambda f: dinv.utils.load_tiff(f).squeeze(0).float(),
)

# %%
# Denoise with the RAM foundation model
# -------------------------------------
#
# :class:`deepinv.models.RAM` is a pretrained foundation model for image
# restoration. We process the data in batches. The `gain`
# argument sets the Poisson noise level used by the model.

model = dinv.models.RAM(device=device)

for x, y in torch.utils.data.DataLoader(
    dataset, batch_size=1
):  # adjust batch size as necessary
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        rescale = y.max()
        x_net = model(y / rescale, gain=0.15) * rescale

    break  # remove this to process the whole dataset

# %%
# Visualize the results
# ---------------------
#
# We compare the ground truth, the noisy low-intensity measurement, and the
# denoised reconstruction.

dinv.utils.plot(
    {
        "8 µW excitation": x,
        "1.5 µW excitation": y,
        "1.5 µW excitation\ndenoised": x_net,
    },
    figsize=(7, 7),
)

# %%
# Interactive viewer with napari
# ------------------------------
#
# Use :func:`deepinv.utils.plot_napari` to interactively visualise microscopy images/stacks/volumes
# with the `napari <https://napari.org>`_ viewer.
#
# Since it opens an interactive window, it requires a display and is not executed in this gallery.
# To reproduce the screenshot below, run:
#
# .. code-block:: python
#
#     dinv.utils.plot_napari(x, y, x_net, screenshot=False)
#
# Pass ``screenshot=True`` to take a static screenshot instead:
#
# .. image:: /_static/demo_microscopy_denoising_screenshot.png
#    :width: 600
#    :align: center
#    :alt: napari viewer showing the ground truth, noisy and denoised images side by side

# %%
# :References:
#
# .. footbibliography::
