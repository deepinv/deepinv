r"""
Ultrasound despeckling from B-mode images
=========================================

This example despeckles real clinical ultrasound B-mode images with Speckle2Self
:footcite:p:`li2025speckle2self`, a model pretrained without clean reference images.

We compare its performance against a classical BM3D :class:`deepinv.models.BM3D` baseline.

We use B-mode images from the carotid dataset acquired in Speckle2Self.

Ultrasound B-mode images are corrupted by speckle, a granular noise pattern that
arises from interference between backscattered echoes. Speckle lowers contrast and
hides fine tissue structure, which makes clinical interpretation harder.

Speckle arises from the beamforming inverse problem, but often in practice we only have access to
envelope-detected B-mode grayscale in the log domain. Therefore, here, we treat despeckling as a
standard image denoising problem.

This example requires the `ptwt` (for BM3D) and `zea` (for Speckle2Self) packages, which can be installed with
``pip install ptwt zea``.
"""

import deepinv as dinv
import torch

device = dinv.utils.get_device()

# %%
# Load the dataset
# ----------------
# We download the Speckle2Self test set of 104 carotid B-mode images at 512x512 resolution,
# acquired with a Clarius L7 portable scanner from two healthy volunteers, taken from the Clarius envelope collection API by :footcite:t:`li2025speckle2self`.
#
# .. note::
#   The data is envelope-detected B-mode grayscale, already in the log domain, where the multiplicative speckle of the linear domain can now be approximated by additive noise.
#
# .. note::
#   For this demo, we show despeckling on two slices on CPU or four on GPU.

url = "https://drive.usercontent.google.com/download?id=1rQxgyCzDuLao05tE8y5l8vw8YFrPjbzx&export=download"
y = dinv.io.load_np(dinv.io.load_url(url)).unsqueeze(1).to(device)  # (B, 1, H, W)

y = y[: 2 if dinv.utils.devices_equal(device, "cpu") else 4]

y = y / y.max()  # to allow a meaningful sigma

# %%
# Despeckle with BM3D
# -------------------
# As a classical baseline, we use a custom efficient implementation of BM3D
# :footcite:p:`dabov2007image`, faster particularly on GPU. The `sigma` argument sets the assumed noise level.
#
# .. note::
#   Fully-developed speckle is Gamma-distributed, so in log domain, the Gaussian is only an approximation for BM3D.
#   See the :ref:`denoisers user guide <denoisers>` for further classical and deep denoisers.
#   For example, you can train your own with :class:`deepinv.physics.FisherTippettNoise`, which models the noise more accurately.

model = dinv.models.BM3D(use_legacy=False, device=device)
with torch.no_grad():
    x_hat = model(y, sigma=0.3)

# %%
# Despeckle with Speckle2Self
# ---------------------------
# We run the pretrained in-vivo Speckle2Self model, which is available in the `zea` toolbox :footcite:p:`stevens2026zea`.

import os

os.environ["KERAS_BACKEND"] = "torch"
from zea.models.speckle2self import Speckle2Self

model = Speckle2Self.from_preset("hf://zeahub/speckle2self-invivo").to(device)

with torch.no_grad():
    x_net = model(y.moveaxis(1, -1)).moveaxis(-1, 1)  # zea expects channel last

# %%
# Visualise the results
# ---------------------
# We compare the input B-mode image with the BM3D and Speckle2Self reconstructions.

dinv.utils.plot(
    {"Vendor bmode": y, "BM3D": x_hat, "Speckle2Self": x_net},
    figsize=(7, 10),
)

# %%
# :References:
#
# .. footbibliography::
