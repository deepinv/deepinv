r"""
Ultrasound despeckling with BM3D
================================

This example despeckles real clinical ultrasound B-mode images with the classical
BM3D denoiser :class:`deepinv.models.BM3D`.

We use B-mode images from the carotid dataset acquired in Speckle2Self :footcite:p:`li2025speckle2self`.

Ultrasound B-mode images are corrupted by speckle, a granular noise pattern that
arises from interference between backscattered echoes. Speckle lowers contrast and
hides fine tissue structure, which makes clinical interpretation harder.

Speckle arises from the beamforming inverse problem, but often in practice we only have access to
envelope-detected B-mode grayscale in the log domain. Therefore, here, we treat despeckling as a
standard image denoising problem.

This example requires the `ptwt` package, which can be installed with ``pip install ptwt``.
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
# The `sigma` argument sets the noise level used by the denoiser. We use a custom efficient implementation of BM3D
# :footcite:p:`dabov2007image`, that's faster particularly on GPU.
#
# .. note::
#   BM3D is a non-learned denoiser, and used as a comparison in :footcite:t:`li2025speckle2self`.
#
# .. note::
#   Fully-developed speckle is Gamma-distributed, so in log domain, the Gaussian is only an approximation for BM3D.
#   See the :ref:`denoisers user guide <denoisers>` for further classical and deep denoisers.
#   For example, you can train your own with :class:`deepinv.physics.FisherTippettNoise`, which models the noise more accurately.


model = dinv.models.BM3D(use_legacy=False, device=device)
with torch.no_grad():
    x_hat = model(y, sigma=0.3)

# %%
# Visualise the results
# ---------------------
# We compare the input image with the BM3D despeckled image, for a few examples.

dinv.utils.plot(
    {"Vendor bmode image": y, "BM3D despeckled": x_hat},
    figsize=(7, 10),
)

# %%
# :References:
#
# .. footbibliography::
