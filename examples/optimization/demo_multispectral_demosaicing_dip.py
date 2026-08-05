r"""
Multispectral demosaicing raw images with Deep Image Prior
==========================================================

This example reconstructs a full-resolution multispectral image from raw snapshot
mosaiced measurements with an untrained :class:`deepinv.models.DeepImagePrior` :footcite:p:`ulyanov2018deep`.

Snapshot multispectral cameras cover the sensor with a mosaic of spectral filters (multispectral filter array),
so that each pixel records only one band. Demosaicing recovers every band at the full sensor resolution.

We use raw oral-tissue data from the MODID dataset :footcite:p:`chand2024modid` acquired in-vivo for screening of oral diseases such as oral squamous cell carcinoma.
The data is unprocessed from an imec CMV2K-SSM4x4-VIS CMOS sensor covering 16 bands from 460 to 600 nm before any corrections.
The sensor uses a sequential 4x4 MSFA i.e. 16 bands.
"""

import torch
import deepinv as dinv

device = dinv.utils.get_device()

# %%
# Load and mosaic the raw data
# ----------------------------
# We take sample 2 from MODID, hosted on the DeepInverse HuggingFace repository.
# You can explore further samples yourself from the `original data source <https://datadryad.org/dataset/doi:10.5061/dryad.nvx0k6dxw>`_ (licensed CC0 1.0).
# The file is a raw stream which we reshape to the pixel array of shape 1088x2048 and crop a 768x768 region of interest.

H, W = 1088, 2048
TILE = 4
CROP = 768
r0 = ((H - CROP) // 2 // TILE) * TILE
c0 = ((W - CROP) // 2 // TILE) * TILE

raw = torch.frombuffer(
    bytearray(dinv.utils.load_url(dinv.utils.get_image_url("modid.raw")).getvalue()),
    dtype=torch.float32,
).reshape(H, W)
raw = raw[r0 : r0 + CROP, c0 : c0 + CROP].to(device)  # 1, H, W where H=W=768

# %%
# Unravel the mosaic into a 16-band image of shape (1, 16, 768, 768) and build the binary MSFA mask, which is 1 where a band is sampled and 0 elsewhere.
# Demosaicing is simply inpainting under this mask (see :class:`deepinv.physics.Demosaicing`).

y = torch.zeros(1, 16, *raw.shape[-2:], device=device)
mask = torch.zeros_like(y)
for i in range(4):
    for j in range(4):
        c = i * 4 + j
        y[:, c, i::4, j::4] = raw[i::4, j::4]
        mask[:, c, i::4, j::4] = 1

y = y / y.max()  # normalise

physics = dinv.physics.Inpainting(img_size=mask.shape[1:], mask=mask, device=device)

# %%
# Reconstruct with Deep Image Prior
# ---------------------------------
# The Deep Image Prior fits an untrained network to the single measurement, using the network
# structure itself as the prior. Here the generator is :class:`deepinv.models.ConvDecoder`
# :footcite:p:`darestani2021accelerated`. A DIP was previously applied to demosaicing by :footcite:t:`park2020joint`.
#
# .. note::
#   We run only 100 iterations, which gives poor reconstructions. Increase it, for example to
#   10000, for better results. The DIP tends to overfit the measurement, so monitor the error.
#
# .. note::
#   You can try out various types of reconstruction algorithms listed in the :ref:`user guide <reconstructors>`. For example, instead of training a DIP from scratch, one can
#   fine-tune a foundation model such as :class:`deepinv.models.RAM` as done in :footcite:t:`wang2026perspective`.

model = dinv.models.DeepImagePrior(
    dinv.models.ConvDecoder(img_size=mask.shape[1:]).to(device),
    img_size=(256, 4, 4),
    verbose=True,
    re_init=True,
    iterations=1000,
).to(device)

with torch.no_grad():
    x_net = model(y, physics)

# %%
# Compare with classical interpolation
# ------------------------------------
# As a classical baseline we fill the unsampled pixels of each band by Gaussian interpolation.


class GaussianInterpolation(dinv.models.Reconstructor):
    def forward(self, y, physics, **kwargs):
        f = dinv.physics.functional.gaussian_blur(sigma=(1.5, 1.5), device=y.device)
        return torch.where(
            physics.mask > 0,
            y,
            dinv.physics.functional.conv2d(y, f, padding="replicate")
            / (
                dinv.physics.functional.conv2d(physics.mask, f, padding="replicate")
                + 1e-8
            ),
        )


model_classic = GaussianInterpolation().to(device)
with torch.no_grad():
    x_classic = model_classic(y, physics)

# %%
# Visualise the results
# ---------------------
# We plot the three bands closest to RGB: the full reconstruction has 16 bands.
# Interpolation shows chromatic errors at bright spots, an artefact of interpolating across bands.
# Run the DIP for more iterations for better results!

dinv.utils.plot(
    {
        "Raw": y[:, [3, 11, 12]],
        "Interpolated": x_classic[:, [3, 11, 12]],
        "DIP": x_net[:, [3, 11, 12]],
    },
    plot_inset=True,
    inset_loc=(0.0, 0.6),
    extract_loc=(0.56, 0.13),
    figsize=(9, 3),
)

# %%
# :References:
#
# .. footbibliography::
