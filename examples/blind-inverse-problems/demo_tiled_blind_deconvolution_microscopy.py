r"""
Tiled blind deconvolution of a fluorescence microscopy image
============================================================

This example deblurs a real widefield fluorescence image whose blur varies across the field of view.
We estimate one point spread function (PSF) per tile with the PSF identification network :class:`deepinv.models.DeepBlur`.
These PSFs define the space-varying blur operator :class:`deepinv.physics.TiledSpaceVaryingBlur`

.. math::

    y = \sum_{k=1}^K h_k \star (m_k \odot x),

where :math:`h_k` is the PSF of tile :math:`k` and :math:`m_k` are masks blending the overlapping tiles.
We then reconstruct the image with :class:`deepinv.models.RAM` :footcite:p:`terris2025reconstruct`, fine-tuned for fluorescence microscopy.

Unlike :ref:`sphx_glr_auto_examples_blind-inverse-problems_demo_blind_deblurring.py`, where a network directly estimates a space-varying blur,
DeepBlur assumes the blur is invariant within a patch, and the tiling makes the estimate space-varying.
"""

# sphinx_gallery_tags = ["Deblurring", "Microscopy"]
# sphinx_gallery_thumbnail_number = 2

import torch
from torchvision.utils import make_grid
import plotly.express as px
import deepinv as dinv

device = dinv.utils.get_device()

# %%
# Load the blurry image
# ~~~~~~~~~~~~~~~~~~~~~
#
# The image shows F-actin in bovine pulmonary artery endothelial cells, imaged with a widefield microscope and a NA 1.49 objective.
# The slide is tilted with respect to the focal plane, so the defocus, and hence the PSF, changes across the field of view:
# the left of the image is in focus, the bottom right is out of focus.

y = dinv.utils.load_example("bpaec_tilted_widefield.tif", device=device)

dinv.utils.plot({"Blurry image": y})

# %%
# Estimate the PSFs on tiles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :meth:`deepinv.models.DeepBlur.forward_tiled` splits the image into overlapping tiles and estimates one :math:`71 \times 71` PSF per tile.
# The network is trained on patches of 128 to 256 pixels: smaller tiles follow faster blur variations, larger tiles give more context.
# Each tile is min-max normalized to :math:`[0, 1]`, like the patches DeepBlur was trained on, so the intensity range of the image does not matter.
# We plot the center of each PSF (square root, for visibility) at the position of its tile:
# the PSFs widen towards the out-of-focus region.

patch_size, stride = 128, 64

model = dinv.models.DeepBlur(device=device)
with torch.no_grad():
    filters = model.forward_tiled(y, patch_size=patch_size, stride=stride)

n_rows, n_cols = dinv.physics.TiledSpaceVaryingBlur.num_filters(
    y.shape[-2:], patch_size, stride
)
psf_grid = make_grid(
    filters[0, 0, :, None, 20:-20, 20:-20].sqrt(),
    nrow=n_cols,
    normalize=True,
    scale_each=True,
)
dinv.utils.plot({"Estimated PSFs": psf_grid[None, :1]}, cmap="magma", figsize=(6, 6))

# %%
# Build the space-varying blur operator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :class:`deepinv.physics.TiledSpaceVaryingBlur` uses valid convolutions: the image :math:`x` is ``psf_size - 1`` pixels larger than :math:`y`.
# Tiles of :math:`x` of size ``patch_size + psf_size - 1`` with the same stride are blurred into the tiles of :math:`y` on which the PSFs were estimated.

physics = dinv.physics.TiledSpaceVaryingBlur(
    filters=filters,
    patch_size=patch_size + model.psf_size - 1,
    stride=stride,
    device=device,
)

# %%
# Deconvolve with RAM
# ~~~~~~~~~~~~~~~~~~~
#
# We load RAM weights fine-tuned on grayscale, fluorescence-like images with simulated optical blur and Poisson-Gaussian noise.
# As the physics has no noise model, RAM assumes a low noise level. For noisier images, pass the noise levels with the ``sigma`` and ``gain`` arguments.
# Finally, we crop the reconstruction to the field of view of :math:`y`.
#
# Zoom in either image of the interactive figure below (made with `plotly <https://plotly.com/python/>`_) to compare them: the views are linked.
# Each image is rescaled between its 0.1% and 99.8% percentiles, so that a few bright pixels do not compress the display range,
# and the slider brightens dim regions.

ram = dinv.models.RAM(pretrained=False, device=device)
ram.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://mambo-utoulouse.fr/weights/RAM/checkpoint.pth",
        map_location=device,
        file_name="ram_deepblur_gray.pth",
        weights_only=True,
    )
)

crop = model.psf_size // 2
with torch.no_grad():
    x_hat = ram(y, physics)[..., crop:-crop, crop:-crop]


def interactive_plot(images: dict[str, torch.Tensor], height: int = 430):
    r"""Plot images side by side with linked zoom, a magma colormap and a brightness slider."""
    x = torch.cat(list(images.values()))[:, 0].cpu()
    low, high = x.flatten(1).quantile(torch.tensor([0.001, 0.998]), dim=1)
    x = ((x - low[:, None, None]) / (high - low)[:, None, None]).clamp(0, 1)
    fig = px.imshow(
        (255 * x).to(torch.uint8).numpy(),  # 8 bits keep the page light
        facet_col=0,
        color_continuous_scale="magma",
        height=height,
    )
    steps = [
        dict(label=str(g), method="relayout", args=[{"coloraxis.cmax": 255 / g}])
        for g in (1, 1.5, 2, 3)
    ]
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=30, b=0),
        sliders=[dict(currentvalue=dict(prefix="Brightness: x"), steps=steps)],
    )
    for annotation, title in zip(fig.layout.annotations, images, strict=True):
        annotation.text = title
    return fig


interactive_plot({"Blurry image": y, "DeepBlur + RAM": x_hat})

# %%
# Comparison with a spatially invariant PSF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We now assume a single PSF for the whole image, estimated by DeepBlur on the central :math:`256 \times 256` patch.
# This PSF is compact, as the center of the slide is close to focus: it restores the in-focus regions,
# but leaves the out-of-focus bottom right blurry, where the tiled PSFs recover the filaments.
# The figure opens on this region; double-click to see the whole image.

H, W = y.shape[-2:]
with torch.no_grad():
    h_center = model(y[..., H // 2 - 128 : H // 2 + 128, W // 2 - 128 : W // 2 + 128])
    physics.update(filters=h_center[:, :, None].expand_as(filters))
    x_invariant = ram(y, physics)[..., crop:-crop, crop:-crop]

fig = interactive_plot(
    {"Blurry image": y, "Invariant PSF": x_invariant, "Tiled PSFs": x_hat}, height=330
)
fig.update_xaxes(range=[560, 760]).update_yaxes(range=[750, 550], autorange=False)

# %%
# :References:
#
# .. footbibliography::
