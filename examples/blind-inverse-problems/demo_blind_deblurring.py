r"""
Blind deblurring with kernel estimation network
==================================================

This example demonstrates blind image deblurring using the pretrained kernel estimation network from
the paper :footcite:t:`carbajal2023blind`. The network estimates spatially-varying blur kernels from a blurred image,
which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.

The model estimates 25 spatially-varying (33 x 33) blur kernels and corresponding spatial masks (weights) of the space-varying blur model:

.. math::

    y \approx \sum_{k=1}^{25} w_k \odot (h_k \star x)

where :math:`\star` is a convolution with circular padding, :math:`\odot` is a Hadamard product, :math:`w_k` are masks
and :math:`h_k` are filters.

"""

import torch
import deepinv as dinv
from deepinv.models import KernelIdentificationNetwork, RAM
from deepinv.optim import DPIR

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load blurry image
# ~~~~~~~~~~~~~~~~~
#
# We load a real motion-blurred image from the Kohler dataset.
# You can access the whole dataset using :class:`deepinv.datasets.Kohler`.
#

y = dinv.utils.load_example("kohler.png", device=device)[:, :3, ...]

dinv.utils.plot({"Blurry Image": y})  # plot blurry image


# %%
# Estimate blur kernels
# ~~~~~~~~~~~~~~~~~~~~~
#
# We use the pretrained kernel estimation network to estimate the spatially-varying blur kernels from the blurry image.
# The network provides 25 filters and corresponding spatial masks (weights) of the space-varying blur model (kernels first configuration).
#
# We can visualise the estimated kernels by applying the forward operator to a Dirac comb input.
#
# .. note::
#     The kernel estimation network is trained on non-gamma corrected images.
#     If your input image is gamma-corrected (e.g., standard sRGB images),
#     consider applying an inverse gamma correction before passing it to the network for better results.


# load pretrained kernel estimation network
kernel_estimator = KernelIdentificationNetwork(device=device)

# define space-varying blur physics, using the convention of the pretrained network
# (masks applied after the convolutions, circular padding)
physics = dinv.physics.SpaceVaryingBlur(
    device=device, padding="circular", mask_first=False
)

with torch.no_grad():
    params = kernel_estimator(y)  # this outputs {"filters": ..., "multipliers": ...}
    physics.update(**params)
    dirac_comb = dinv.utils.dirac_comb_like(y, step=32)
    kernel_map = physics.A(dirac_comb)

    # visualize on a zoomed region
    dinv.utils.plot(
        {
            "Estimated Kernels": kernel_map[..., 200:400, 300:500],
            "Blurry Image": y[..., 200:400, 300:500],
        }
    )


# %%
# Deblur using non-blind reconstruction methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we use two different non-blind deblurring algorithms to reconstruct the sharp image from the blurry observation and the estimated blur kernels:
# Here we use the general reconstruction model :class:`deepinv.models.RAM` and the plug-and-play method :class:`deepinv.optim.DPIR`.

sigma = 0.02

model = RAM(device=device)
with torch.no_grad():
    x_ram = model(y, physics, sigma=sigma)
    x_ram = x_ram.clamp(0, 1)

model = DPIR(sigma=sigma, device=device)
x_dpir = model(y, physics)
x_dpir = x_dpir.clamp(0, 1)


# %%
# No reference metrics and visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As here we assume that we do not have access to the ground truth sharp image,
# we cannot compute reference metrics such as PSNR or SSIM.
# However, we can still compute no-reference metrics such as Blur Strength (lower is better) and
# Sharpness Index (higher is better)
# to assess the quality of the reconstructions.

center_crop = -10  # remove 10 pixels from each border to avoid boundary effects

bs = dinv.metric.BlurStrength(center_crop=center_crop)

bs_blurry = bs(y).item()
bs_ram = bs(x_ram).item()
bs_dpir = bs(x_dpir).item()

si = dinv.metric.SharpnessIndex(center_crop=center_crop)

si_blurry = si(y).item()
si_ram = si(x_ram).item()
si_dpir = si(x_dpir).item()

dinv.utils.plot(
    {
        "Blurry": y[..., 200:400, 300:500],
        "RAM": x_ram[..., 200:400, 300:500],
        "DPIR": x_dpir[..., 200:400, 300:500],
    },
    subtitles=[
        f"SI: {si_blurry:.0f} \n BS: {bs_blurry:.3f}",
        f"SI: {si_ram:.0f} \n BS: {bs_ram:.3f}",
        f"SI: {si_dpir:.0f} \n BS: {bs_dpir:.3f}",
    ],
    figsize=(10, 5),
)

# %%
# :References:
#
# .. footbibliography::
