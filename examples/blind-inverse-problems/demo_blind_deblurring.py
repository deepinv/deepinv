r"""
Blind deblurring with kernel estimation network
==================================================

This example demonstrates blind image deblurring using the pretrained kernel estimation network from
the paper :footcite:t:`carbajal2023blind`. The network estimates spatially-varying blur kernels from a blurred image,
which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.

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
# The network provides 25 filters and corresponding spatial multipliers (weights) of the space-varying blur model (:class:`deepinv.physics.SpaceVaryingBlur`).
#
# We can visualise the estimated kernels by applying the forward operator to a Dirac comb input.
#
# .. note::
#     The kernel estimation network is trained on non-gamma corrected images.
#     If your input image is gamma-corrected (e.g., standard sRGB images),
#     consider applying an inverse gamma correction before passing it to the network for better results.


# load pretrained kernel estimation network
kernel_estimator = KernelIdentificationNetwork(device=device)

# define space-varying blur physics
physics = dinv.physics.SpaceVaryingBlur(device=device, padding="constant")

with torch.no_grad():
    params = kernel_estimator(y)  # this outputs {"filters": ..., "multipliers": ...}
    physics.update(**params)
    dirac_comb = dinv.utils.dirac_comb_like(y, step=32)
    kernel_map = physics.A(dirac_comb)

    # visualize on a zoomed region
    dinv.utils.plot(
        {
            "Estimated Kernels": kernel_map[..., 128:512, 128:512],
            "Blurry Image": y[..., 200:300, 200:300],
        }
    )


# %%
# Deblur using non-blind reconstruction methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we use two different non-blind deblurring algorithms to reconstruct the sharp image from the blurry observation and the estimated blur kernels:
# Here we use the general reconstruction model :class:`deepinv.models.RAM` and the plug-and-play method :class:`deepinv.optim.DPIR`.

model = RAM(device=device)
with torch.no_grad():
    x_ram = model(y, physics, sigma=0.05)
    x_ram = x_ram.clamp(0, 1)

model = DPIR(sigma=0.05, device=device)
x_dpir = model(y, physics)
x_dpir = x_dpir.clamp(0, 1)

dinv.utils.plot({"Blurry": y, "Deblurred RAM": x_ram, "Deblurred DPIR": x_dpir})


# %%
# No reference metrics
# ~~~~~~~~~~~~~~~~~~~~
#
# As here we assume that we do not have access to the ground truth sharp image,
# we cannot compute reference metrics such as PSNR or SSIM.
# However, we can still compute no-reference metrics such as NIQE (lower is better) to assess the quality of the reconstructions.

metric = dinv.metric.NIQE()

niqe_blurry = metric(y).item()
niqe_ram = metric(x_ram).item()
niqe_dpir = metric(x_dpir).item()

print(
    f"NIQE Blurry: {niqe_blurry:.2f}, NIQE RAM: {niqe_ram:.2f}, NIQE DPIR: {niqe_dpir:.2f}"
)


# %%
# :References:
#
# .. footbibliography::
