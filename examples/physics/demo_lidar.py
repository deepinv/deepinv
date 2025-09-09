r"""
Single photon lidar operator for depth ranging.
===================================================

In this example we show how to use the :class:`deepinv.physics.SinglePhotonLidar` forward model.

"""

import deepinv as dinv
import torch
import matplotlib.pyplot as plt


# %%
# Create forward model
# -----------------------------------------------
#
# We create a lidar model with 100 bins per pixel and a Gaussian impulse response function
# with a standard deviation of 2 bins.
#
# The forward model for the case of a single depth per pixel is defined as (e.g. see :footcite:t:`rapp2020advances`):
#
# .. math::
#
#       y_{i,j,t} = \mathcal{P}(h(t-d_{i,j}) r_{i,j} + b_{i,j})
#
# where :math:`\mathcal{P}` is the Poisson noise model, :math:`h(t)` is a Gaussian impulse response function at
# time :math:`t`, :math:`d_{i,j}` is the depth of the scene at pixel :math:`(i,j)`,
# :math:`r_{i,j}` is the intensity of the scene at pixel :math:`(i,j)` and :math:`b_{i,j}` is the background noise
# at pixel :math:`(i,j)`.
#

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
bins = 100
irf_sigma = 2
physics = dinv.physics.SinglePhotonLidar(bins=bins, sigma=irf_sigma, device=device)

# %%
# Generate toy signal and measurement
# -----------------------------------------------
#
# We generate a toy signal with a single peak per pixel located around the 50th bin
# and a signal-to-background ratio of 10%.
#
# Signals should have size (B, 3, H, W) where the first channel contains the depth of the scene,
# the second channel contains the intensity of the scene and the third channel contains the
# per pixel background noise levels.
#
# The measurement associated with a signal has size (B, bins, H, W).

sbr = 0.1
signal = 50
bkg_level = signal / sbr / bins
depth = bins / 2

# depth
d = torch.ones(1, 1, 2, 2, device=device) * depth
# signal
r = torch.ones_like(d) * signal
# background
b = torch.ones_like(d) * bkg_level

x = torch.cat([d, r, b], dim=1)  # signal of size (B, 3, H, W)

y = physics(x)  # measurement of size (B, bins, H, W)


# %%
# Apply matched filtering to recover the signal and plot the results
# -------------------------------------------------------------------------------
#
# We apply matched filtering to recover the signal and plot the results.
#
# The measurements are shown in blue and the depth and intensity of the
# recovered signals are shown in red.


xhat = physics.A_dagger(y)

plt.figure()
for i in range(2):
    for j in range(2):
        plt.subplot(2, 2, i * 2 + j + 1)
        plt.plot(y[0, :, i, j].detach().cpu().numpy())
        plt.stem(
            xhat[0, 0, i, j].detach().cpu().numpy(),
            xhat[0, 1, i, j].detach().cpu().numpy() / 4,
            linefmt="red",
            markerfmt="red",
        )
        plt.title(f"pixel ({i}, {j})")


plt.tight_layout()
plt.show()

# %%
# :References:
#
# .. footbibliography::
