r"""
Image generation with Stochastic Differential Equation Modeling
====================================================================================================

This code shows you how to use the :meth:`deepinv.sampling.DiffusionSDE` to generate from a pre-trained denoiser.

The diffusion models with SDE paper can be found at https://arxiv.org/abs/2011.13456.

This method requires:

* A well-trained denoiser with varying noise levels (ideally with large noise levels) (e.g., :class:`deepinv.model.edm.NCSNpp`).
* Define a drift term :math:`f(x, t)` and a diffusion term :math:`g(t)` for the forward-time SDE.

The forward-time SDE is defined as follows, for :math:`t \in [0, T]`:

.. math::
    d\, x_t = f(x_t, t) d\,t + g(t) d\, w_t.

Let :math:`p_t` denote the distribution of the random vector :math:`x_t`.
The reverse-time SDE is defined as follows, running backward in time:

.. math::
    d\, x_t =\left(f(x_t, t) - g(t)^2 \nabla \log p_t(x_t)\right( d\,t + g(t) d\, w_t.

This reverse-time SDE can be used as a generative process. 
The (Stein) score function :math:`\nabla \log p_t(x_t)` can be approximated by Tweedie's formula. In particular, if 

.. math::
    x_t \vert x_0 \sim \mathcal{N}(\mu_tx_0, \sigma_t^2 \mathrm{Id}),

then

.. math::
    \nabla \log p_t(x_t) = \frac{\mu_t  D_{\sigma_t}(x_t) -  x_t }{\sigma_t^2}.

Starting from a random point following the end-point distribution :math:`p_T` of the forward process, 
solving the reverse-time SDE gives us a sample of the data distribution :math:`p_0`.
"""

# %% Define the forward process
# -------------------------------
# Let us import the necessary modules and define the denoiser

import torch
import numpy as np

import deepinv as dinv
from deepinv.models.edm import NCSNpp, EDMPrecond, ADMUNet
from deepinv.utils.plotting import plot

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# %%
# Load the pre-trained denoiser.
# In this example, we use the pre-trained FFHQ-64 model from the
# EDM framework: https://arxiv.org/pdf/2206.00364
# The network architecture is from Song et al: https://arxiv.org/abs/2011.13456

unet = NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")
denoiser = EDMPrecond(model=unet).to(device)

# %%
# Define the SDE. In this example, we use the Variance-Exploding SDE, whose forward process is defined as:
#
# .. math::
#     d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}}\left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right(^t

from deepinv.sampling.sde import VESDE

sigma_min = 0.02
sigma_max = 20

# The reproducibility of the SDE class can be controlled by providing the pseudo-random number generator.
rng = torch.Generator(device).manual_seed(42)

sde = VESDE(
    denoiser=denoiser,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    rng=rng,
    device=device,
    use_backward_ode=False,
)

# %% Reverse-time SDE as generative process
# -----------------------------------------

timesteps = np.linspace(0, 1, 200)[::-1]
solution = sde(
    (1, 3, 64, 64), timesteps=timesteps, method="Euler", seed=1, full_trajectory=True
)
dinv.utils.plot_videos(
    solution.trajectory.cpu()[::4],
    time_dim=0,
    titles=[f"SDE Trajectory"],
    display=True,
)

# %% Varying samples
# One can obtain varying samples by using a different seed.
# To ensure the reproducibility, if the parameter :attrib:`rng` is given, the same sample will
# be generated when the same seed is used
solution = sde(
    (1, 3, 64, 64), timesteps=timesteps, method="Euler", seed=1, full_trajectory=True
)
dinv.utils.plot(solution.sample, suptitle=f"Backward sample, nfe = {solution.nfe}")

# By changing the seed, we can obtain different samples
solution = sde(
    (1, 3, 64, 64), timesteps=timesteps, method="Euler", seed=111, full_trajectory=True
)
dinv.utils.plot(solution.sample, suptitle=f"Backward sample, nfe = {solution.nfe}")


# %% Plug-and-play Image Generation with arbitrary denoisers
# ----------------------------------------------------------
# The `SDE` class can be used together with any (well-trained) denoisers for image generation.
# For example, we can use, for example the :meth:`deepinv.models.DRUNet` for image generation.
# However, it should be careful to set the parameter `rescale` to `True` when instantiating the SDE class, since
# the DRUNet was trained on [0, 1] images.

sigma_min = 0.02
sigma_max = 10
rng = torch.Generator(device)
denoiser = dinv.models.DRUNet(pretrained="download").to(device)

sde = VESDE(
    denoiser=denoiser,
    rescale=True,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    rng=rng,
    device=device,
    use_backward_ode=False,
)

# We then can generate an image by solving the reverse-time SDE
timesteps = np.linspace(0.001, 1, 200)[::-1]
solution = sde(
    (1, 3, 64, 64), timesteps=timesteps, method="Euler", seed=1, full_trajectory=True
)

dinv.utils.plot_videos(
    solution.trajectory.cpu()[::4],
    time_dim=0,
    titles=[f"SDE Trajectory"],
    display=True,
)

# %% The underlying image distribution depends on the dataset on which the denoiser was trained,
# and the end-point distribution. In VE-SDE, this depends on the `sigma_max`.
# We can change its value as well.
sigma_min = 0.02
sigma_max = 20
rng = torch.Generator(device)
denoiser = dinv.models.DRUNet(pretrained="download").to(device)

sde = VESDE(
    denoiser=denoiser,
    rescale=True,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    rng=rng,
    device=device,
    use_backward_ode=False,
)

# We then can generate an image by solving the reverse-time SDE
timesteps = np.linspace(0.001, 1, 200)[::-1]
solution = sde(
    (1, 3, 64, 64), timesteps=timesteps, method="Euler", seed=1, full_trajectory=True
)

# %% We could also use the :meth:`deepinv.models.DRUNet`.
#
dinv.utils.plot_videos(
    solution.trajectory.cpu()[::4],
    time_dim=0,
    titles=[f"SDE Trajectory"],
    display=True,
)

denoiser = dinv.models.DiffUNet(pretrained="download").to(device)

sde = VESDE(
    denoiser=denoiser,
    rescale=True,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    rng=rng,
    device=device,
    use_backward_ode=False,
)

# We then can generate an image by solving the reverse-time SDE
timesteps = np.linspace(0.001, 1, 200)[::-1]
solution = sde(
    (1, 3, 64, 64), timesteps=timesteps, method="Euler", seed=1, full_trajectory=True
)

dinv.utils.plot_videos(
    solution.trajectory.cpu()[::4],
    time_dim=0,
    titles=[f"SDE Trajectory"],
    display=True,
)
