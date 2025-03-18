r"""
Image generation and Posterior Sampling with Stochastic Differential Equations
====================================================================================================

This demo shows you how to use
:class:`deepinv.sampling.DiffusionSDE` to perform unconditional image generation from a pre-trained denoiser and
:class:`deepinv.sampling.PosteriorDiffusion` to perform posterior sampling.


Unconditional Image Generation
==============================

The diffusion models with SDE paper can be found at https://arxiv.org/abs/2011.13456.

This method requires:

* A well-trained denoiser with varying noise levels (ideally with large noise levels) (e.g.,
:class:`deepinv.models.NCSNpp`).
* Define a drift term :math:`f(x, t)` and a diffusion term :math:`g(t)` for the forward-time SDE.

The forward-time SDE is defined as follows, for :math:`t \in [0, T]`:

.. math::
    d\, x_t = f(x_t, t) d\,t + g(t) d\, w_t.

Let :math:`p_t` denote the distribution of the random vector :math:`x_t`.
The reverse-time SDE is defined as follows, running backward in time:

.. math::
   d\, x_{t} = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_t(x_t) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t},

where a scalar :math:`\alpha \in [0,1]` weighting the diffusion term. :math:`\alpha = 0` corresponds to the ODE sampling and :math:`\alpha > 0` corresponds to the SDE sampling.

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
# -----------------------------
# Let us import the necessary modules and define the denoiser

import torch
import numpy as np

import deepinv as dinv
from deepinv.models import NCSNpp, EDMPrecond

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# %%
# Load the pre-trained denoiser.
# In this example, we use the pre-trained FFHQ-64 model from the
# EDM framework: https://arxiv.org/pdf/2206.00364 .
# The network architecture is from Song et al: https://arxiv.org/abs/2011.13456 .

unet = NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")
denoiser = EDMPrecond(model=unet).to(device)

# %%
# Define the SDE
# ______________
#
# In this example, we use the Variance-Exploding SDE, whose forward process is defined as:
#
# .. math::
#     d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}}\left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t

from deepinv.sampling import VarianceExplodingDiffusion, EulerSolver

sigma_min = 0.02
sigma_max = 20

sde = VarianceExplodingDiffusion(
    denoiser=denoiser,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    device=device,
    dtype=dtype,
)

# %%
# Reverse-time SDE as generative process
# --------------------------------------
#
# Sampling is performed by solving the reverse-time SDE. To do so, we generate a reverse-time trajectory.

num_steps = 100
timesteps = np.linspace(0, 1, num_steps)[::-1]

# The solution is obtained by calling the SDE object with a desired solver (here, Euler).
# The reproducibility of the SDE Solver class can be controlled by providing the pseudo-random number generator.
rng = torch.Generator(device).manual_seed(42)
solver = EulerSolver(timesteps=timesteps, rng=rng)
sample_seed_1, trajectory_seed_1 = sde.sample(
    (1, 3, 64, 64), solver=solver, seed=1, get_trajectory=True
)

dinv.utils.plot(
    sample_seed_1, titles="VE-SDE sample", save_fn="sde_sample.png", show=True
)
dinv.utils.save_videos(
    trajectory_seed_1.cpu()[::4],
    time_dim=0,
    titles=["VE-SDE Trajectory"],
    save_fn="sde_trajectory.gif",
)

# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path


try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.copyfile("sde_trajectory.gif", final_dir / "sde_trajectory_0.gif")
    shutil.copyfile("sde_sample.png", final_dir / "sde_sample.png")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore
# %%
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/sde_trajectory_0.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/sde_trajectory_0.gif
#       :class: custom-gif
#
#    .. image-sg:: /auto_examples/images/sde_sample.png
#       :alt: other example
#       :srcset: /auto_examples/images/sde_sample.png
#       :class: custom-gif


# %% Varying samples
# -----------------
#
# One can obtain varying samples by using a different seed.
# To ensure the reproducibility, if the parameter `rng` is given, the same sample will
# be generated when the same seed is used

# By changing the seed, we can obtain different samples:
sample_seed_111 = sde.sample(
    (1, 3, 64, 64), solver=solver, seed=111, get_trajectory=False
)

dinv.utils.plot(
    [sample_seed_1, sample_seed_111],
    titles=[
        "seed 1",
        "seed 111",
    ],
    show=True,
)


# %%
# Plug-and-play Image Generation with arbitrary denoisers
# -------------------------------------------------------
# The `SDE` class can be used together with any (well-trained) denoisers for image generation.
# For example, we can use, for example the :meth:`deepinv.models.DRUNet` for image generation.
# However, it should be careful to set the parameter `rescale` to `True` when instantiating the SDE class, since
# the DRUNet was trained on [0, 1] images.

sigma_min = 0.02
sigma_max = 10
rng = torch.Generator(device)
denoiser = dinv.models.DRUNet(pretrained="download").to(device)

sde = VarianceExplodingDiffusion(
    denoiser=denoiser,
    rescale=True,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    device=device,
)

# We then can generate an image by solving the reverse-time SDE
timesteps = np.linspace(0.001, 1, num_steps)[::-1]
sample, trajectory = sde.sample(
    (1, 3, 64, 64), solver=solver, seed=101, get_trajectory=True
)
dinv.utils.plot(sample, titles=["VE-SDE sample"], show=True, save_fn="sde_sample.png")
dinv.utils.save_videos(
    trajectory.cpu()[::4],
    time_dim=0,
    suptitle="VE-SDE Trajectory",
    save_fn="sde_trajectory.gif",
)

# %%
# Below, we show the trajectory and the sample generated by the VE-SDE with the DRUNet denoiser.

# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path

try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.copyfile("sde_trajectory.gif", final_dir / "sde_trajectory_1.gif")
    shutil.copyfile("sde_sample.png", final_dir / "sde_sample_1.png")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/sde_trajectory_1.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/sde_trajectory_1.gif
#       :class: custom-gif
#
#    .. image-sg:: /auto_examples/images/sde_sample_1.png
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/sde_sample_1.png
#       :class: custom-gif
#

# %%
# The underlying image distribution depends on the dataset on which the denoiser was trained,
# and the end-point distribution. In VE-SDE, this depends on the `sigma_max` parameter.
# We can change its value as well.
sigma_min = 0.02
sigma_max = 20
denoiser = dinv.models.DRUNet(pretrained="download").to(device)

sde = VarianceExplodingDiffusion(
    denoiser=denoiser,
    rescale=True,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    device=device,
)

# We then can generate an image by solving the reverse-time SDE
timesteps = np.linspace(0.001, 1, num_steps)[::-1]
sample, trajectory = sde.sample(
    (1, 3, 64, 64), solver=solver, seed=111, get_trajectory=True
)

dinv.utils.save_videos(
    trajectory.cpu()[::4],
    time_dim=0,
    titles=["VE-SDE Trajectory"],
    save_fn="sde_trajectory.gif",
)

# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path

try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.copyfile("sde_trajectory.gif", final_dir / "sde_trajectory_2.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# Below, we show the trajectory and the sample generated by the VE-SDE with the DRUNet denoiser, for a different `sigma_max` parameter.
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/sde_trajectory_2.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/sde_trajectory_2.gif
#       :class: custom-gif
#
# We can switch to a different denoiser, for example, the DiffUNet denoiser from the EDM framework.

denoiser = dinv.models.DiffUNet(pretrained="download").to(device)

sde = VarianceExplodingDiffusion(
    denoiser=denoiser,
    rescale=True,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    device=device,
)

# We then can generate an image by solving the reverse-time SDE
timesteps = np.linspace(0.001, 1, num_steps)[::-1]
sample, trajectory = sde.sample(
    (1, 3, 64, 64), solver=solver, seed=10, get_trajectory=True
)

dinv.utils.save_videos(
    trajectory.cpu()[::4],
    time_dim=0,
    titles=["SDE Trajectory"],
    save_fn="sde_trajectory.gif",
)

# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path

try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.copyfile("sde_trajectory.gif", final_dir / "sde_trajectory_3.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We obtain the following trajectory and sample using the DiffUNet denoiser.
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/sde_trajectory_3.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/sde_trajectory_3.gif
#       :class: custom-gif


# %%
# Posterior Sampling for Inverse Problems
# ---------------------------------------
#
# The
# :meth:`deepinv.sampling.PosteriorDiffusion` class can be used to perform posterior sampling for inverse problems.
#
# Consider the acquisition model:
# .. math::
#     y = \noise{\forw{x}}.
#
# This class defines the reverse-time SDE for the posterior distribution :math:`p(x|y)` given the data :math:`y`:
#
# .. math::
#     d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}
#
# where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion. The drift term and the diffusion coefficient are defined by the underlying (unconditional) forward-time SDE `unconditional_sde`. The (conditional) score function :math:`\nabla_{x_t} \log p_t(x_t | y)` can be decomposed using the Bayes' rule:
#
# .. math::
#     \nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).
#
# The first term is the score function of the unconditional SDE, which is typically approximated by a MMSE denoiser using the well-known Tweedie's formula, while the
# second term is approximated by the (noisy) data-fidelity term. We implement various data-fidelity terms in
# :meth:`deepinv.sampling.NoisyDataFidelity`.

from deepinv.sampling import PosteriorDiffusion, DPSDataFidelity
from deepinv.optim.data_fidelity import Zero

unet = NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")
denoiser = EDMPrecond(model=unet).to(device)
sigma_min = 0.02
sigma_max = 10
num_steps = 100

sde = VarianceExplodingDiffusion(
    denoiser=denoiser,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    device=device,
    dtype=dtype,
    alpha=0.5,
)

rng = torch.Generator(device).manual_seed(42)
timesteps = np.linspace(0.001, 1, num_steps)[::-1]
solver = EulerSolver(timesteps=timesteps, rng=rng)

# %%
#  When the data fidelity is not given, the posterior diffusion is equivalent to the unconditional diffusion.
model = PosteriorDiffusion(
    data_fidelity=Zero(),
    unconditional_sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
)
sample = model(
    y=None,
    physics=None,
    x_init=(1, 3, 64, 64),
    seed=123,
    timesteps=timesteps,
)
dinv.utils.plot(sample, titles="Unconditional generation", show=True)


# %%
# When the data fidelity is given, together with the measurements and the physics, this class can be used to perform posterior sampling for inverse problems.
# For example, consider the inpainting problem, where we have a noisy image and we want to recover the original image.

x = sample
physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=0.5, device=device)
y = physics(x)

model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser),
    unconditional_sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
x_hat, trajectory = model(
    y,
    physics,
    x_init=(1, 3, 64, 64),
    seed=1,
    timesteps=timesteps,
    get_trajectory=True,
)
# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    suptitle="Posterior Sampling",
    titles=["Original", "Measurement", "Posterior sample"],
)
# %% We can also save the trajectory of the posterior sample
dinv.utils.save_videos(
    trajectory,
    time_dim=0,
    save_fn="posterior_trajectory.gif",
    figsize=(5, 5),
)

# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path

try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.copyfile("posterior_trajectory.gif", final_dir / "posterior_trajectory.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior trajectory
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/posterior_trajectory.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/posterior_trajectory.gif
#       :class: custom-gif
