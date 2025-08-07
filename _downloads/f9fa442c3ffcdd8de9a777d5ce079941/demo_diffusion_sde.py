r"""
Building your diffusion posterior sampling method using SDEs
============================================================

This demo shows you how to use
:class:`deepinv.sampling.PosteriorDiffusion` to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.

This method requires:

* A well-trained denoiser with varying noise levels (ideally with large noise levels) (e.g., :class:`deepinv.models.NCSNpp`).

* A (noisy) data fidelity term (e.g., :class:`deepinv.sampling.DPSDataFidelity`).

* Define a drift term :math:`f(x, t)` and a diffusion term :math:`g(t)` for the forward-time SDE. They can be defined through the :class:`deepinv.sampling.DiffusionSDE` (e.g., :class:`deepinv.sampling.VarianceExplodingDiffusion`).

The :class:`deepinv.sampling.PosteriorDiffusion` class can be used to perform posterior sampling for inverse problems.
Consider the acquisition model:

.. math::
     y = \noise{\forw{x}}

where :math:`\forw{x}` is the forward operator (e.g., a convolutional operator) and :math:`\noise{\cdot}` is the noise operator (e.g., Gaussian noise).
This class defines the reverse-time SDE for the posterior distribution :math:`p(x|y)` given the data :math:`y`:

.. math::
     d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}

where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion.
The drift term and the diffusion coefficient are defined by the underlying (unconditional) forward-time SDE `sde`.
In this example, we will use 2 well-known SDE in the literature: the Variance-Exploding (VE) and Variance-Preserving (VP or DDPM).

The (conditional) score function :math:`\nabla_{x_t} \log p_t(x_t | y)` can be decomposed using the Bayes' rule:

.. math::
     \nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).

The first term is the score function of the unconditional SDE, which is typically approximated by an MMSE denoiser (`denoiser`) using the well-known Tweedie's formula, while the
second term is approximated by the (noisy) data-fidelity term (`data_fidelity`).
We implement various data-fidelity terms in `the user guide <https://deepinv.github.io/deepinv/user_guide/reconstruction/sampling.html#id2>`_.

.. note::

    In this demo, we limit the number of diffusion steps for the sake of speed, but in practice, you should use a larger number of steps to obtain better results.
"""

# %% Define the underlying SDE for posterior sampling
# ---------------------------------------------------
#
# Let us import the necessary modules, define the denoiser and the SDE.
#
# In this first example, we use the Variance-Exploding SDE, whose forward process is defined as:
#
# .. math::
#     d\, x_t = g(t) d\, w_t \quad \mbox{where } g(t) = \sigma_{\mathrm{min}}\left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t

import torch
import deepinv as dinv
from deepinv.models import NCSNpp

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
figsize = 2.5
gif_frequency = 10  # Increase this value to save the GIF saving time
# %%
from deepinv.sampling import (
    PosteriorDiffusion,
    DPSDataFidelity,
    EulerSolver,
    VarianceExplodingDiffusion,
)
from deepinv.optim import ZeroFidelity

# In this example, we use the pre-trained FFHQ-64 model from the
# EDM framework: https://arxiv.org/pdf/2206.00364 .
# The network architecture is from Song et al: https://arxiv.org/abs/2011.13456 .
denoiser = NCSNpp(pretrained="download").to(device)

# The solution is obtained by calling the SDE object with a desired solver (here, Euler).
# The reproducibility of the SDE Solver class can be controlled by providing the pseudo-random number generator.
num_steps = 150
rng = torch.Generator(device).manual_seed(42)
timesteps = torch.linspace(1, 0.001, num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)


sigma_min = 0.005
sigma_max = 5
sde = VarianceExplodingDiffusion(
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    alpha=0.5,
    device=device,
    dtype=dtype,
)

# %%
# Reverse-time SDE as sampling process
# --------------------------------------
#
# When the data fidelity is not given, the posterior diffusion is equivalent to the unconditional diffusion.
# Sampling is performed by solving the reverse-time SDE. To do so, we generate a reverse-time trajectory.

model = PosteriorDiffusion(
    data_fidelity=ZeroFidelity(),
    sde=sde,
    denoiser=denoiser,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)
x, trajectory = model(
    y=None,
    physics=None,
    x_init=(1, 3, 64, 64),
    seed=1,
    get_trajectory=True,
)
dinv.utils.plot(
    x,
    titles="Unconditional generation",
    save_fn="sde_sample.png",
    figsize=(figsize, figsize),
)

dinv.utils.save_videos(
    trajectory.cpu()[::gif_frequency],
    time_dim=0,
    titles=["VE-SDE Trajectory"],
    save_fn="sde_trajectory.gif",
    figsize=(figsize, figsize),
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
    shutil.move("sde_trajectory.gif", final_dir / "sde_trajectory.gif")
    shutil.move("sde_sample.png", final_dir / "sde_sample.png")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore
# %%
# We obtain the following unconditional sample
#
# .. container:: image-row
#
#    .. image-sg-ignore:: /auto_examples/images/sde_sample.png
#       :alt: example of unconditional sample
#       :srcset: /auto_examples/images/sde_sample.png
#       :class: custom-img
#       :ignore_missing: true
#
#    .. image-sg-ignore:: /auto_examples/images/sde_trajectory.gif
#       :alt: example of unconditional trajectory
#       :srcset: /auto_examples/images/sde_trajectory.gif
#       :class: custom-gif
#       :ignore_missing: true

# %%
#
# When the data fidelity is given, together with the measurements and the physics, this class can be used to perform posterior sampling for inverse problems.
# For example, consider the inpainting problem, where we have a noisy image and we want to recover the original image.
# We can use the :class:`deepinv.sampling.DPSDataFidelity` as the data fidelity term.

del trajectory  # clean memory
mask = torch.ones_like(x)
mask[..., 24:40, 24:40] = 0.0
physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=mask, device=device)
y = physics(x)

weight = 1.0  # guidance strength
dps_fidelity = DPSDataFidelity(denoiser=denoiser, weight=weight)

model = PosteriorDiffusion(
    data_fidelity=dps_fidelity,
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
# Moreover, when the physics is given, the initial point can be inferred from the physics if not given explicitly.
seed_1 = 11
x_hat, trajectory = model(
    y,
    physics,
    seed=seed_1,
    get_trajectory=True,
)
# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample"],
    save_fn="posterior_sample.png",
    figsize=(figsize * 3, figsize),
)
# We can also save the trajectory of the posterior sample
dinv.utils.save_videos(
    trajectory[::gif_frequency],
    time_dim=0,
    titles=["Posterior sample with VE"],
    save_fn="posterior_trajectory.gif",
    figsize=(figsize, figsize),
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
    shutil.move("posterior_trajectory.gif", final_dir / "posterior_trajectory.gif")
    shutil.move("posterior_sample.png", final_dir / "posterior_sample.png")

except FileNotFoundError:
    pass


# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior sample and trajectory
#
# .. container:: image-col
#
#    .. image-sg-ignore:: /auto_examples/images/posterior_sample.png
#       :alt: example of posterior sample
#       :srcset: /auto_examples/images/posterior_sample.png
#       :ignore_missing: true
#
#    .. image-sg-ignore:: /auto_examples/images/posterior_trajectory.gif
#       :alt: example of posterior trajectory
#       :srcset: /auto_examples/images/posterior_trajectory.gif
#       :ignore_missing: true
#       :class: custom-gif


# %%
#   .. note::
#
#       **Reproducibility**: To ensure the reproducibility, if the parameter `rng` is given, the same sample will
#       be generated when the same seed is used.
#       One can obtain varying samples by using a different seed.
#
#       **Parallel sampling**: one can draw multiple samples in parallel by giving the initial shape, e.g., `x_init = (B, C, H, W)`
#

# %%
# Varying the SDE
# ---------------
#
# One can also change the underlying SDE for sampling.
# For example, we can also use the Variance-Preserving (VP or DDPM) in :class:`deepinv.sampling.VariancePreservingDiffusion`, whose forward drift and diffusion term are defined as:
#
# .. math::
#     f(x_t, t) = -\frac{1}{2} \beta(t)x_t \qquad \mbox{ and } \qquad g(t) = \beta(t)  \qquad \mbox{ with } \beta(t) = \beta_{\mathrm{min}}  + t \left( \beta_{\mathrm{max}} - \beta_{\mathrm{min}} \right).

from deepinv.sampling import VariancePreservingDiffusion

del trajectory
sde = VariancePreservingDiffusion(device=device, dtype=dtype)
model = PosteriorDiffusion(
    data_fidelity=dps_fidelity,
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    device=device,
    dtype=dtype,
    verbose=True,
)

x_hat_vp, trajectory = model(
    y,
    physics,
    seed=111,
    timesteps=torch.linspace(1, 0.001, 150),
    get_trajectory=True,
)
dinv.utils.plot(
    [x_hat, x_hat_vp],
    titles=[
        "posterior sample with VE",
        "posterior sample with VP",
    ],
    save_fn="posterior_sample_ve_vp.png",
    figsize=(figsize * 3, figsize),
)


# We can also save the trajectory of the posterior sample
dinv.utils.save_videos(
    trajectory[::gif_frequency],
    time_dim=0,
    titles=["Posterior sample with VP"],
    save_fn="posterior_trajectory_vp.gif",
    figsize=(figsize, figsize),
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
    shutil.move("posterior_sample_ve_vp.png", final_dir / "posterior_sample_ve_vp.png")
    shutil.move(
        "posterior_trajectory_vp.gif", final_dir / "posterior_trajectory_vp.gif"
    )

except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We can comparing the sampling trajectory depending on the underlying SDE
#
# .. container:: image-col
#
#    .. image-sg-ignore:: /auto_examples/images/posterior_sample_ve_vp.png
#       :alt: posterior sample with VP
#       :srcset: /auto_examples/images/posterior_sample_ve_vp.png
#       :ignore_missing: true
#    .. container:: image-row
#
#       .. image-sg-ignore:: /auto_examples/images/posterior_trajectory.gif
#           :alt: posterior trajectory with VE
#           :srcset: /auto_examples/images/posterior_trajectory.gif
#           :ignore_missing: true
#           :class: custom-gif
#
#       .. image-sg-ignore:: /auto_examples/images/posterior_trajectory_vp.gif
#           :alt: posterior trajectory with VP
#           :srcset: /auto_examples/images/posterior_trajectory_vp.gif
#           :ignore_missing: true
#           :class: custom-gif

# %%
# Plug-and-play Posterior Sampling with arbitrary denoisers
# ---------------------------------------------------------
#
# The :class:`deepinv.sampling.PosteriorDiffusion` class can be used together with any (well-trained) denoisers for posterior sampling.
# For example, we can use the :class:`deepinv.models.DRUNet` for posterior sampling.
# We can also change the underlying SDE, for example change the `sigma_max` value.

del trajectory  # clean memory
sigma_min = 0.001
sigma_max = 10.0
rng = torch.Generator(device)
dtype = torch.float32
timesteps = torch.linspace(1, 0.001, 250)
solver = EulerSolver(timesteps=timesteps, rng=rng)
denoiser = dinv.models.DRUNet(pretrained="download").to(device)

sde = VarianceExplodingDiffusion(
    sigma_max=sigma_max, sigma_min=sigma_min, alpha=0.75, device=device, dtype=dtype
)
x = dinv.utils.load_example(
    "butterfly.png",
    img_size=256,
    resize_mode="resize",
).to(device)

mask = torch.ones_like(x)
mask[..., 70:150, 120:180] = 0
physics = dinv.physics.Inpainting(
    mask=mask,
    img_size=x.shape[1:],
    device=device,
)

y = physics(x)
model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser, weight=0.3),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
x_hat, trajectory = model(
    y=y,
    physics=physics,
    seed=12,
    get_trajectory=True,
)

# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat.clip(0, 1)],
    titles=["Original", "Measurement", "Posterior sample DRUNet"],
    figsize=(figsize * 3, figsize),
    save_fn="posterior_sample_DRUNet.png",
)

# We can also save the trajectory of the posterior sample
dinv.utils.save_videos(
    trajectory[::gif_frequency].clip(0, 1),
    time_dim=0,
    titles=["Posterior trajectory DRUNet"],
    save_fn="posterior_sample_DRUNet.gif",
    figsize=(figsize, figsize),
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
    shutil.move(
        "posterior_sample_DRUNet.png", final_dir / "posterior_sample_DRUNet.png"
    )
    shutil.move(
        "posterior_sample_DRUNet.gif", final_dir / "posterior_sample_DRUNet.gif"
    )

except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior trajectory
#
# .. container:: image-col
#
#    .. image-sg-ignore:: /auto_examples/images/posterior_sample_DRUNet.png
#       :alt: posterior sample DRUNet
#       :srcset: /auto_examples/images/posterior_sample_DRUNet.png
#       :ignore_missing: true
#
#    .. image-sg-ignore:: /auto_examples/images/posterior_sample_DRUNet.gif
#       :alt: posterior trajectory DRUNet
#       :srcset: /auto_examples/images/posterior_sample_DRUNet.gif
#       :ignore_missing: true
#       :class: custom-gif
