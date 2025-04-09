r"""
Posterior Sampling for Inverse Problems with Stochastic Differential Equations modeling.
====================================================================================================

This demo shows you how to use
:class:`deepinv.sampling.PosteriorDiffusion` to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.

This method requires:

* A well-trained denoiser with varying noise levels (ideally with large noise levels) (e.g., :class:`deepinv.models.NCSNpp`).

* A (noisy) data fidelity term (e.g., :class:`deepinv.sampling.DPSDataFidelity`).

* Define a drift term :math:`f(x, t)` and a diffusion term :math:`g(t)` for the forward-time SDE. They can be defined through the :class:`deepinv.sampling.DiffusionSDE` (e.g., :class:`deepinv.sampling.VarianceExplodingDiffusion`).

The :class:`deepinv.sampling.PosteriorDiffusion` class can be used to perform posterior sampling for inverse problems.
Consider the acquisition model:

.. math::
     y = \noise{\forw{x}}.

This class defines the reverse-time SDE for the posterior distribution :math:`p(x|y)` given the data :math:`y`:

.. math::
     d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}

where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion. The drift term and the diffusion coefficient are defined by the underlying (unconditional) forward-time SDE `sde`. The (conditional) score function :math:`\nabla_{x_t} \log p_t(x_t | y)` can be decomposed using the Bayes' rule:

.. math::
     \nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).

The first term is the score function of the unconditional SDE, which is typically approximated by a MMSE denoiser (`denoiser`) using the well-known Tweedie's formula, while the
second term is approximated by the (noisy) data-fidelity term (`data_fidelity`).
We implement various data-fidelity terms in `the documentations <https://deepinv.github.io/deepinv/user_guide/reconstruction/sampling.html#id2>`_.
"""

# %% Define the underlying SDE for posterior sampling
# ---------------------------------------------------
#
# Let us import the necessary modules, define the denoiser and the SDE.
#
# In this example, we use the Variance-Exploding SDE, whose forward process is defined as:
#
# .. math::
#     d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}}\left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t

import torch
import deepinv as dinv
from deepinv.models import NCSNpp, EDMPrecond

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
figsize = 2.5
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
unet = NCSNpp(pretrained="download")
denoiser = EDMPrecond(model=unet).to(device)

# The solution is obtained by calling the SDE object with a desired solver (here, Euler).
# The reproducibility of the SDE Solver class can be controlled by providing the pseudo-random number generator.
num_steps = 150
rng = torch.Generator(device).manual_seed(42)
timesteps = torch.linspace(1, 0.001, num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)


sigma_min = 0.02
sigma_max = 20
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
)
sample_seed_1, trajectory_seed_1 = model(
    y=None,
    physics=None,
    x_init=(1, 3, 64, 64),
    seed=1,
    get_trajectory=True,
)
dinv.utils.plot(
    sample_seed_1,
    titles="Unconditional generation",
    show=True,
    save_fn="sde_sample.png",
    figsize=(figsize, figsize),
)
dinv.utils.save_videos(
    trajectory_seed_1.cpu()[::10],
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
    shutil.copyfile("sde_trajectory.gif", final_dir / "sde_trajectory.gif")
    shutil.copyfile("sde_sample.png", final_dir / "sde_sample.png")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore
# %%
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/sde_sample.png
#       :alt: other example
#       :srcset: /auto_examples/images/sde_sample.png
#       :class: custom-gif
#
#    .. image-sg:: /auto_examples/images/sde_trajectory.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/sde_trajectory.gif
#       :class: custom-gif

# %%
#
# When the data fidelity is given, together with the measurements and the physics, this class can be used to perform posterior sampling for inverse problems.
# For example, consider the inpainting problem, where we have a noisy image and we want to recover the original image.

x = sample_seed_1
physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=0.4, device=device)
y = physics(x)

model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
# Moreover, when the physics is given, the initial point can be inferred from the physics if not given explicitly.
x_hat, trajectory = model(
    y,
    physics,
    seed=11,
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
# %% We can also save the trajectory of the posterior sample
dinv.utils.save_videos(
    trajectory[::10],
    time_dim=0,
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
    shutil.copyfile("posterior_trajectory.gif", final_dir / "posterior_trajectory.gif")
    shutil.copyfile("posterior_sample.png", final_dir / "posterior_sample.png")

except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior trajectory
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/posterior_sample.png
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/posterior_sample.png
#       :class: custom-gif
#
#    .. image-sg:: /auto_examples/images/posterior_trajectory.gif
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/posterior_trajectory.gif
#       :class: custom-gif


# %%
# Varying samples
# ---------------
#
# One can obtain varying samples by using a different seed.
# To ensure the reproducibility, if the parameter `rng` is given, the same sample will
# be generated when the same seed is used

# By changing the seed, we can obtain different samples:
x_hat_seed_111 = model(
    y,
    physics,
    seed=111,
)
dinv.utils.plot(
    [x_hat, x_hat_seed_111],
    titles=[
        "posterior sample: seed 11",
        "posterior sample: seed 111",
    ],
    show=True,
    save_fn="posterior_samples.png",
    figsize=(figsize * 2, figsize),
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
    shutil.copyfile("posterior_samples.png", final_dir / "posterior_samples.png")

except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior trajectory
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/posterior_samples.png
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/posterior_samples.png
#       :class: custom-gif

# %%
# Plug-and-play Posterior Sampling with arbitrary denoisers
# ---------------------------------------------------------
#
# The :class:`deepinv.sampling.PosteriorDiffusion` class can be used together with any (well-trained) denoisers for posterior sampling.
# For example, we can use the :class:`deepinv.models.DRUNet` for posterior sampling.
# We can also change the underlying SDE, for example change the `sigma_max` value.

sigma_min = 0.02
sigma_max = 2.0
rng = torch.Generator(device)
timesteps = torch.linspace(1, 0.001, 200)
solver = EulerSolver(timesteps=timesteps, rng=rng)

denoiser = dinv.models.DRUNet(pretrained="download").to(device)

sde = VarianceExplodingDiffusion(
    sigma_max=sigma_max, sigma_min=sigma_min, alpha=0.1, device=device, dtype=dtype
)

# As a plug-and-play sampling method, we can also change the data fidelity term.
# But the sample quality depends on the quality of the denoiser and the data fidelity term.
model = PosteriorDiffusion(
    data_fidelity=dinv.optim.L2(),
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
x_hat, trajectory = model(
    y,
    physics,
    seed=11,
    get_trajectory=True,
)
# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample DRUNet"],
    figsize=(figsize * 3, figsize),
    save_fn="posterior_sample_DRUNet.png",
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
    shutil.copyfile(
        "posterior_sample_DRUNet.png", final_dir / "posterior_sample_DRUNet.png"
    )

except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior trajectory
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/posterior_sample_DRUNet.png
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/posterior_sample_DRUNet.png
#       :class: custom-gif

# %%
#
# We can switch to a different denoiser, for example, the DiffUNet denoiser from the EDM framework.
#
denoiser = dinv.models.DiffUNet(pretrained="download").to(device)

sigma_min = 0.02
sigma_max = 5.0
rng = torch.Generator(device)
timesteps = torch.linspace(1, 0.001, 200)
solver = EulerSolver(timesteps=timesteps, rng=rng)
sde = VarianceExplodingDiffusion(
    sigma_max=sigma_max, sigma_min=sigma_min, alpha=0.5, device=device, dtype=dtype
)
model = PosteriorDiffusion(
    data_fidelity=DPSDataFidelity(denoiser=denoiser),
    denoiser=denoiser,
    rescale=True,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
x_hat, trajectory = model(
    y,
    physics,
    seed=1,
    get_trajectory=True,
)
# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample DiffUNet"],
    save_fn="posterior_sample_DiffUNet.png",
    figsize=(figsize * 3, figsize),
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
    shutil.copyfile(
        "posterior_sample_DiffUNet.png", final_dir / "posterior_sample_DiffUNet.png"
    )
except FileNotFoundError:
    pass
# sphinx_gallery_end_ignore

# %%
# We obtain the following posterior trajectory
#
# .. container:: image-row
#
#    .. image-sg:: /auto_examples/images/posterior_sample_DiffUNet.png
#       :alt: example learn_samples
#       :srcset: /auto_examples/images/posterior_sample_DiffUNet.png
#       :class: custom-gif
