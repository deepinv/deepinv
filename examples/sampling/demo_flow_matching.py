r"""
Flow-Matching for posterior sampling and unconditional generation
==================================================================

This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).

Flow matching consists in building a continuous transportation between a reference distribution :math:`p_0` which is easy to sample from (e.g., a Gaussian distribution) and the data distribution :math:`p_1`.
Sampling is done by solving the following ordinary differential equation (ODE) defined by a time-dependent velocity field :math:`v_\theta(x,t)`:

.. math::
    \frac{dx_t}{dt} = v_\theta(x_t,t), \quad x_0 \sim p_0 \quad t \in [0,1]

The velocity field :math:`v_\theta(x,t)` is typically trained to approximate the conditional expectation:

.. math::
    v_\theta(x_t,t) \approx \mathbb{E}_{x_0 \sim p_0, x_1 \sim p_1}\Big[ \frac{d}{dt} x_t | x_t = a(t) x_0 + b(t) x_1 \Big]

where :math:`a(t)` and :math:`b(t)` are interpolation coefficients such that :math:`x_t` interpolates between :math:`x_0` and :math:`x_1`.
When the reference distribution :math:`p_0` is the standard Gaussian, the velocity field can be expressed as a function of a Gaussian denoiser :math:`D(x, \sigma)` as follows:

.. math::
    v_\theta(x_t,t) = - \frac{b'(t)}{b(t)} x_t + \frac{1}{2}\frac{a(t) b'(t) - a'(t) b(t)}{a(t) b(t)} \left(D\left(\frac{x_t}{a(t)}, \frac{b(t)}{a(t)} \right) - x_t\right)

The most common choice of time schedulers is the linear schedule :math:`a(t) = 1 - t` and :math:`b(t) = t`.

In this demo, we will show how to :

    *  Perform unconditional generation using, instead of a trained denoiser, the closed-form MMSE denoiser

    .. math::
        D(x, \sigma) = \mathbb{E}_{x_0 \sim p_{data}, \epsilon \sim \mathcal{N}(0, I)} \Big[ x_0 | x = x_0 + \sigma \epsilon \Big]

    Given a dataset of clean images, it can be computed by evaluating the distance between the input image and all the points of the dataset (see :class:`deepinv.models.MMSE`).

    *  Perform posterior sampling using Flow-Matching combined with a DPS data fidelity term (see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for more details)

    *  Explore different choices of time schedulers :math:`a(t)` and :math:`b(t)`.

"""

# %%
import torch
import deepinv as dinv
from deepinv.sampling import (
    PosteriorDiffusion,
    DPSDataFidelity,
    EulerSolver,
    FlowMatching,
)
import numpy as np
from torchvision import datasets, transforms
from deepinv.models import MMSE, NCSNpp
import os
import shutil
from pathlib import Path

# %% Define the closed-form MMSE denoiser
# -----------------------------
#
# We start by working with the closed-form MMSE denoser.  It is calculated by computing the distance between the input image and all the points of the dataset.
# This can be quite long to compute for large images and large datasets.  In this toy example, we use the validation set of MNIST.
# When using this closed-form MMSE denoiser, the sampling is guaranteed to output an image of the dataset.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.float32

figsize = 2.5

# We use the closed-form MMSE denoiser defined using as atoms the testset of MNIST.
# The deepinv MMSE denoiser takes as input a dataloader.
dataset = datasets.MNIST(
    root=".", train=False, download=True, transform=transforms.ToTensor()
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
n_max = (
    1000  # limit the number of images to speed up the computation of the MMSE denoiser
)
tensors = torch.cat([data[0] for data in iter(dataloader)], dim=0)  # (N,1,28,28)
tensors = tensors[:n_max].to(device)
denoiser = MMSE(dataloader=tensors, device=device, dtype=dtype)


# %% Define the Flow-Matching ODE and perform unconditional generation
# ---------------------------------------------------------------------
#
# The FlowMatching module :class:`deepinv.sampling.FlowMatching` uses by default the following schedules: :math:`a_t=1-t`, :math:`b_t=t`.
# The module FlowMatching module takes as input the denoiser and the ODE solver.

num_steps = 100
timesteps = torch.linspace(0.99, 0.0, num_steps)
rng = torch.Generator(device).manual_seed(5)
solver = EulerSolver(timesteps=timesteps, rng=rng)
sde = FlowMatching(denoiser=denoiser, solver=solver, device=device, dtype=dtype)


sample, trajectory = sde(
    x_init=(1, 1, 28, 28),
    seed=0,
    get_trajectory=True,
)

dinv.utils.plot(
    sample,
    titles="Unconditional FM generation",
    save_fn="FM_sample.png",
    figsize=(figsize, figsize),
)

# sphinx_gallery_start_ignore
# cleanup
try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.move("FM_trajectory.gif", final_dir / "FM_trajectory.gif")
    shutil.move("FM_sample.png", final_dir / "FM_sample.png")
except FileNotFoundError:
    pass
# sphinx_gallery_end_ignore

# %%
# We obtain the following unconditional sample, which belongs to the MNIST dataset.
#
# .. container:: image-row
#
#    .. image-sg-ignore:: /auto_examples/images/FM_sample.png
#       :alt: example of unconditional sample
#       :srcset: /auto_examples/images/FM_sample.png
#       :class: custom-img
#       :ignore_missing: true
#
#    .. image-sg-ignore:: /auto_examples/images/FM_trajectory.gif
#       :alt: example of unconditional trajectory
#       :srcset: /auto_examples/images/FM_trajectory.gif
#       :class: custom-gif
#       :ignore_missing: true


# %% Perform posterior sampling
# -----------------------------------------------------------------------
#
# Now, we can use the Flow-Matching model to perform posterior sampling.
# In order not to replicate training image data, we now use a pretrained deep denoiser, here the NCSNpp denoiser  :footcite:t:`song2020score`, with pretrained weights from :footcite:t:`karras2022elucidating`.
# We consider the inpainting problem, where we have a masked image and we want to recover the original image.
# We use DPS :class:`deepinv.sampling.DPSDataFidelity` as data fidelity term (see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for more details).
# Note that due to the division by :math:`a(t)` in the velocity field, initialization close to t=1 causes instability.

# denoiser = NCSNpp(pretrained="download").to(device)

# x = dinv.utils.load_example(
#         "celeba_example.jpg",
#         img_size=64,
#         resize_mode="resize",
#         device=device,
# )

x = next(iter(dataloader))[0][:1].to(device)

mask = torch.ones_like(x)
mask[..., 10:20, 10:20] = 0.0
physics = dinv.physics.Inpainting(
    img_size=x.shape[1:],
    mask=mask,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.1),
)
y = physics(x)
dps_fidelity = DPSDataFidelity(denoiser=denoiser, weight=1.0)
model = PosteriorDiffusion(
    data_fidelity=dps_fidelity,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)
x_hat, trajectory = model(
    y,
    physics,
    x_init=None,
    get_trajectory=True,
    seed=0,
)

# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample"],
    figsize=(figsize * 3, figsize),
    save_fn="FM_posterior.png",
)


# sphinx_gallery_start_ignore
# cleanup
try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.move("FM_posterior.png", final_dir / "FM_posterior.png")
except FileNotFoundError:
    pass
# sphinx_gallery_end_ignore

# %%
# We obtain the following conditional sample:
#
# .. container:: image-row
#
#    .. image-sg-ignore:: /auto_examples/images/FM_posterior.png
#       :alt: example of unconditional sample
#       :srcset: /auto_examples/images/FM_posterior.png
#       :class: custom-img
#       :ignore_missing: true


# %% Explore different time schedulers for Flow-Matching
# ----------------------------------------------------------------
#
# Finally, we show how to use different choices of time schedulers :math:`a_t` and :math:`b_t`.
# Here, we use another typical choice of schedulers :math:`a_t = \cos(\frac{\pi}{2} t)` and :math:`b_t = \sin(\frac{\pi}{2} t)` which also satisfy the interpolation condition :math:`a_0 = 1`, :math:`b_0 = 0`, :math:`a_1 = 0`, :math:`b_1 = 1`.
# Note that, again, due to the division by :math:`a_t` in the velocity field, initialization close to t=1 causes instability.

a_t = lambda t: torch.cos(np.pi / 2 * t)
a_prime_t = lambda t: -np.pi / 2 * torch.sin(np.pi / 2 * t)
b_t = lambda t: torch.sin(np.pi / 2 * t)
b_prime_t = lambda t: np.pi / 2 * torch.cos(np.pi / 2 * t)

sde = FlowMatching(
    a_t=a_t,
    a_prime_t=a_prime_t,
    b_t=b_t,
    b_prime_t=b_prime_t,
    denoiser=denoiser,
    solver=solver,
    device=device,
    dtype=dtype,
)

model = PosteriorDiffusion(
    data_fidelity=dps_fidelity,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)

x_hat, trajectory = model(
    y,
    physics,
    x_init=None,
    get_trajectory=True,
)

# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample"],
    figsize=(figsize * 3, figsize),
    save_fn="FM_posterior_new_at_bt.png",
)

# sphinx_gallery_start_ignore
# cleanup
try:
    final_dir = (
        Path(os.getcwd()).parent.parent / "docs" / "source" / "auto_examples" / "images"
    )
    shutil.move("FM_posterior_new_at_bt.png", final_dir / "FM_posterior_new_at_bt.png")
except FileNotFoundError:
    pass
# sphinx_gallery_end_ignore

# %%
# We obtain the following conditional sample:
#
# .. container:: image-row
#
#    .. image-sg-ignore:: /auto_examples/images/FM_posterior_new_at_bt.png
#       :alt: example of unconditional sample
#       :srcset: /auto_examples/images/FM_posterior_new_at_bt.png
#       :class: custom-img
#       :ignore_missing: true

# %%
# :References:
#
# .. footbibliography::
