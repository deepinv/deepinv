r"""
Use a pretrained model
====================================================================================================

Follow this example to reconstruct images using a pretrained model in one line.

We show three sets of general pretrained reconstruction methods, including:

* Pretrained feedforward :class:`Reconstruct Anything Model (RAM) <deepinv.models.MedianFilter>`; # TODO RAM
* :ref:`Plug-and-play <iterative>` with a pretrained denoiser.
* Pretrained :ref:`diffusion model <diffusion>`;

See :ref:`User Guide <pretrained-reconstructors>` for a principled comparison between methods demonstrated in this example.

.. tip::

    * Want to use your own dataset? See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py`
    * Want to use your own physics? See :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py`

"""

import deepinv as dinv
import torch

# %%
# One-line reconstruction
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's say you want to reconstruct a butterfly from noisy, partial measurements:

# Ground truth
x = dinv.utils.load_example("butterfly.png")

# Define physics
physics = dinv.physics.Inpainting(
    x.shape[1:], mask=0.5, noise_model=dinv.physics.GaussianNoise(0.1)
)

y = physics(x)

# %%
# For each model, define model in one line and reconstruct in one line.
# Pretrained Reconstruct Anything Model:

model = dinv.models.MedianFilter()  # TODO dinv.models.RAM(pretrained=True)

x_hat1 = model(y, physics)

# %%
# PnP algorithm with pretrained denoiser:
#
# .. seealso::
#     See :ref:`pretrained denoisers <pretrained-weights>` for a full list of denoisers that can be plugged into iterative/sampling algorithms.

denoiser = dinv.models.DRUNet(pretrained="download")
model = dinv.optim.DPIR(sigma=0.1, denoiser=denoiser, device="cpu")

x_hat2 = model(y, physics)

# %%
# Pretrained diffusion model (we reduce the image size for demo speed on CPU, as diffusion model is slow):

model = dinv.sampling.DDRM(denoiser, sigmas=torch.linspace(1, 0, 10))

y = y[..., :64, :64]
physics.update(mask=physics.mask[..., :64, :64])

x_hat3 = model(y, physics)

# %%
# Plot results
dinv.utils.plot(
    {
        "Ground truth": x,
        "Pretrained RAM": x_hat1,
        "Pretrained PnP": x_hat2,
        "Pretrained diffusion": x_hat3,
    }
)

# %%
# Reconstruct Anything
# ~~~~~~~~~~~~~~~~~~~~
#
# These models, such as Reconstruct Anything Model, can be used on various problems involving different physics and data.

model = dinv.models.MedianFilter()  # TODO dinv.models.RAM(pretrained=True)

# %%
# Accelerated brain MRI, using a sample image from `FastMRI <https://fastmri.med.nyu.edu/>`:

x = dinv.datasets.SimpleFastMRISliceDataset("data", anatomy="brain", download=True)[
    0
].unsqueeze(0)

physics = dinv.physics.MRI()

physics_generator = dinv.physics.generator.GaussianMaskGenerator((320, 320))

y = physics(x, **physics_generator.step())

dinv.utils.plot(
    {
        "Ground truth": x,
        "Linear inverse": physics.A_adjoint(y),
        "Pretrained RAM": model(y, physics),
    }
)

# %%
# Joint random motion deblurring and denoising, using data from color BSD:

x = dinv.utils.load_example("CBSD_0010.png")

physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:], noise_model=dinv.physics.GaussianNoise(sigma=0.05)
)

# fmt: off
physics_generator = ( 
    dinv.physics.generator.MotionBlurGenerator((31, 31), l=2.0, sigma=2.4) +
    dinv.physics.generator.SigmaGenerator(sigma_min=0.001, sigma_max=0.2)
)
# fmt: on

y = physics(x, **physics_generator.step())

dinv.utils.plot(
    {
        "Ground truth": x,
        "Linear inverse": physics.A_adjoint(y),
        "Pretrained RAM": model(y, physics),
    }
)

# %%
# Computed Tomography with limited angles and log-Poisson noise,
# using data from the `The Cancer Imaging Archive <https://link.springer.com/article/10.1007/s10278-013-9622-7>`_ of lungs:
#

x = torch.tensor(
    dinv.datasets.utils.loadmat(
        dinv.utils.demo.load_url(dinv.utils.get_image_url("CT100_256x256.mat"))
    )["DATA"]
)[None, [0]].float()

physics = dinv.physics.Tomography(
    img_width=256,
    angles=10,
    noise_model=dinv.physics.LogPoissonNoise(mu=1 / 50.0 * 362.0 / 256),
)

y = physics(x)

dinv.utils.plot(
    {
        "Ground truth": x,
        "FBP pseudo-inverse": physics.A_dagger(y),
        "Pretrained RAM": model(y, physics),
    }
)

# %%
# Satellite denoising with Poisson noise using urban data from the `WorldView-3 satellite <https://earth.esa.int/eogateway/missions/worldview-3>`_
# over Jacksonville:
#

x = dinv.utils.load_example("JAX_018_011_RGB.tif")

physics = dinv.physics.Denoising(noise_model=dinv.physics.PoissonNoise(gain=0.1))

y = physics(x)

dinv.utils.plot(
    {
        "Ground truth": x,
        "Noisy measurement": y,
        "Pretrained RAM": model(y, physics),
    }
)

# %%
# 🎉 Well done, you now know how to use DeepInverse!
#
# What's next?
# ~~~~~~~~~~~~
#
# **Want more performance**? Check out the :ref:`example on how to fine-tune a foundation model to your own problem <sphx_glr_auto_examples_models_demo_finetuning.py>`.
