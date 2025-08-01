r"""
Use a pretrained model
====================================================================================================

Follow this example to reconstruct images using a pretrained model in one line.

We show three sets of general pretrained reconstruction methods, including:

* Pretrained feedforward :class:`Reconstruct Anything Model (RAM) <deepinv.models.RAM>`;
* :ref:`Plug-and-play <iterative>` with a pretrained denoiser.
* Pretrained :ref:`diffusion model <diffusion>`;

TODO link to pretrained weights for other denoisers and specific reconstructors

See :ref:`User Guide <pretrained-reconstructors>` for a principled comparison between methods demonstrated in this example.

.. tip::

    * Want to use your own dataset? See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py`
    * Want to use your own physics? See :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py`

"""

import deepinv as dinv
import torch

# %%
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
    {"Pretrained RAM": x_hat1, "Pretrained PnP": x_hat2, "Pretrained diffusion": x_hat3}
)

# %%
# These models, such as Reconstruct Anything Model, can be used on various problems involving different physics and data:

# TODO

# Want more performance?
# TODO link to finetuning demo
# Want more performance? Or want help?
# TODO get in touch??
