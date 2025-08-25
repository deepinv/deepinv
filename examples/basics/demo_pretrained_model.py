r"""
Use a pretrained model
====================================================================================================

Follow this example to reconstruct images using a pretrained model in one line.

We show three sets of general pretrained reconstruction methods, including:

* Pretrained feedforward :class:`Reconstruct Anything Model (RAM) <deepinv.models.RAM>`;
* :ref:`Plug-and-play <iterative>` with a pretrained denoiser.
* Pretrained :ref:`diffusion model <diffusion>`;

See :ref:`pretrained models <pretrained-models>` for a principled comparison between methods demonstrated in this example.

.. tip::

    * Want to use your own dataset? See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py`
    * Want to use your own physics? See :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py`

"""

import deepinv as dinv
import torch

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Let's say you want to reconstruct a butterfly from noisy, blurry measurements:

# Ground truth
x = dinv.utils.load_example("butterfly.png", device=device)

# Define physics
physics = dinv.physics.BlurFFT(
    x.shape[1:],
    filter=dinv.physics.blur.gaussian_blur((5, 5)),
    noise_model=dinv.physics.GaussianNoise(sigma=0.1),
    device=device,
)

y = physics(x)

# %%
# For each model, define model in one line and reconstruct in one line.
# Pretrained Reconstruct Anything Model:
#
# .. seealso::
#     See :ref:`sphx_glr_auto_examples_models_demo_foundation_model.py` for further one-line examples for the RAM model across various domains.

model = dinv.models.RAM(pretrained=True, device=device)

with torch.no_grad():
    x_hat1 = model(y, physics)

# %%
# PnP algorithm with pretrained denoiser:
#
# .. seealso::
#     See :ref:`pretrained denoisers <pretrained-weights>` for a full list of denoisers that can be plugged into iterative/sampling algorithms.
#
#     See :ref:`sphx_glr_auto_examples_plug-and-play_demo_PnP_DPIR_deblur.py` for a further example of using plug-and-play with a pretrained denoiser.

denoiser = dinv.models.DRUNet(device=device)
model = dinv.optim.DPIR(sigma=0.1, denoiser=denoiser, device=device)

x_hat2 = model(y, physics)

# %%
# Pretrained diffusion model (we reduce the image size for demo speed on CPU, as diffusion model is slow):
#
# .. seealso::
#     See :ref:`sphx_glr_auto_examples_sampling_demo_ddrm.py` for a further example of using a pretrained diffusion model.


model = dinv.sampling.DDRM(denoiser, sigmas=torch.linspace(1, 0, 20)).to(device)

x_hat3 = model(y, physics)

# %%
# Plot results
dinv.utils.plot(
    {
        "Ground truth": x,
        "Blurred measurement": y,
        "Pretrained RAM": x_hat1,
        "Pretrained PnP": x_hat2,
        "Pretrained diffusion": x_hat3,
    }
)


# %%
# 🎉 Well done, you now know how to use pretrained models!
#
# What's next?
# ~~~~~~~~~~~~
# * Check out the :ref:`example on how to fine-tune a foundation model to your own problem <sphx_glr_auto_examples_models_demo_foundation_model.py>`.
# * See :ref:`pretrained models <pretrained-models>` for a comparison between methods demonstrated in this example.
# * See :ref:`diffusion <diffusion>` and :ref:`iterative <iterative>` for how to fully customize your sampling or iterative algorithm using a pretrained denoiser.
