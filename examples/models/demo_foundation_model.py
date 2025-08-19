"""
Inference and fine-tune a foundation model
==========================================

This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footcite:p:`terris2025reconstruct` to solve inverse problems.

The :class:`Reconstruct Anything Model <deepinv.models.RAM>` is a model that has been trained to work on a large
variety of linear image reconstruction tasks and datasets (deblurring, inpainting, denoising, tomography, MRI, etc.)
and is robust to a wide variety of imaging domains.

.. tip::

    * Want to use your own dataset? See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py`
    * Want to use your own physics? See :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py`

"""

import deepinv as dinv
import torch

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

model = dinv.models.RAM(device=device, pretrained=True)

# %%
# 1. Zero-shot inference
# ----------------------
#
# First, let's evaluate the zero-shot inference performance of the foundation model.
#
# Accelerated medical imaging
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here, we demonstrated reconstructing brain MRI from an accelerated noisy MRI scan from `FastMRI <https://fastmri.med.nyu.edu/>`_:

x = dinv.utils.load_example("demo_mini_subset_fastmri_brain_0.pt", device=device)

# Define physics
physics = dinv.physics.MRI(noise_model=dinv.physics.GaussianNoise(0.05), device=device)

physics_generator = dinv.physics.generator.GaussianMaskGenerator(
    (320, 320), device=device
)

# Generate measurement
y = physics(x, **physics_generator.step())

# Perform inference
with torch.no_grad():
    x_hat = model(y, physics)
    x_lin = physics.A_adjoint(y)

psnr = dinv.metric.PSNR()

dinv.utils.plot(
    {
        "Ground truth": x,
        f"Linear inverse\n PSNR {psnr(x_lin, x).item():.2f}dB": x_lin,
        f"Pretrained RAM\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    }
)

# %%
# Computational photography
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Joint random motion deblurring and denoising, using a cropped image from color BSD:

x = dinv.utils.load_example("CBSD_0010.png", img_size=(200, 200), device=device)

physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=0.05),
    device=device,
)

# fmt: off
physics_generator = ( 
    dinv.physics.generator.MotionBlurGenerator((31, 31), l=2.0, sigma=2.4, device=device) +
    dinv.physics.generator.SigmaGenerator(sigma_min=0.001, sigma_max=0.2, device=device)
)
# fmt: on

y = physics(x, **physics_generator.step())

with torch.no_grad():
    x_hat = model(y, physics)
    x_lin = physics.A_adjoint(y)

dinv.utils.plot(
    {
        "Ground truth": x,
        f"Linear inverse\n PSNR {psnr(x_lin, x).item():.2f}dB": x_lin,
        f"Pretrained RAM\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    }
)

# %%
# Tomography
# ~~~~~~~~~~
# Computed Tomography with limited angles
# using data from the `The Cancer Imaging Archive <https://link.springer.com/article/10.1007/s10278-013-9622-7>`_ of lungs:
#

x = dinv.utils.load_example("CT100_256x256_0.pt", device=device)

physics = dinv.physics.Tomography(
    img_width=256,
    angles=10,
    normalize=True,
    device=device,
)

y = physics(x)

with torch.no_grad():
    x_hat = model(y, physics)
    x_lin = physics.A_dagger(y)

dinv.utils.plot(
    {
        "Ground truth": x,
        f"FBP pseudo-inverse\n PSNR {psnr(x_lin, x).item():.2f}dB": x_lin,
        f"Pretrained RAM\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    }
)

# %%
# Remote sensing
# ~~~~~~~~~~~~~~
# Satellite denoising with Poisson-Gaussian noise using urban data from the `WorldView-3 satellite <https://earth.esa.int/eogateway/missions/worldview-3>`_
# over Jacksonville:
#

x = dinv.utils.load_example("JAX_018_011_RGB.tif", device=device)[..., :300, :300]

physics = dinv.physics.Denoising(
    noise_model=dinv.physics.PoissonGaussianNoise(sigma=0.1, gain=0.1)
)

y = physics(x)

with torch.no_grad():
    x_hat = model(y, physics)
    # Alternatively, use the model without physics:
    # x_hat = model(y, sigma=0.1, gain=0.1)
    x_lin = physics.A_adjoint(y)

dinv.utils.plot(
    {
        "Ground truth": x,
        f"Linear inverse\n PSNR {psnr(x_lin, x).item():.2f}dB": x_lin,
        f"Pretrained RAM\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    }
)


# %%
# 2. Fine-tuning
# --------------
# As with all models, there may be a drop in performance when used zero-shot on problems or data outside those seen during training.
#
# For instance, RAM is not trained on image demosaicing:

x = dinv.utils.load_example("butterfly.png", img_size=(127, 129), device=device)

physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=dinv.physics.PoissonNoise(0.1), device=device
)

# Generate measurement
y = physics(x)

# Run inference
with torch.no_grad():
    x_hat = model(y, physics)

# Show results
dinv.utils.plot(
    {
        "Original": x,
        f"Measurement\n PSNR {psnr(y, x).item():.2f}dB": y,
        f"Reconstruction\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    },
)

# %%
# To improve results, we can fine-tune the model on our problem and data,
# **even in the absence of ground truth data**, using a :ref:`self-supervised loss <self-supervised-losses>`,
# and **even on a single image only**.
#
# Here, since this example is run in a no-GPU environment, we will use a small patch of the image to speed up training,
# but in practice, we can use the full image.
#
# .. note::
#     You can also fine-tune on larger datasets if you want, by replacing the :ref:`dataset <datasets>`.

# Take small patch
x_train = x[..., :64, :64]

physics_train = dinv.physics.Demosaicing(
    img_size=x_train.shape[1:],
    noise_model=dinv.physics.PoissonNoise(0.1, clip_positive=True),
    device=device,
)

y_train = physics_train(x_train)

# Define training loss
losses = [
    dinv.loss.R2RLoss(),
    dinv.loss.EILoss(dinv.transform.Shift(shift_max=0.4), weight=0.1),
]

dataset = dinv.datasets.TensorDataset(y=y_train)
train_dataloader = torch.utils.data.DataLoader(dataset)

# %%
# We fine-tune using early stopping using a validation set, again without ground truth.
# We use a small patch of another set of measurements.

eval_dataloader = torch.utils.data.DataLoader(
    dinv.datasets.TensorDataset(
        y=physics_train(
            dinv.utils.load_example("leaves.png", device=device)[..., :64, :64]
        )
    )
)

max_epochs = 20
trainer = dinv.Trainer(
    model=model,
    physics=physics_train,
    eval_interval=5,
    ckp_interval=max_epochs - 1,
    metrics=losses[0],
    early_stop=True,
    device=device,
    losses=losses,
    epochs=max_epochs,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-5),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
)

finetuned_model = trainer.train()

# %%
# We can now use the fine-tuned model to reconstruct the image from the measurement `y`.

with torch.no_grad():
    x_hat_ft = finetuned_model(y, physics)

# Show results
dinv.utils.plot(
    {
        "Original": x,
        f"Measurement\n PSNR {psnr(y, x).item():.2f}dB": y,
        f"Zero-shot reconstruction\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
        f"Fine-tuned reconstruction\n PSNR {psnr(x_hat_ft, x).item():.2f}dB": x_hat_ft,
    },
)

# %%
# :References:
#
# .. footbibliography::
