"""
Inference and fine-tune Reconstruct Anything Model (RAM) foundation model
====================================================================================================

This example shows how to perform inference on and fine-tune the RAM foundation model to solve inverse problems.

:class:`RAM <deepinv.models.RAM>` :footcite:t:`terris2025reconstruct` is a model that has been trained to work on a large
variety of linear image reconstruction tasks and datasets (deblurring, inpainting, denoising, tomography, MRI, etc.).

See :ref:`sphx_glr_auto_examples_basics_demo_pretrained_model.py` for more examples of RAM on different datasets and physics. TODO copy over to here!

.. tip::

    * Want to use your own dataset? See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py`
    * Want to use your own physics? See :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py`

"""

# %%
# 1. Zero-shot inference
# ----------------------
#
# First, let's evaluate the zero-shot inference performance of the foundation model.
# Here we use an example of inpainting removing 70% of pixels with 5% Gaussian noise:

import torch
import deepinv as dinv

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

model = dinv.models.RAM(device=device, pretrained=True)

# Load image
x = dinv.utils.load_example("butterfly.png", img_size=(127, 129)).to(device)

# Define forward operator
physics = dinv.physics.Inpainting(
    img_size=(3, 127, 129),
    mask=0.3,
    noise_model=dinv.physics.GaussianNoise(0.05),
    device=device,
)

# Generate measurement
y = physics(x)

# Run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)

# Show results
psnr = dinv.metric.PSNR()
dinv.utils.plot(
    {
        "Original": x,
        f"Measurement\n PSNR {psnr(y, x).item():.2f}dB": y,
        f"Reconstruction\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    },
    figsize=(8, 3),
)

# %%
# This model was also trained on various denoising problems, in particular on Poisson-Gaussian denoising.

sigma, gain = 0.2, 0.5
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain),
)

# Generate measurement
y = physics(x)

# Run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)
    # or alternatively, we can use the model without physics:
    # x_hat = model(y, sigma=sigma, gain=gain)

# Show results
dinv.utils.plot(
    {
        "Original": x,
        f"Measurement\n PSNR {psnr(y, x).item():.2f}dB": y,
        f"Reconstruction\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    },
    figsize=(8, 3),
)

# %%
# 2. Fine-tuning
# --------------
# As with all models, there may be a drop in performance when used zero-shot on problems or data outside those seen during training.
#
# For instance, RAM is not trained on image demosaicing:

physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=dinv.physics.PoissonNoise(0.1), device=device
)

# Generate measurement
y = physics(x)

# Run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)

# Show results
dinv.utils.plot(
    {
        "Original": x,
        f"Measurement\n PSNR {psnr(y, x).item():.2f}dB": y,
        f"Reconstruction\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    },
    figsize=(8, 3),
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

physics = dinv.physics.Demosaicing(
    img_size=(3, 64, 64),
    noise_model=dinv.physics.PoissonNoise(0.1, clip_positive=True),
    device=device,
)

x = x[..., :64, :64]
y = physics(x)

losses = [
    dinv.loss.R2RLoss(),
    dinv.loss.EILoss(dinv.transform.Shift(shift_max=0.4), weight=0.1),
]

dataset = dinv.datasets.TensorDataset(y=y)
train_dataloader = torch.utils.data.DataLoader(dataset)

# %%
# We fine-tune using early stopping using a validation set, again without ground truth.
# We use a small patch of another set of measurements.

eval_dataloader = torch.utils.data.DataLoader(
    dinv.datasets.TensorDataset(
        y=physics(dinv.utils.load_example("leaves.png")[..., :64, :64].to(device))
    )
)

max_epochs = 20
trainer = dinv.Trainer(
    model=model,
    physics=physics,
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
    x_hat = finetuned_model(y, physics=physics)

# Show results
dinv.utils.plot(
    {
        "Original": x,
        f"Measurement\n PSNR {psnr(y, x).item():.2f}dB": y,
        f"Fine-tuned reconstruction\n PSNR {psnr(x_hat, x).item():.2f}dB": x_hat,
    },
    figsize=(8, 3),
)
