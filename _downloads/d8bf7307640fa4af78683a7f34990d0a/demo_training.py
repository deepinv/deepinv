r"""
Training a reconstruction model
====================================================================================================

This example provides a very simple quick start introduction to training reconstruction networks with
DeepInverse for solving imaging inverse problems.

Training requires these components, all of which you can define with DeepInverse:

* A `model` to be trained from :ref:`reconstructors <reconstructors>` or define your own.
* A `physics` from our :ref:`list of physics <physics>`. Or, :ref:`bring your own physics <sphx_glr_auto_examples_basics_demo_custom_dataset.py>`.
* A `dataset` of images and/or measurements from :ref:`datasets <datasets>`. Or, :ref:`bring your own dataset <sphx_glr_auto_examples_basics_demo_custom_dataset.py>`.
* A `loss` from our :ref:`loss functions <loss>`.
* A `metric` from our :ref:`metrics <metric>`.

Here, we demonstrate a simple experiment of training a UNet
on an inpainting task on the Urban100 dataset of natural images.

"""

import deepinv as dinv
import torch

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=device).manual_seed(0)

# %%
# Setup
# -----
#
# First, define the physics that we want to train on.
#

physics = dinv.physics.Inpainting((1, 64, 64), mask=0.8, device=device, rng=rng)

# %%
# Then define the dataset. Here we simulate a dataset of measurements from Urban100.
#
# .. tip::
#     See :ref:`datasets <datasets>` for types of datasets DeepInverse supports: e.g. paired, ground-truth-free,
#     single-image...
#

from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Grayscale

dataset = dinv.datasets.Urban100HR(
    ".",
    download=True,
    transform=Compose([ToTensor(), Grayscale(), Resize(256), CenterCrop(64)]),
)

train_dataset, test_dataset = torch.utils.data.random_split(
    torch.utils.data.Subset(dataset, range(50)), (0.8, 0.2)
)

dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=".",
    batch_size=1,
)

train_dataloader = torch.utils.data.DataLoader(
    dinv.datasets.HDF5Dataset(dataset_path, train=True), shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dinv.datasets.HDF5Dataset(dataset_path, train=False), shuffle=False
)

# %%
# Visualize a data sample:
#

x, y = next(iter(test_dataloader))
dinv.utils.plot({"Ground truth": x, "Measurement": y, "Mask": physics.mask})


# %%
# For the model we use an artifact removal model, where
# :math:`\phi_{\theta}` is a U-Net
#
# .. math::
#
#     f_{\theta}(y) = \phi_{\theta}(A^{\top}(y))
#

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(1, 1, scales=2, batch_norm=False).to(device)
)

# %%
# Train the model
# ----------------------------------------------------------------------------------------
# We train the model using the :class:`deepinv.Trainer` class,
# which cleanly handles all steps for training.
#
# We perform supervised learning and use the mean squared error as loss function.
# See :ref:`losses <loss>` for all supported state-of-the-art loss functions.
#
# We evaluate using the PSNR metric.
# See :ref:`metrics <metric>` for all supported metrics.
#
# .. note::
#
#       In this example, we only train for a few epochs to keep the training time short.
#       For a good reconstruction quality, we recommend to train for at least 100 epochs.
#


trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=5,
    losses=dinv.loss.SupLoss(metric=dinv.metric.MSE()),
    metrics=dinv.metric.PSNR(),
    device=device,
    plot_images=True,
    show_progress_bar=False,
)

_ = trainer.train()


# %%
# Test the network
# --------------------------------------------
# We can now test the trained network using the :func:`deepinv.test` function.
#
# The testing function will compute metrics and plot and save the results.

trainer.test(test_dataloader)
