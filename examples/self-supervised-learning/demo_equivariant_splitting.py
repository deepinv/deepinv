r"""
Self-supervised learning with Equivariant Splitting
===================================================

This example shows you how to train a reconstruction network in a fully self-supervised way, i.e., using measurement data only.

The equivariant splitting loss is presented in :footcite:t:`sechaud26Equivariant`.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv

# %%
# Setup paths for data loading and results.
# ---------------------------------------------------------------
#

import numpy as np
import random

random.seed(0)  # set random seed for reproducibility
np.random.seed(0)  # set random seed for reproducibility
torch.manual_seed(0)  # set random seed for reproducibility
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------
# In this example, we use a mini demo subset of the single-coil `FastMRI dataset <https://fastmri.org/>`_
# as the base image dataset, consisting of 2 knee images of size 320x320.
#
# .. seealso::
#
#   Datasets :class:`deepinv.datasets.FastMRISliceDataset` :class:`deepinv.datasets.SimpleFastMRISliceDataset`
#       We provide convenient datasets to easily load both raw and reconstructed FastMRI images.
#       You can download more data on the `FastMRI site <https://fastmri.med.nyu.edu/>`_.
#
# .. important::
#
#    By using this dataset, you confirm that you have agreed to and signed the `FastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.
#
# .. note::
#
#       We reduce to the size to 128x128 for faster training in the demo.
#

operation = "ES"
channels = 3
img_size = 64

transform = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ]
)

dataset = dinv.datasets.Urban100HR(root="Urban100", transform=transform, download=True)
print(len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [90, 10], generator=torch.Generator().manual_seed(0)
)

# %%
# Generate a dataset of knee images and load it.
# ----------------------------------------------------------------------------------
#
#

# defined physics
physics = dinv.physics.Inpainting(
    mask=0.7, img_size=(channels, img_size, img_size), device=device
)
# physics.noise_model = dinv.physics.GaussianNoise(sigma=0.01)
physics.noise_model = dinv.physics.ZeroNoise()

# Use parallel dataloader if using a GPU to speed up training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0

my_dataset_name = "demo_equivariant_splitting"
measurement_dir = DATA_DIR
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
    overwrite_existing=False,
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the reconstruction network
# ---------------------------------------------------------------
#
# As a (static) reconstruction network, we use an unrolled network
# (half-quadratic splitting) with a trainable denoising prior based on the
# DnCNN architecture which was proposed in MoDL :footcite:t:`aggarwal2018modl`.
# See :class:`deepinv.models.MoDL` for details.
#

# backbone = dinv.models.UNet(
#     in_channels=channels,
#     out_channels=channels,
#     scales=2,
#     bias=True,
#     cat=True,
#     residual=True,
#     batch_norm=True,
# )
# model = MoDL(backbone, num_iter=2).to(device)
# model = dinv.models.ArtifactRemoval(backbone, mode="adjoint").to(device)
model = dinv.models.RAM(pretrained=True).to(device)


# %%
# Set up the training parameters
# --------------------------------------------
# We choose a self-supervised training scheme with two losses: the measurement consistency loss (MC)
# and the equivariant imaging loss (EI).
# The EI loss requires a group of transformations to be defined. The forward model should not be equivariant to
# these transformations :footcite:t:`tachella2023sensing`.
# Here we use the group of 4 rotations of 90 degrees, as the accelerated MRI acquisition is
# not equivariant to rotations (while it is equivariant to translations).
#
# See :ref:`docs <transform>` for full list of available transforms.
#
# .. note::
#
#       We use a pretrained model to reduce training time. You can get the same results by training from scratch
#       for 150 epochs using a larger knee dataset of ~1000 images.

epochs = 30  # choose training epochs
# epochs = 10  # debugging
# epochs = 2
# epochs = 0
learning_rate = 5e-4
weight_decay = 1e-8
batch_size = 4

# A random transformation from the group D4
train_transform = dinv.transform.Rotate(
    n_trans=1, multiples=90, positive=True
) * dinv.transform.Reflect(n_trans=1, dim=[-1])
# # All of the transformations from the group D4
eval_transform = dinv.transform.Rotate(
    n_trans=4, multiples=90, positive=True
) * dinv.transform.Reflect(n_trans=2, dim=[-1])

# consistency_loss = None
consistency_loss = dinv.loss.MCLoss(metric=dinv.metric.MSE())
# consistency_loss = dinv.loss.R2RLoss(alpha=0.2, eval_n_samples=10)
# consistency_loss = dinv.loss.SureGaussianLoss(sigma=physics.noise_model.sigma, tau=physics.noise_model.sigma / 100)
# prediction_loss = None
prediction_loss = dinv.loss.MCLoss(metric=dinv.metric.MSE())
# prediction_loss = dinv.loss.R2RLoss(alpha=0.2, eval_n_samples=10)
# prediction_loss = dinv.loss.SureGaussianLoss(sigma=physics.noise_model.sigma, tau=physics.noise_model.sigma / 100)

mask_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
    img_size=(1, img_size, img_size),
    split_ratio=0.9,
    pixelwise=True,
    device=device,
)

losses = [
    dinv.loss.ESLoss(
        mask_generator=mask_generator,
        consistency_loss=consistency_loss,
        prediction_loss=prediction_loss,
        transform=train_transform,
        eval_transform=eval_transform,
        eval_n_samples=5,
    )
]
# losses = [
#         dinv.loss.SupLoss(metric=dinv.metric.MSE()),
# ]  # DEBUGGING
if len(losses) == 1 and isinstance(losses[0], dinv.loss.ESLoss):
    _ = losses[-1].adapt_model(model)

# choose optimizer and scheduler
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8) + 1)
scheduler = None

# %%
# Train the network
# --------------------------------------------
# To simulate a realistic self-supervised learning scenario, we do not use any supervised metrics for training,
# such as PSNR or SSIM, which require clean ground truth images.
#
# .. tip::
#
#       We can use the same self-supervised loss for evaluation, as it does not require clean images,
#       to monitor the training process (e.g. for early stopping). This is done automatically when `metrics=None` and `early_stop>0` in the trainer.


verbose = True  # print training information

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# Inspect train and test dataloaders
x_train, y_train = next(iter(train_dataloader))
print(f"Train batch shapes: x={x_train.shape}, y={y_train.shape}")
x_test, y_test = next(iter(test_dataloader))
print(f"Test batch shapes: x={x_test.shape}, y={y_test.shape}")

dinv.utils.plot(
    [x_train[0], y_train[0], x_test[0], y_test[0]],
    ["train measurement", "train image", "test measurement", "test image"],
)


# Initialize the trainer
trainer = dinv.Trainer(
    model,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    metrics=[dinv.metric.PSNR()],
    plot_images=False,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    ckp_interval=10,
    no_learning_method="A_adjoint",
)

_ = trainer.train()

if epochs > 0:
    _ = trainer.load_best_model()


# %%
# Test the network
# --------------------------------------------
#
# We now assume that we have access to a small test set of ground-truth images to evaluate the performance of the trained network.
# and we compute the PSNR between the denoised images and the clean ground truth images.
#

trainer.compute_eval_losses = False
trainer.early_stop_on_losses = False
trainer.test(test_dataloader, metrics=dinv.metric.PSNR())

# Show reconstructions on test set
x_test, y_test = next(iter(test_dataloader))
model.eval()
with torch.no_grad():
    x_rec = model(y_test.to(device), physics=physics)
dinv.utils.plot(
    [y_test[0], x_rec[0].cpu(), x_test[0]],
    ["measurement", "reconstruction", "ground truth"],
)

# %%
# :References:
#
# .. footbibliography::
