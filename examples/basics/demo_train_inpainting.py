r"""
Training a reconstruction network.
====================================================================================================

This example shows how to train a simple reconstruction network for an image
inpainting inverse problem.

"""

import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset

# %%
# Setup paths for data loading and results.
# --------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# --------------------------------------------------------------------------------------------
# In this example, we use the CBSD68 dataset for training and the set3c dataset for testing.
# We work with images of size 32x32 if no GPU is available, else 128x128.


operation = "inpainting"
train_dataset_name = "CBSD68"
test_dataset_name = "set3c"
img_size = 128 if torch.cuda.is_available() else 32

test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)

train_dataset = load_dataset(train_dataset_name, train_transform)
test_dataset = load_dataset(test_dataset_name, test_transform)

# %%
# Define forward operator and generate dataset
# --------------------------------------------------------------------------------------------
# We define an inpainting operator that randomly masks pixels with probability 0.5.
#
# A dataset of pairs of measurements and ground truth images is then generated using the
# :func:`deepinv.datasets.generate_dataset` function.
#
# Once the dataset is generated, we can load it using the :class:`deepinv.datasets.HDF5Dataset` class.

n_channels = 3  # 3 for color images, 1 for gray-scale images
probability_mask = 0.5  # probability to mask pixel

# Generate inpainting operator
physics = dinv.physics.Inpainting(
    tensor_size=(n_channels, img_size, img_size), mask=probability_mask, device=device
)


# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0
n_images_max = (
    1000 if torch.cuda.is_available() else 50
)  # maximal number of images used for training
my_dataset_name = "demo_training_inpainting"
measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)


train_batch_size = 32 if torch.cuda.is_available() else 1
test_batch_size = 32 if torch.cuda.is_available() else 1

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Set up the reconstruction network
# --------------------------------------------------------
# We use a simple inversion architecture of the form
#
#      .. math::
#
#               f_{\theta}(y) = \phi_{\theta}(A^{\top}(y))
#
# where the linear reconstruction :math:`A^{\top}y` is post-processed by a U-Net network :math:`\phi_{\theta}` is a
# neural network with trainable parameters :math:`\theta`.


# choose backbone model
backbone = dinv.models.UNet(
    in_channels=3, out_channels=3, scales=3, batch_norm=False
).to(device)

# choose a reconstruction architecture
model = dinv.models.ArtifactRemoval(backbone)

# %%
# Train the model
# ----------------------------------------------------------------------------------------
# We train the model using the :class:`deepinv.Trainer` class.
#
# We perform supervised learning and use the mean squared error as loss function. This can be easily done using the
# :class:`deepinv.loss.SupLoss` class.
#
# .. note::
#
#       In this example, we only train for a few epochs to keep the training time short.
#       For a good reconstruction quality, we recommend to train for at least 100 epochs.
#


verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

epochs = 4  # choose training epochs
learning_rate = 5e-4

# choose training losses
losses = dinv.loss.SupLoss(metric=dinv.metric.MSE())

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))
trainer = dinv.Trainer(
    model,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    optimizer=optimizer,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
)
model = trainer.train()

# %%
# Test the network
# --------------------------------------------
# We can now test the trained network using the :func:`deepinv.test` function.
#
# The testing function will compute test_psnr metrics and plot and save the results.

trainer.test(test_dataloader)
