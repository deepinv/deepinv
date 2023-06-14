r"""
Self-supervised learning with Equivariant Imaging for MRI.
====================================================================================================

This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.

The equivariant imaging loss is presented in `"Equivariant Imaging: Learning Beyond the Range Space"
<http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.pdf>`_.

"""


import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from deepinv.utils.demo import get_git_root, load_dataset, load_degradation
from deepinv.training_utils import train, test
from deepinv.utils.demo import CTData

# Setup paths for data loading, results and checkpoints.
BASE_DIR = Path(get_git_root())
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# Use parallel dataloader if using a GPU to fasten training, otherwise, as all computes are on CPU, use synchronous dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

# Parameters
epochs = 100  # choose training epochs
learning_rate = 5e-4
train_batch_size = 2
test_batch_size = 2
img_size = 256
n_channels = 1  # 3 for color images, 1 for gray-scale images
n_images_max = 100  # maximal number of images used for training
radon_view = 100  # number of views (angles) for Radon transformation

# Logging parameters
verbose = True
wandb_vis = True  # plot curves and images in Weight&Bias


# Setup the variable to fetch dataset and operators.
operation = "tomography"  # "tomography"
train_dataset_name = "CT100"  # "CT100"
val_dataset_name = "CT100"


train_dataset = load_dataset(train_dataset_name, ORIGINAL_DATA_DIR, train=True)
test_dataset = load_dataset(train_dataset_name, ORIGINAL_DATA_DIR, train=False)

# Generate a degradation operator, for CT here
physics = dinv.physics.Tomography(img_width=img_size, radon_view=radon_view)


# Generate training and evaluation datasets in HDF5 folders and load them.
my_dataset_name = "demo_training_ct"
measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=dinv.device,
    save_dir=measurement_dir,
    max_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)
train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)
train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)


# choose training losses
losses = []
losses.append(dinv.loss.MCLoss(metric=dinv.metric.mse()))  # self-supervised loss
losses.append(dinv.loss.EILoss(transform=dinv.transform.Rotate(n_trans=1)))

# choose backbone model
backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=3).to(dinv.device)

# choose a reconstruction architecture
model = dinv.models.ArtifactRemoval(backbone)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

# train the network
train(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=p,
    optimizer=optimizer,
    device=dinv.device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
    log_interval=2,
    eval_interval=2,
    ckp_interval=2,
)

# %%
# Test the network
# --------------------------------------------
#
#

plot_images = True
save_images = True
method = "artifact_removal"

test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=dinv.device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
