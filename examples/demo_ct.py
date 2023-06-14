import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from deepinv.utils.demo import get_git_root, download_dataset
from deepinv.training_utils import train, test
from deepinv.datasets.ct100 import CTData

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


# Generate a degradation operator, for CT here
p = dinv.physics.Tomography(img_width=img_size, radon_view=radon_view)

# Setup the variable to fetch dataset and operators.
operation = "tomography"  # "tomography"
train_dataset_name = "CT100"  # "CT100"
val_dataset_name = "CT100"
train_dataset_path = ORIGINAL_DATA_DIR / train_dataset_name
test_dataset_path = ORIGINAL_DATA_DIR / val_dataset_name
if not train_dataset_path.exists():
    download_dataset(train_dataset_path, ORIGINAL_DATA_DIR)
if not test_dataset_path.exists():
    download_dataset(test_dataset_path, ORIGINAL_DATA_DIR)
measurement_dir = DATA_DIR / train_dataset_name / operation

if train_dataset_path.exits():
    download_dataset(train)

# Generate training and evaluation datasets in HDF5 folders and load them.
my_dataset_name = "demo_training_ct"
generated_datasets_path = measurement_dir / str(my_dataset_name + "0.h5")
if not generated_datasets_path.exists():
    train_dataset = CTData("train")
    test_dataset = CTData("test")

    generated_datasets_paths = dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=p,
        device=dinv.device,
        save_dir=measurement_dir,
        max_datapoints=n_images_max,
        num_workers=num_workers,
        dataset_filename=str(my_dataset_name),
    )
train_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=False)
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
