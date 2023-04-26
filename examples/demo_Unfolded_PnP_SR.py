import numpy as np
import deepinv as dinv
import hdf5storage
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import L2
from deepinv.unfolded.unfolded import Unfolded
from deepinv.training_utils import train
from torchvision import datasets, transforms
from deepinv.utils.demo import get_git_root, download_dataset, download_degradation

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


# Degradation parameters
img_size = 128
n_channels = 3  # 3 for color images, 1 for gray-scale images
factor = 2
noise_level_img = 0.03


# Generate the gaussian blur downsampling operator.
p = dinv.physics.Downsampling(
    img_size=(n_channels, img_size, img_size),
    factor=factor,
    mode="gauss",
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)


# Setup the variable to fetch dataset and operators.
operation = "super-resolution"
train_dataset_name = "DRUNET"
val_dataset_name = "CBSD68"
train_dataset_path = ORIGINAL_DATA_DIR / train_dataset_name
test_dataset_path = ORIGINAL_DATA_DIR / val_dataset_name
if not train_dataset_path.exists():
    download_dataset(train_dataset_path, ORIGINAL_DATA_DIR)
if not test_dataset_path.exists():
    download_dataset(test_dataset_path, ORIGINAL_DATA_DIR)
measurement_dir = DATA_DIR / train_dataset_name / operation

# training parameters
epochs = 10  # choose training epochs
learning_rate = 5e-4
train_batch_size = 32
test_batch_size = 32
n_images_max = 1000  # maximal number of images used for training

# Logging parameters
verbose = True
wandb_vis = True  # plot curves and images in Weight&Bias


# Generate training and evaluation datasets in HDF5 folders and load them.
test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
my_dataset_name = "demo_unfolded_sr"
generated_datasets_path = measurement_dir / str(my_dataset_name + "0.h5")
if not generated_datasets_path.exists():
    train_dataset = datasets.ImageFolder(
        root=train_dataset_path, transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=test_dataset_path, transform=test_transform
    )
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


# Select the data fidelity term
data_fidelity = L2()


# prior parameters
denoiser_name = "dncnn"
depth = 7
ckpt_path = None

# algorithm parameters
max_iter = 5  # number of unfolded layers
lamb = 1.0  # initialization of the regularization parameter
# For both 'stepsize' and 'g_param', if initialized with a table of lenght max_iter, then a distinct stepsize/g_param value is trained for each iteration.
# For fixed trained 'stepsize' and 'g_param' values across iterations, initialize them with a single float.
stepsize = [1.0] * max_iter  # ininitialization of the stepsizes.
sigma_denoiser = [0.01] * max_iter  # initialization of the denoiser parameters
params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
}
trainable_params = [
    "lambda",
    "stepsize",
    "g_param",
]  # define which parameters from 'params_algo' are trainable

# Set up the trainable denoising prior
model_spec = {
    "name": denoiser_name,
    "args": {
        "in_channels": n_channels,
        "out_channels": n_channels,
        "depth": depth,
        "pretrained": ckpt_path,
        "train": True,
        "device": dinv.device,
    },
}
# If the prior dict value is initialized with a table of lenght max_iter, then a distinct model is trained for each iteration.
# For fixed trained model prior across iterations, initialize with a single model.
prior = {
    "prox_g": Denoiser(model_spec)
}  # here the prior model is common for all iterations


# Define the unfolded trainable model.
model = Unfolded(
    "DRS",
    params_algo=params_algo,
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    verbose=verbose,
)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))

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
)
