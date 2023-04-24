import numpy as np
import deepinv as dinv
import hdf5storage
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import Unfolded
from deepinv.training_utils import test
from torchvision import datasets, transforms
from deepinv.utils.demo import get_git_root, download_dataset, download_degradation

# Setup paths for data loading, results and checkpoints.
BASE_DIR = Path(get_git_root())
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"


# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)


# Setup the variable to fetch dataset and operators.
denoiser_name = "dncnn"
train_dataset_name = "set3c"
val_dataset_name = "CBSD68"
operation = "super-resolution"


dataset_path = ORIGINAL_DATA_DIR / dataset_name
if not dataset_path.exists():
    download_dataset(dataset_name, ORIGINAL_DATA_DIR)
measurement_dir = DATA_DIR / dataset_name / operation





num_workers = (
    4 if torch.cuda.is_available() else 0
)  # set to 0 if using small cpu, else 4
problem = "deblur"
G = 1
denoiser_name = "dncnn"
depth = 7
ckpt_path = None
algo_name = "PGD"
dataset_path = f"../datasets/"
train_dataset_name = "drunet"
test_dataset_name = "CSBD68"
noise_level_img = 0.03
lamb = 10
max_iter = 5
verbose = True
early_stop = False
n_channels = 3
pretrain = False
epochs = 100
img_size = 32
batch_size = 32
max_datapoints = 100
anderson_acceleration = False

wandb_vis = False

if wandb_vis:
    wandb.init(project="unrolling")

kernels = hdf5storage.loadmat("../kernels/Levin09.mat")["kernels"]
filter_np = kernels[0, 1].astype(np.float64)
filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)
p = dinv.physics.BlurFFT(
    img_size=(3, img_size, img_size),
    filter=filter_torch,
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)
data_fidelity = L2()


if not os.path.exists(
    f"{dataset_path}/artificial/{train_dataset_name}/dinv_dataset0.h5"
):
    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(img_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    train_input_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/{train_dataset_name}/", transform=train_transform
    )
    test_input_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/{test_dataset_name}/", transform=val_transform
    )
    dinv.datasets.generate_dataset(
        train_dataset=train_input_dataset,
        test_dataset=test_input_dataset,
        physics=p,
        device=dinv.device,
        save_dir=f"{dataset_path}/artificial/{train_dataset_name}/",
        max_datapoints=max_datapoints,
        num_workers=num_workers,
    )

train_dataset = dinv.datasets.HDF5Dataset(
    path=f"{dataset_path}/artificial/{train_dataset_name}/dinv_dataset0.h5", train=True
)
eval_dataset = dinv.datasets.HDF5Dataset(
    path=f"{dataset_path}/artificial/{train_dataset_name}/dinv_dataset0.h5", train=False
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

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

# prior = {'prox_g': Denoiser(model_spec)}
prior = {"prox_g": [Denoiser(model_spec) for i in range(max_iter)]}
params_algo = {
    "stepsize": [1.0] * max_iter,
    "g_param": [0.01] * max_iter,
    "lambda": 1.0,
}
trainable_params = ["stepsize", "g_param"]
model = Unfolded(
    algo_name,
    params_algo=params_algo,
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, " is trainable")

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

train(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=p,
    optimizer=optimizer,
    device=dinv.device,
    ckp_interval=int(epochs / 2.0),
    save_path=f"../checkpoints/tests/demo_unrolled",
    plot_images=False,
    verbose=True,
    wandb_vis=wandb_vis,
    debug=False,
)
