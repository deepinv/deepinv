import numpy as np
import deepinv as dinv
import hdf5storage
import torch
import os
from torch.utils.data import DataLoader
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import *
from deepinv.optim.optimizers import *
from deepinv.training_utils import test
from torchvision import datasets, transforms
from deepinv.utils.parameters import get_ProxPnP_params

torch.manual_seed(0)

num_workers = (
    4 if torch.cuda.is_available() else 0
)  # set to 0 if using small cpu, else 4
train = False
denoiser_name = "proxdrunet"
ckpt_path = "../checkpoints/Prox_DRUNet.ckpt"
if not os.path.exists(ckpt_path):
    ckpt_path = None
batch_size = 1
n_channels = 3
pretrain = True
problem = "super_resolution"
G = 1
img_size = 256
dataset = "set3c"
dataset_path = f"../datasets/{dataset}/images"
save_dir = f"../datasets/{dataset}/{problem}/"
noise_level_img = 0.01
crit_conv = "cost"
thres_conv = 1e-5
early_stop = False
verbose = True
k_index = 2
factor = 2
algo_name = "PGD"
plot_metrics = True

# TODO : add kernel downloading code
kernels = hdf5storage.loadmat("../kernels/kernels_12.mat")["kernels"]
filter_np = kernels[0, k_index].astype(np.float64)
filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)

p = dinv.physics.Downsampling(
    img_size=(n_channels, img_size, img_size),
    factor=factor,
    filter=filter_torch,
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)
data_fidelity = L2()

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
dinv.datasets.generate_dataset(
    train_dataset=dataset,
    test_dataset=None,
    physics=p,
    device=dinv.device,
    save_dir=save_dir,
    max_datapoints=3,
    num_workers=num_workers,
)
dataset = dinv.datasets.HDF5Dataset(path=f"{save_dir}/dinv_dataset0.h5", train=True)
dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

lamb, sigma_denoiser, stepsize, max_iter, alpha_denoiser = get_ProxPnP_params(
    algo_name, noise_level_img
)

model_spec = {
    "name": denoiser_name,
    "args": {
        "alpha": alpha_denoiser,
        "in_channels": n_channels + 1,
        "out_channels": n_channels,
        "pretrained": ckpt_path,
        "train": False,
        "device": dinv.device,
    },
}

params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
prior = {"prox_g": Denoiser(model_spec)}
model = Optim(
    algo_name=algo_name,
    prior=prior,
    g_first=False,
    data_fidelity=data_fidelity,
    params_algo=params_algo,
    early_stop=early_stop,
    max_iter=max_iter,
    crit_conv=crit_conv,
    thres_conv=thres_conv,
    backtracking=False,
    F_fn=None,
    return_dual=False,
    verbose=True,
    return_metrics=plot_metrics,
)

test(
    model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot_images=True,
    plot_input=True,
    save_folder="../results/",
    verbose=verbose,
    plot_metrics=plot_metrics,
    wandb_vis=True,
)
