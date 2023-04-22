import numpy as np
import deepinv as dinv
import hdf5storage
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import *
from deepinv.optim.optimizers import *
from deepinv.training_utils import test
from torchvision import datasets, transforms


# create a nn.Module class to parametrize the custom prior
class L2Prior(nn.Module):
    def __init__(self, prior_params=None):
        super(L2Prior, self).__init__()

    def forward(self, x, g_param):
        return torch.norm(x.view(x.shape[0], -1), p=2, dim=-1)


torch.manual_seed(0)
num_workers = (
    4 if torch.cuda.is_available() else 0
)  # set to 0 if using small cpu, else 4
train = False
batch_size = 1
problem = "deblur"
G = 1
img_size = 256
dataset = "set3c"
dataset_path = f"../datasets/{dataset}/images"
save_dir = f"../datasets/{dataset}/{problem}/"
noise_level_img = 0.03
crit_conv = "residual"
thres_conv = 1e-3
early_stop = True
verbose = True
k_index = 1
plot_metrics = True
max_iter = 200

kernels = hdf5storage.loadmat("../kernels/Levin09.mat")["kernels"]
filter_np = kernels[0, k_index].astype(np.float64)
filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)
p = dinv.physics.BlurFFT(
    img_size=(3, img_size, img_size),
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

params_algo = {"stepsize": 1, "g_param": 1.0, "lamb": 1}

stepsize_prox_inter = 1.0
max_iter_prox_inter = 50
tol_prox_inter = 1e-3

prior = {"g": L2Prior()}

F_fn = lambda x, cur_params, y, physics: params_algo["lamb"][0] * data_fidelity.f(
    physics.A(x), y
) + prior["g"][0](x, cur_params["g_param"])
model = Optim(
    algo_name="PGD",
    prior=prior,
    g_first=False,
    data_fidelity=data_fidelity,
    params_algo=params_algo,
    early_stop=early_stop,
    max_iter=max_iter,
    crit_conv=crit_conv,
    thres_conv=thres_conv,
    backtracking=True,
    F_fn=F_fn,
    return_dual=False,
    verbose=True,
    return_metrics=plot_metrics,
    stepsize_prox_inter=stepsize_prox_inter,
    max_iter_prox_inter=max_iter_prox_inter,
    tol_prox_inter=tol_prox_inter,
)

test(
    model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot_images=True,
    plot_input=True,
    save_folder="../results/",
    plot_metrics=plot_metrics,
    verbose=verbose,
    wandb_vis=True,
)
