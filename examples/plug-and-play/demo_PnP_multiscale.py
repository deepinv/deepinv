"""
Multiscale Plug-and-Play
====================================================================================================

This example describe the usage of multiple scales with Plug-and-Play schemes.
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import deepinv.physics.blur
from deepinv.physics.blur import gaussian_blur
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.training import test
from torchvision import transforms
from deepinv.utils.demo import load_dataset

from deepinv.physics import Inpainting, GaussianNoise, Blur

from deepinv.optim.optim_iterators import PGDIteration

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"

# %%
# Define the PnP setting.
# ----------------------------------------------------------------------------------------
# In this example, we use the Set3C dataset

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

val_transform = transforms.Compose([transforms.ToTensor()])

dataset_name = "set3c"
dataset = load_dataset(dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)
dataloader = DataLoader(dataset, batch_size=3, num_workers=1, shuffle=False)

# set the data-fidelity and the prior
data_fidelity = L2()
noise_level = 0.1
noise_model = GaussianNoise(sigma=noise_level)
set3c_img_size = (3, 256, 256)  # set3c contains 256x256 rgb images
physics = Inpainting(
    tensor_size=set3c_img_size, mask=0.5, noise_model=noise_model, device=device
)

prior = PnP(denoiser=DRUNet(pretrained="download", device=device))

# set values of the PnP parameters
max_iter = 200
params_algo = {"stepsize": 1.0, "g_param": 0.05}


# %%
# Classical case : single scale PnP.
# ----------------------------------------------------------------------------------------

# create the iterative algorithm model
model = optim_builder(
    iteration=PGDIteration(),
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=max_iter,
    params_algo=params_algo,
)

# Set the model to evaluation mode since we do not require training.
model.eval()

# run the model on chosen dataset
test(
    model=model,
    test_dataloader=dataloader,
    physics=physics,
    metrics=[dinv.metric.PSNR()],
    device=device,
    online_measurements=True,
    plot_images=True,
    plot_convergence_metrics=True,
    verbose=True,
)

# %%
# Multiscale case : use a coarse setting to initialize the fine setting.
# ----------------------------------------------------------------------------------------
# The PnP algorithm iterates on a coarse scale to obtain a first estimate.
# This estimate is then upsampled and used as initialization in the fine scale.
# As shown in the result, the reconstruction quality significantly improves.


# define the function which will be used to initialize the fine setting.
def custom_init(y, physics, F_fn=None):
    p_coarse = physics.to_coarse()
    y_coarse = physics.downsample_measurement(y, p_coarse)
    params_algo = {"stepsize": 1.0, "g_param": 0.05}

    model = optim_builder(
        iteration=PGDIteration(),
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=False,
        max_iter=8,
        params_algo=params_algo,
    )

    x_coarse = model(y_coarse, p_coarse)

    # upsample coarse estimation
    x_up = physics.upsample_signal(x_coarse)
    return {"est": [x_up]}


# define the multiscale model by setting the "custom_init" field
model = optim_builder(
    iteration=PGDIteration(),
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=max_iter,
    params_algo=params_algo,
    custom_init=custom_init,
)

# Set the model to evaluation mode since we do not require training.
model.eval()

# run the multiscale algorithm exactly as any other algorithms
test(
    model=model,
    test_dataloader=dataloader,
    physics=physics,
    metrics=[dinv.metric.PSNR()],
    device=device,
    online_measurements=True,
    plot_images=True,
    plot_convergence_metrics=True,
    verbose=True,
)
