r"""
Super-resolution with GSPnP RED.
====================================================================================================

Hurault, S., Leclaire, A., & Papadakis, N. 
Gradient Step Denoiser for convergent Plug-and-Play. 
In International Conference on Learning Representations.
"""

import numpy as np
import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import ScoreDenoiser
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.training_utils import test
from torchvision import datasets, transforms
from deepinv.utils.parameters import get_GSPnP_params
from deepinv.utils.demo import get_git_root, download_dataset, download_degradation


# Setup paths for data loading, results and checkpoints.
BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reprod
# ucibility of the example.
torch.manual_seed(0)


# Setup the variable to fetch dataset and operators.
method = "GSPnP"
denoiser_name = "gsdrunet"
dataset_name = "CBSD68"
operation = "super-resolution"
dataset_path = ORIGINAL_DATA_DIR / dataset_name
if not dataset_path.exists():
    download_dataset(dataset_name, ORIGINAL_DATA_DIR)
measurement_dir = DATA_DIR / dataset_name / operation


# Use parallel dataloader if using a GPU to fasten training, otherwise, as all computes are on CPU, use synchronous dataloading.
num_workers = 4 if torch.cuda.is_available() else 0


# Parameters of the algorithm to solve the inverse problem
n_images_max = 3  # Maximal number of images to restore from the input dataset
batch_size = 1
noise_level_img = 0.03  # Gaussian Noise standart deviation for the degradation
img_size = 256
n_channels = 3  # 3 for color images, 1 for gray-scale images
early_stop = True  # Stop algorithm when convergence criteria is reached
crit_conv = "cost"  # Convergence is reached when the difference of cost function between consecutive iterates is smaller than thres_conv
thres_conv = 1e-5
backtracking = True  # use backtraking to automatically adjust the stepsize
factor = 2  # down-sampling factor
use_bicubic_init = False  # Use bicobic interpolation to initialize the algorithm


# Logging parameters
verbose = True
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR
wandb_vis = False  # plot curves and images in Weight&Bias
plot_images = True  # plot results
save_images = True  # save images in RESULTS_DIR


# Generate the degradation operator.
kernel_index = 2  # which kernel to chose
kernel_path = DEG_DIR / "kernels" / "kernels_12.npy"
if not kernel_path.exists():
    download_degradation("kernels_12.npy", DEG_DIR / "kernels")
kernels = np.load(kernel_path)
filter_torch = torch.from_numpy(kernels[kernel_index]).unsqueeze(0).unsqueeze(0)
p = dinv.physics.Downsampling(
    img_size=(n_channels, img_size, img_size),
    factor=factor,
    filter=filter_torch,
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# load specific parameters for GSPnP
lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
    operation, noise_level_img, kernel_index
)

params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}

# Select the data fidelity term
data_fidelity = L2()

# Specify the Denoising prior
ckpt_path = CKPT_DIR / "gsdrunet.ckpt"
model_spec = {
    "name": denoiser_name,
    "args": {
        "in_channels": n_channels,
        "out_channels": n_channels,
        "pretrained": str(ckpt_path) if ckpt_path.exists() else "download",
        "train": False,
        "device": dinv.device,
    },
}
# The prior g needs to be a dictionary with specified "g" and/or proximal operator "prox_g" and/or gradient "grad_g".
# For RED image restoration, the denoiser replaces "grad_g".

denoiser = ScoreDenoiser(model_spec, sigma_normalize=False)
prior = {"grad_g": denoiser, "g": denoiser.denoiser.potential}

# Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
generated_datasets_paths = dinv.datasets.generate_dataset(
    train_dataset=dataset,
    test_dataset=None,
    physics=p,
    device=dinv.device,
    save_dir=measurement_dir,
    max_datapoints=n_images_max,
    num_workers=num_workers,
)
dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_paths[0], train=True)
dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# By default the algorithm is initialized with the adjoint of the degradation matrix applied to the degraded image.
# For custom initialization, we need to write a a function of the degraded image.
if use_bicubic_init:
    custom_init = lambda y: torch.nn.functional.interpolate(
        y, scale_factor=factor, mode="bicubic"
    )
else:
    custom_init = None

# instanciate the algorithm class to solve the IP problem.
model = optim_builder(
    algo_name="PGD",
    prior=prior,
    g_first=True,
    data_fidelity=data_fidelity,
    params_algo=params_algo,
    early_stop=early_stop,
    max_iter=max_iter,
    crit_conv=crit_conv,
    thres_conv=thres_conv,
    backtracking=backtracking,
    return_dual=True,
    verbose=verbose,
    return_metrics=plot_metrics,
    custom_init=custom_init,
)


# Evaluate the model on the problem.
test(
    model=model,
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation / dataset_name,
    plot_metrics=plot_metrics,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
