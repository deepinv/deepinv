"""
Image deblurring with custom deep explicit prior function.
"""

import numpy as np
import deepinv as dinv
import hdf5storage
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import Optim
from deepinv.training_utils import test
from torchvision import datasets, transforms
from deepinv.utils.demo import get_git_root, download_dataset, download_degradation


# create a nn.Module class to parametrize the custom prior
class L2Prior(nn.Module):
    def __init__(self, prior_params=None):
        super(L2Prior, self).__init__()

    def forward(self, x, g_param):
        return torch.norm(x.view(x.shape[0], -1), p=2, dim=-1)


# Setup paths for data loading, results and checkpoints.
BASE_DIR = Path(get_git_root())
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# Setup the variable to fetch dataset and operators.
method = "L2_prior"
dataset_name = "set3c"
operation = "deblur"
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
max_iter = 500  # Maximum number of iterations
stepsize_prox_inter = 1.0  # Stepsize used for gradient descent calculation of the prox of the custom prior.
max_iter_prox_inter = 50  # Maximum number of iterations for gradient descent calculation of the prox of the custom prior.
tol_prox_inter = 1e-3  # Convergence criteria for gradient descent calculation of the prox of the custom prior.

# Logging parameters
verbose = True
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR
wandb_vis = True  # plot curves and images in Weight&Bias
plot_images = True  # plot results
save_images = True  # save images in RESULTS_DIR


# Generate a motion blur operator.
kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
kernel_path = DEG_DIR / "kernels" / "Levin09.mat"
if not kernel_path.exists():
    download_degradation("Levin09.mat", DEG_DIR / "kernels")
kernels = hdf5storage.loadmat(str(kernel_path))["kernels"]
filter_np = kernels[0, kernel_index].astype(np.float64)
filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)
# The BlurFFT instance from physics enables to compute efficently backward operators with Fourier transform.
p = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=filter_torch,
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)


# Select the data fidelity term
data_fidelity = L2()


# Specify the custom prior
prior = {"g": L2Prior()}


# Specific parameters for restoration with the given prior (Note that these parameters have not been optimized here)
params_algo = {"stepsize": 1, "g_param": 1.0, "lamb": 1}


# Desine the cost function that is minimized by the algorithm.
F_fn = lambda x, cur_params, y, physics: params_algo["lamb"][0] * data_fidelity.f(
    physics.A(x), y
) + prior["g"][0](x, cur_params["g_param"])

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

# instanciate the algorithm class to solve the IP problem.
model = Optim(
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
    F_fn=F_fn,
    verbose=verbose,
    return_metrics=plot_metrics,
)


# Evaluate the model on the problem.
test(
    model=model,
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot_images=plot_images,
    save_images=save_images,
    plot_input=True,
    save_folder=RESULTS_DIR / method / operation / dataset_name,
    plot_metrics=plot_metrics,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
