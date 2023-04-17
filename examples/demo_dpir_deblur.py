from pathlib import Path

import torch
import hdf5storage
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import deepinv as dinv
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import Optim
from deepinv.training_utils import test
from deepinv.utils.parameters import get_DPIR_params


# Setup paths for data loading, results and checkpoints.
ORIGINAL_DATA_DIR = Path("../datasets")
DATA_DIR = Path("../measurments")
RESULTS_DIR = Path("../results")
CKPT_DIR = Path("../checkpoints")


# Set the global random seed from pytorch to ensure
# reproducibility of the example.
torch.manual_seed(0)


# Setup the variable to fetch dataset and operators.
denoiser_name = 'drunet'
dataset = 'set3c'
ckpt_path = CKPT_DIR / 'drunet_color.pth'
dataset_path = ORIGINAL_DATA_DIR / dataset
measurment_dir = DATA_DIR / dataset / 'deblur'

# Use parallel dataloader if using a GPU to fasten training, otherwise, as
# all computes are on CPU, use synchronous dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

# Parameters of the algorithm to solve the inverse problem
batch_size = 1
noise_level_img = 0.03
max_iter = 8
verbose = True
early_stop = False
n_channels = 3
train = False
crit_conv = 'residual'
thres_conv = 1e-4
img_size = 256

lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level_img)
params_algo = {'stepsize': stepsize, 'g_param': sigma_denoiser, 'lambda': lamb}

# Generate a motion blur operator.
kernels = hdf5storage.loadmat('../kernels/Levin09.mat')['kernels']
filter_np = kernels[0, 1].astype(np.float64)
filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)
p = dinv.physics.BlurFFT(
    img_size=(3, img_size, img_size),
    filter=filter_torch,
    device=dinv.device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img)
)

# Select the data fidelity term
data_fidelity = L2()


# Specify the prior
model_spec = {
    'name': denoiser_name,
    'args': {
        'in_channels': n_channels+1,
        'out_channels': n_channels,
        'pretrained': ckpt_path,
        'train': False,
        'device': dinv.device
    }
}
prior = {'prox_g': Denoiser(model_spec)}


# Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
val_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
generated_datasets = dinv.datasets.generate_dataset(
    train_dataset=dataset,
    test_dataset=None,
    physics=p,
    device=dinv.device,
    save_dir=measurment_dir,
    max_datapoints=3,
    num_workers=num_workers
)
dataset = dinv.datasets.HDF5Dataset(path=generated_datasets[0], train=True)
dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# isntanciate the algorithm class to solve the IP problem.
model = Optim(
    algo_name='HQS',
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    crit_conv=crit_conv,
    thres_conv=thres_conv,
    backtracking=False,
    F_fn=None,
    verbose=True,
    params_algo=params_algo
)

# Evaluate the model on the problem. this will generate a figure saved in
# '../results/results_pnp.png'
test(
    model=model,
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=True,
    save_folder=str(RESULTS_DIR),
    save_plot_path=str(RESULTS_DIR / 'results_pnp.png'),
    verbose=verbose
)
