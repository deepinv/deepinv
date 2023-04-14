import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import ProxDenoiser
from deepinv.optim.data_fidelity import *
from deepinv.optim.optimizers import Optim
from deepinv.training_utils import test
from torchvision import datasets, transforms


num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
denoiser_name = 'gsdrunet'
ckpt_path = None
algo_name = 'PGD'
batch_size = 1
dataset_path = '../../datasets/set3c'
dir = f'../datasets/set3c/deblur/'
noise_level_img = 0.03
sigma_denoiser = 2.*noise_level_img
stepsize = 1.
max_iter = 50
crit_conv = 1e-3
verbose = True
pretrain = 'download'


physics = dinv.physics.BlurFFT((3, 256, 256), filter=dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.),
                         device=dinv.device, noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img))

data_fidelity = L2()

val_transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=5,
                               num_workers=num_workers)

dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


model_spec = {'name': denoiser_name,
              'args': {
                    'in_channels': 4,
                    'out_channels': 3,
                    'ckpt_path': ckpt_path,
                    'pretrain': pretrain,
                    'train': False, 
                    'device': dinv.device
                    }}

# STEP 2: Defining the model
prox_g = ProxDenoiser(model_spec, sigma_denoiser=sigma_denoiser, stepsize=stepsize, max_iter=max_iter)
model = Optim(algo_name, prox_g=prox_g, data_fidelity=data_fidelity, stepsize=prox_g.stepsize, device=dinv.device,
             g_param=prox_g.sigma_denoiser, max_iter=max_iter, crit_conv=crit_conv, verbose=True)


test(model=model, test_dataloader=dataloader, physics=physics, device=dinv.device, plot=True, plot_input=True,
    save_img_path='../results/results_pnp.png', verbose=verbose)
