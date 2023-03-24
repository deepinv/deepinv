import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.diffops.models.denoiser import ProxDenoiser
from deepinv.optim.data_fidelity import *
from deepinv.unfolded.unfolded import Unfolded
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optimizers import *
from deepinv.unfolded.unfolded import *
from deepinv.training_utils import test, train
from torchvision import datasets, transforms
from deepinv.diffops.models.pd_modules import PrimalBlock, DualBlock, Toy, PrimalBlock_list, DualBlock_list
import os

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4

problem = 'CS'
dataset = 'MNIST'
G = 1

# PRIOR SELECTION
# model_spec = {'name': 'tgv', 'args': {'n_it_max':100, 'verbose':True}}
# model_spec = {'name': 'waveletprior',
#               'args': {'wv':'db8', 'level': 3}}
# model_spec = {'name': 'waveletdictprior',
#               'args': {'max_iter':10, 'list_wv': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8'], 'level':2}}
n_channels = 1
name_drunet = 'drunet_color' if n_channels == 3 else 'drunet_gray'
model_spec = {'name': 'drunet',
              'args': {'in_channels':n_channels+1, 'out_channels':n_channels, 'nb':4, 'nc':[64, 128, 256, 512],
                       'ckpt_path': '../checkpoints/'+name_drunet+'.pth'}}

# PATH, BATCH SIZE ETC
batch_size = 3
dataset_path = f'../../datasets/{dataset}/'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 0.03
lamb = 10
stepsize = 1.
sigma_k = 2.
sigma_denoiser = sigma_k*noise_level_img
im_size = 256
epochs = 2
max_iter = 50
crit_conv = 1e-5
verbose = True
early_stop = True

data_fidelity = L2()
# data_fidelity = IndicatorL2(radius=2)

val_transform = None
train_transform = None

physics = []
for g in range(G):
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
    p.sensor_model = lambda x: torch.sign(x)

    p.load_state_dict(torch.load(f'{dir}/G{G}/physics{g}.pt', map_location=dinv.device))
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/G{G}/dinv_dataset0.h5', train=True)
    physics.append(p)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


# STEP 2: Defining the model
prox_g = ProxDenoiser(model_spec, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize)
model = UnfoldedPGD(prox_g=prox_g, data_fidelity=data_fidelity, stepsize=prox_g.stepsize, device=dinv.device,
                    g_param=prox_g.sigma_denoiser, max_iter=max_iter, crit_conv=1e-4, verbose=True)

# STEP 3: Test the model
test(model=model,
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=False,
    save_img_path='../results/results_pnp_PGD.png',
    verbose=verbose)
