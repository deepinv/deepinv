import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import ProxDenoiser
from deepinv.optim.data_fidelity import *
from deepinv.unfolded.unfolded import Unfolded
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optimizers import *
from deepinv.unfolded.unfolded import *
from deepinv.training_utils import test, train
from torchvision import datasets, transforms
from deepinv.models.pd_modules import PrimalBlock, DualBlock, Toy, PrimalBlock_list, DualBlock_list
import os

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4

# PROBLEM SELECTION
problem = 'CS'
dataset = 'MNIST'
G = 1

# PRIOR SELECTION
# model_spec = {'name': 'waveletprior',
#               'args': {'wv':'db8', 'level': 3}}
model_spec = {'name': 'tgv', 'args': {'n_it_max':20, 'verbose':False}}
# model_spec = {'name': 'waveletdictprior',
#               'args': {'max_iter':2, 'list_wv': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8'], 'level':2}}
# n_channels = 1
# model_spec = {'name': 'drunet',
#               'args': {'in_channels':n_channels+1, 'out_channels':n_channels, 'nb':4, 'nc':[64, 128, 256, 512],
#                        'ckpt_path': '../checkpoints/drunet_gray.pth'}}

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
# data_fidelity = IndicatorL2(radius=2)  # If selected, need to use PD algorithm for unrolled model

train_transform = transforms.Compose([
                transforms.RandomCrop(im_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])

val_transform = None
if not os.path.exists(f'{dir}/dinv_dataset0.h5') and not 'MNIST' in dataset:
    dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
    dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=100000,
                               num_workers=num_workers)


physics = []
for g in range(G):
    if problem == 'CS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
        p.sensor_model = lambda x: torch.sign(x)
    elif problem == 'deblur':
        p = dinv.physics.BlurFFT((3, 256, 256), filter=dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.),
                                 device=dinv.device, noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img))
    try:
        p.load_state_dict(torch.load(f'{dir}/G{G}/physics{g}.pt', map_location=dinv.device))
        dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/G{G}/dinv_dataset0.h5', train=True)
    except:
        p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
        dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
    physics.append(p)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

max_iter = 5
prox_g = ProxDenoiser(model_spec, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize)
algo_name = 'PGD'
model = Unfolded(algo_name, prox_g=prox_g, data_fidelity=data_fidelity, stepsize=prox_g.stepsize, device=dinv.device,
                   g_param=prox_g.sigma_denoiser, learn_g_param=True, max_iter=max_iter, crit_conv=1e-4,
                   learn_stepsize=True, constant_stepsize=False)

# choose optimizer and scheduler
print('CHECKING TRAINABLE PARAMETERS:')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, ' is trainable')

epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=int(epochs*0.9))

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))

train(model=model,
        train_dataloader=dataloader,
        epochs=epochs,
        scheduler=scheduler,
        loss_closure=losses,
        physics=p,
        optimizer=optimizer,
        device=dinv.device,
        ckp_interval=2000,
        save_path=f'{dir}/dinv_moi_demo',
        plot=True,
        verbose=True,
        debug=True)

