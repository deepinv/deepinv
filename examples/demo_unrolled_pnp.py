import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import *
from deepinv.training_utils import test, train
from deepinv.unfolded.unfolded import Unfolded_algo
from deepinv.unfolded.deep_equilibrium import DEQ_algo
from torchvision import datasets, transforms
import os
import wandb
from deepinv.models.denoiser import ProxDenoiser

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
problem = 'deblur'
G = 1
denoiser_name = 'dncnn'
depth = 7
ckpt_path = None
pnp_algo = 'PGD'
#path_datasets = '../../datasets'
path_datasets = '../datasets'
train_dataset_name = 'drunet'
# train_dataset_name = 'CBSD68'  # for debugging
test_dataset_name = 'CBSD68'
noise_level_img = 0.03
lamb = 10
stepsize = 1.
sigma_k = 2.
sigma_denoiser = sigma_k*noise_level_img
max_iter = 5
crit_conv = 1e-5
verbose = True
early_stop = False 
n_channels = 3
pretrain = False
epochs = 100
im_size = 32
batch_size = 32
max_datapoints = 100
deep_equilibrium = False
anderson_acceleration = True
anderson_beta=1.
anderson_history_size=5
max_iter_backward=10

wandb_vis = True

if wandb_vis :
    wandb.init(project='unrolling')

if problem == 'CS':
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, im_size, im_size), device=dinv.device)
elif problem == 'onebitCS':
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, im_size, im_size), device=dinv.device)
    p.sensor_model = lambda x: torch.sign(x)
elif problem == 'inpainting':
    p = dinv.physics.Inpainting(tensor_size=(1, im_size, im_size), mask=.5, device=dinv.device)
elif problem == 'denoising':
    p = dinv.physics.Denoising(sigma=.2)
elif problem == 'blind_deblur':
    p = dinv.physics.BlindBlur(kernel_size=11)
elif problem == 'deblur':
    p = dinv.physics.BlurFFT((3, im_size, im_size), filter=dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.),
                                 device=dinv.device, noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img))
else:
    raise Exception("The inverse problem chosen doesn't exist")

data_fidelity = L2()


if not os.path.exists(f'{path_datasets}/artificial/{train_dataset_name}/dinv_dataset0.h5'):
    val_transform = transforms.Compose([
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
    ])
    train_transform = transforms.Compose([
                    transforms.RandomCrop(im_size, pad_if_needed=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                ])
    train_input_dataset = datasets.ImageFolder(root=f'{path_datasets}/{train_dataset_name}/', transform=train_transform)
    test_input_dataset = datasets.ImageFolder(root=f'{path_datasets}/{test_dataset_name}/', transform=val_transform)
    dinv.datasets.generate_dataset(train_dataset=train_input_dataset, test_dataset=test_input_dataset,
                                physics=p, device=dinv.device, save_dir=f'{path_datasets}/artificial/{train_dataset_name}/', max_datapoints=max_datapoints,
                                num_workers=num_workers)

train_dataset = dinv.datasets.HDF5Dataset(path=f'{path_datasets}/artificial/{train_dataset_name}/dinv_dataset0.h5', train=True)
eval_dataset = dinv.datasets.HDF5Dataset(path=f'{path_datasets}/artificial/{train_dataset_name}/dinv_dataset0.h5', train=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

model_spec = {'name': denoiser_name,
              'args': {
                    'in_channels':n_channels, 
                    'out_channels':n_channels,
                    'depth': depth,
                    'ckpt_path': ckpt_path,
                    'pretrain':pretrain, 
                    'train': True, 
                    'device':dinv.device
                    }}

prox_g = ProxDenoiser(model_spec, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize)

if deep_equilibrium: 
    model = DEQ_algo(pnp_algo, prox_g=prox_g, data_fidelity=data_fidelity, stepsize=prox_g.stepsize, device=dinv.device,
                    g_param=prox_g.sigma_denoiser, learn_g_param=True, max_iter=max_iter, crit_conv=1e-4,
                    learn_stepsize=True, constant_stepsize=False, anderson_acceleration=anderson_acceleration, max_iter_backward=max_iter_backward)
else: 
    model = Unfolded_algo(pnp_algo, prox_g=prox_g, data_fidelity=data_fidelity, stepsize=prox_g.stepsize, device=dinv.device,
                    g_param=prox_g.sigma_denoiser, learn_g_param=True, max_iter=max_iter, crit_conv=1e-4,
                    learn_stepsize=True, constant_stepsize=False)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, ' is trainable')

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))

train(model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=epochs,
        scheduler=scheduler,
        loss_closure=losses,
        physics=p,
        optimizer=optimizer,
        device=dinv.device,
        ckp_interval=int(epochs/2.),
        save_path=f'../checkpoints/tests/demo_unrolled',
        plot=False,
        plot_input=True,
        verbose=True,
        wandb_vis=wandb_vis,
        debug=False)
