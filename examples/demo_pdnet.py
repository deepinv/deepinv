import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.diffops.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import *
from deepinv.pnp.pnp import PnP
from deepinv.unfolded.unfolded import Unfolded
from deepinv.optim.fixed_point import FixedPoint
# from deepinv.optim.optim_iterator import *
from deepinv.optim.optimizers.primal_dual import PD
from deepinv.optim.optimizers.proximal_gradient_descent import PGD
from deepinv.training_utils import test, train
from torchvision import datasets, transforms
from deepinv.diffops.models.pd_modules import PrimalBlock, DualBlock, Toy, PrimalBlock_list, DualBlock_list
import os

# num_workers = 4  # set to 0 if using small cpu
num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4

# PROBLEM SELECTION
# # EITHER
# dataset = 'set3c'
# problem = 'deblur'
# G = 1

# OR
problem = 'CS'
dataset = 'MNIST'
G = 1

# PRIOR SELECTION
# model_spec = {'name': 'tgv', 'args': {'n_it_max':500, 'verbose':True}}
model_spec = {'name': 'waveletprior',
              'args': {'wv':'db8', 'level': 3}}
# model_spec = {'name': 'waveletdictprior',
#               'args': {'max_iter':10, 'list_wv': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8'], 'level':2}}
# n_channels = 3
# model_spec = {'name': 'drunet',
#               'args': {'in_channels':n_channels+1, 'out_channels':n_channels, 'nb':4, 'nc':[64, 128, 256, 512],
#                        'ckpt_path': '../checkpoints/drunet_color.pth'}}

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

if problem == 'CS':
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device)
elif problem == 'onebitCS':
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device)
    p.sensor_model = lambda x: torch.sign(x)
elif problem == 'inpainting':
    p = dinv.physics.Inpainting(tensor_size=(1, 28, 28), mask=.5, device=dinv.device)
elif problem == 'denoising':
    p = dinv.physics.Denoising(sigma=.2)
elif problem == 'blind_deblur':
    p = dinv.physics.BlindBlur(kernel_size=11)
elif problem == 'deblur':
    p = dinv.physics.BlurFFT((3,256,256), filter=dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device, noise_model = dinv.physics.GaussianNoise(sigma=noise_level_img))
else:
    raise Exception("The inverse problem chosen doesn't exist")

data_fidelity = L2()
# data_fidelity = IndicatorL2(radius=2)

# val_transform = transforms.Compose([
#             transforms.CenterCrop(im_size),
#             transforms.ToTensor(),
#  ])
val_transform = None
# train_transform = transforms.Compose([
#                 transforms.RandomCrop(im_size, pad_if_needed=True),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 transforms.ToTensor(),
#             ])
train_transform = None
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


# STEP 2: debugging PD
denoiser = Denoiser(model_spec=model_spec)
PnP_module = PnP(denoiser=denoiser, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize)
# iterator = PD(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=PnP_module.stepsize,
#                device=dinv.device, g_param=PnP_module.sigma_denoiser)
iterator = PGD(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=PnP_module.stepsize,
               device=dinv.device, g_param=PnP_module.sigma_denoiser)
# iterator = DRS(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=PnP_module.stepsize,
#                device=dinv.device, g_param=PnP_module.sigma_denoiser)
model = Unfolded(iterator, max_iter=max_iter, crit_conv=1e-4, learn_g_param=True, learn_stepsize=True,
                 trainable=denoiser, verbose=True)


test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=False,
    save_img_path='../results/results_pnp_PGD_nonlearned_weights.png',
    verbose=verbose)


# # STEP 3: TRAIN
# custom_primal_prox = nn.ModuleList([PrimalBlock() for _ in range(max_iter)])
# custom_dual_prox = nn.ModuleList([DualBlock() for _ in range(max_iter)])
#
# max_iter = 5
# denoiser = Denoiser(model_spec=model_spec)
# PnP_module = PnP(denoiser=denoiser, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize)
# # iterator = PD(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=PnP_module.stepsize,
# #                device=dinv.device, g_param=PnP_module.sigma_denoiser)
# # iterator = DRS(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=PnP_module.stepsize,
# #                device=dinv.device, g_param=PnP_module.sigma_denoiser)
# # model = Unfolded(iterator, max_iter=max_iter, crit_conv=1e-4, learn_g_param=True, learn_stepsize=True,
# #                  trainable=denoiser, verbose=True)
#
#
# custom_g_step = PrimalBlock_list(max_it=max_iter)
# custom_f_step = DualBlock_list(max_it=max_iter)
#
# iterator = PD(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=PnP_module.stepsize,
#                device=dinv.device, g_param=PnP_module.sigma_denoiser)
# model = Unfolded(iterator,
#                  custom_g_step=custom_g_step, custom_f_step=custom_f_step,
#                  # custom_primal_prox=custom_primal_prox, custom_dual_prox=custom_dual_prox,
#                  max_iter=max_iter, verbose=False)
#
# # choose optimizer and scheduler
#
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, ' is trainable')
#
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(10000000))
# # choose training losses
# losses = []
# losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))
# #
# train(model=model,
#         train_dataloader=dataloader,
#         epochs=1000,
#         scheduler=scheduler,
#         loss_closure=losses,
#         physics=p,
#         optimizer=optimizer,
#         device=dinv.device,
#         ckp_interval=2000,
#         save_path=f'{dir}/dinv_moi_demo',
#         plot=False,
#         verbose=True,
#         debug=True)
