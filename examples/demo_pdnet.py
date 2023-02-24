import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.diffops.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import *
from deepinv.pnp.pnp import PnP
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optim_iterator import *
from deepinv.training_utils import test, train
from torchvision import datasets, transforms
import os

# num_workers = 4  # set to 0 if using small cpu
num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
# problem = 'deblur'
# G = 1
# # denoiser_name = 'tiny_drunet'
# denoiser_name = 'TGV'
# ckpt_path = '../checkpoints/drunet_color.pth'
# batch_size = 128
# dataset = 'set3c'
# dataset_path = '../../datasets/set3c'
problem = 'CS'
G = 1
# denoiser_name = 'tiny_drunet'
denoiser_name = 'TGV'  # <-- THIS IS OPTIONAL
# model_spec = {'name': 'tgv', 'args': {'n_it_max':500, 'verbose':True}}
model_spec = {'name': 'waveletprior',
              'args': {'y_shape':(1,1,28,28), 'max_it':100, 'verbose':False, 'list_wv':['db4'], 'level':1}}
batch_size = 3
dataset = 'MNIST'
dataset_path = f'../../datasets/{dataset}/'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 0.03
lamb = 10
stepsize = 1.
sigma_k = 2.
sigma_denoiser = sigma_k*noise_level_img
max_iter = 6
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
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
    p.sensor_model = lambda x: torch.sign(x)
    p.load_state_dict(torch.load(f'{dir}/G{G}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/G{G}/dinv_dataset0.h5', train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# if denoiser_name=='TGV':
denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_it_max=100, model_spec=model_spec)
sigma_denoiser = sigma_denoiser*1.0  # Small tweak, tested on PGD, but a little bit too high on HQS

# denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_channels=3, pretrain=False, ckpt_path=ckpt_path, train=True)

# pnp_algo = 'HQS'
# pnp = PnP(denoiser=denoiser, sigma_denoiser=sigma_denoiser, algo_name=pnp_algo, data_fidelity=data_fidelity, max_iter=max_iter, stepsize=stepsize, device=dinv.device, unroll=True)

PnP_module = PnP(denoiser=denoiser, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize, unroll=True, weight_tied=True)
iterator = PGD(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=stepsize, device=dinv.device, update_stepsize = PnP_module.update_stepsize)
FP = FixedPoint(iterator, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv,verbose=verbose)
# model = lambda x, physics : FP(physics.A_adjoint(x), x, physics) # FP forward arguments are init, input, physics

def model(x, physics):
    x_init = physics.A_adjoint(x)
    return FP(x_init, x, physics)

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))

# choose optimizer and scheduler
optimizer = torch.optim.Adam(PnP_module.parameters(), lr=1e-4, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))

test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=True,
    save_img_path='../results/results_pnp_1.png')

# train(model=model,
#         train_dataloader=dataloader,
#         epochs=epochs,
#         scheduler=scheduler,
#         loss_closure=losses,
#         physics=p,
#         optimizer=optimizer,
#         device=dinv.device,
#         ckp_interval=250,
#         save_path=f'{dir}/dinv_moi_demo',
#         plot=False,
#         verbose=True)
