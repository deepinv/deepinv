import sys
import numpy as np
import deepinv as dinv
import hdf5storage
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import ProxDenoiser
from deepinv.optim.data_fidelity import *
from deepinv.optim.optimizers import *
from deepinv.training_utils import test
from torchvision import datasets, transforms
from deepinv.utils.parameters import get_DPIR_params, initialize_stepsizes

torch.manual_seed(0)

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
problem = 'deblur'
G = 1
ckpt_path = '../checkpoints/drunet_color.pth'
denoiser_name = 'drunet'
batch_size = 1
dataset = 'starfish'
dataset_path = f'../../datasets/{dataset}'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 0.03
max_iter = 8
verbose = True
early_stop = False 
n_channels = 3
train = False
crit_conv = 'residual'
thres_conv = 1e-4
img_size = 256

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
    kernels = hdf5storage.loadmat('../kernels/Levin09.mat')['kernels']
    filter_np = kernels[0,1].astype(np.float64)
    filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)
    p = dinv.physics.BlurFFT(img_size = (3,img_size,img_size), filter=filter_torch, device=dinv.device, noise_model = dinv.physics.GaussianNoise(sigma=noise_level_img))
    #p = dinv.physics.Blur(filter=filter_torch, device=dinv.device, noise_model = dinv.physics.GaussianNoise(sigma=noise_level_img))
else:
    raise Exception("The inverse problem chosen doesn't exist")

data_fidelity = L2()

val_transform = transforms.Compose([
            transforms.ToTensor(),
 ])
dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=3,
                               num_workers=num_workers)
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


model_spec = {'name': denoiser_name,
              'args': {
                    'in_channels':n_channels+1, 
                    'out_channels':n_channels,
                    'pretrained':ckpt_path, 
                    'train': False, 
                    'device':dinv.device
                    }}


lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level_img)
stepsize, sigma_denoiser = initialize_stepsizes(stepsize, sigma_denoiser, max_iter)

prox_g = ProxDenoiser(model_spec)
model = Optim(algo_name = 'HQS', prox_g=prox_g, g_first = False, data_fidelity=data_fidelity, lamb=lamb, stepsize=stepsize, device=dinv.device,
             g_param=sigma_denoiser, early_stop=early_stop, max_iter=max_iter, crit_conv=crit_conv, thres_conv=thres_conv, backtracking=False, F_fn=None, verbose=True)

test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=True,
    save_folder='../results/',
    save_plot_path='../results/results_pnp.png',
    verbose=verbose)
