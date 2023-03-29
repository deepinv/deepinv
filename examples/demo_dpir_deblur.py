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

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
problem = 'deblur'
G = 1
# denoiser_name = 'gsdrunet'
# ckpt_path = '../checkpoints/GSDRUNet.ckpt'
denoiser_name = 'drunet'
ckpt_path = '../checkpoints/drunet_color.pth'
batch_size = 1
dataset = 'set3c'
dataset_path = '../../datasets/set3c'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 0.03
lamb = 10
stepsize = 1.
sigma_k = 2.
sigma_denoiser = sigma_k*noise_level_img
max_iter = 8
crit_conv = 1e-3
verbose = True
early_stop = True 
n_channels = 3
pretrain = True
train = False

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
    filter_np = kernels[0,0].astype(np.float64)
    filter_torch = torch.from_numpy(filter_np).unsqueeze(0).unsqueeze(0)
    p = dinv.physics.Blur(filter=filter_torch, device=dinv.device, noise_model = dinv.physics.GaussianNoise(sigma=noise_level_img))
else:
    raise Exception("The inverse problem chosen doesn't exist")

data_fidelity = L2()

val_transform = transforms.Compose([
            transforms.ToTensor(),
 ])
dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=5,
                               num_workers=num_workers)
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


model_spec = {'name': denoiser_name,
              'args': {
                    'in_channels':n_channels+1, 
                    'out_channels':n_channels,
                    'ckpt_path': ckpt_path,
                    'pretrain':pretrain, 
                    'train': False, 
                    'device':dinv.device
                    }}

# STEP 2: Defining the model
def get_rho_sigma(sigma=2.55/255, iter_num=8, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = list((modelSigmaS*w+modelSigmaS_lin*(1-w))/255.)
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas

rhos, sigmas = get_rho_sigma(sigma=max(0.255/255., noise_level_img), iter_num=max_iter, modelSigma1=49.0, modelSigma2=2.55, w=1.0)

prox_g = ProxDenoiser(model_spec, sigma_denoiser=sigmas, stepsize=rhos, max_iter=max_iter)
algo_name = 'HQS'
model = Optim(algo_name, prox_g=prox_g, data_fidelity=data_fidelity, stepsize=prox_g.stepsize, device=dinv.device,
             g_param=prox_g.sigma_denoiser, max_iter=max_iter, crit_conv=1e-4, verbose=True)

test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=True,
    save_folder='../results/',
    save_plot_path='../results/results_pnp.png',
    verbose=verbose)
