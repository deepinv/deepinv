import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.diffops.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import *
from deepinv.pnp.pnp import PnP
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.optim_iterator import *
from deepinv.training_utils import test
from torchvision import datasets, transforms

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
problem = 'deblur'
G = 1
denoiser_name = 'gsdrunet'
ckpt_path = '../checkpoints/GSDRUNet.ckpt'
pnp_algo = 'PGD'
batch_size = 1
dataset = 'set3c'
dataset_path = '../../datasets/set3c'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 0.03
lamb = 10
stepsize = 1.
sigma_k = 2.
sigma_denoiser = sigma_k*noise_level_img
max_iter = 50
crit_conv = 1e-5
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
    p = dinv.physics.BlurFFT((3,256,256), filter=dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device, noise_model = dinv.physics.GaussianNoise(sigma=noise_level_img))
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
                    'train':train, 
                    'device':dinv.device
                    }}
denoiser = Denoiser(model_spec=model_spec)

PnP_module = PnP(denoiser=denoiser, max_iter=max_iter, sigma_denoiser=sigma_denoiser, stepsize=stepsize)
iterator = PGD(prox_g=PnP_module.prox_g, data_fidelity=data_fidelity, stepsize=stepsize, device=dinv.device, update_stepsize = PnP_module.update_stepsize)
FP = FixedPoint(iterator, max_iter=max_iter, early_stop=early_stop, crit_conv=crit_conv,verbose=verbose)
model = lambda x,physics : FP(x, x, physics) # FP forward arguments are init, input, physics  

test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=False,
    plot_input=True,
    save_img_path='../results/results_pnp.png',
    verbose=verbose)
