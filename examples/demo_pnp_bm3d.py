import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.pnp.denoiser import Denoiser
from deepinv.pnp.pnp import PnP
from deepinv.pnp.red import RED
from deepinv.training_utils import test
from torchvision import datasets, transforms

num_workers = 4  # set to 0 if using small cpu
problem = 'deblur'
G = 1
denoiser_name = 'drunet'
ckpt_path = '../checkpoints/drunet_color.pth'
pnp_algo = 'HQS'
batch_size = 1
dataset = 'set3c'
dataset_path = '../../datasets/set3c'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 0.03
lamb = 0.1
stepsize = 1 / lamb
sigma_k = 2.
sigma = sigma_k*noise_level_img

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
    p = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device)
else:
    raise Exception("The inverse problem chosen doesn't exist")

p.noise = dinv.physics.GaussianNoise(sigma=noise_level_img)

val_transform = transforms.Compose([
            transforms.ToTensor(),
 ])
dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=5,
                               num_workers=num_workers)
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_channels=3, ckpt_path=ckpt_path)

pnp = PnP(algo_name=pnp_algo, denoiser=denoiser, physics = p, max_iter=10, sigma=0.03, stepsize=stepsize, device=dinv.device)

model = lambda x,physics : pnp(x)


test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    save_img_path='../results/results_pnp_set3c.png')
