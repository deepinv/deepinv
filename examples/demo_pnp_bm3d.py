import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.pnp.denoiser import Denoiser
from deepinv.pnp.pnp import PnP
from deepinv.pnp.red import RED
from deepinv.training_utils import test

num_workers = 4  # set to 0 if using small cpu
dataset = 'MNIST'
problem = 'deblur'
G = 1
dir = f'../datasets/{dataset}/{problem}/G{G}'
denoiser_name = 'bm3d'
pnp_algo = 'HQS'
batch_size = 128

physics = []
dataloader = []
for g in range(G):
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

    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=False)
    dataloader.append(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False))

denoiser = Denoiser(denoiser_name=denoiser_name)

pnp = PnP(algo_name=pnp_algo, denoiser=denoiser, physics = physics, max_iter=100, device=dinv.device)

test(model=pnp,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=physics,
    device=dinv.device,
    plot=True,
    save_img_path='../results/results_pnp_CS_MNIST.png')
