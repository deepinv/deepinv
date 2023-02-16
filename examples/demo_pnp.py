import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.pnp.denoiser import Denoiser
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.pnp.pnp import PnP
from deepinv.training_utils import test
from torchvision import datasets, transforms

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
problem = 'deblur'
G = 1
denoiser_name = 'drunet'
ckpt_path = '../checkpoints/drunet_color.pth'
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

data_fidelity = DataFidelity(type='L2')

val_transform = transforms.Compose([
            transforms.ToTensor(),
 ])
dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=5,
                               num_workers=num_workers)
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_channels=3, ckpt_path=ckpt_path)
denoiser_name = 'TGV'
denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_it_max=100)

if denoiser_name=='TGV':
    sigma_denoiser = sigma_denoiser*5  # Small tweak, tested on PGD, but a little bit too high on HQS

pnp = PnP(denoiser=denoiser, sigma_denoiser=sigma_denoiser, algo_name=pnp_algo, data_fidelity=data_fidelity,
          max_iter=max_iter, crit_conv=1e-3, stepsize=stepsize, device=dinv.device, verbose=True)

test(model=pnp,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=True,
    save_img_path='../results/results_pnp.png')
