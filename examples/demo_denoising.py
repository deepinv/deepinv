import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import *
from deepinv.optim.optimizers import *
from deepinv.training_utils import test
from torchvision import datasets, transforms


num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
problem = 'denoise'
G = 1
ckpt_path = '../checkpoints/drunet_color.pth'
denoiser_name = 'drunet'
batch_size = 1
dataset = 'set3c'
dataset_path = f'../../datasets/{dataset}'
dir = f'../datasets/{dataset}/{problem}/'
noise_level_img = 25./255
verbose = True
n_channels = 3
train = False
img_size = 256

p = dinv.physics.LinearPhysics(noise_model = dinv.physics.GaussianNoise(sigma=noise_level_img))

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

denoiser = Denoiser(model_spec=model_spec)
model = lambda x, physics: denoiser(x, noise_level_img)

test(model=model,  # Safe because it has forward
    test_dataloader=dataloader,
    physics=p,
    device=dinv.device,
    plot=True,
    plot_input=True,
    save_folder='../results/',
    save_plot_path='../results/results_denoising.png',
    verbose=verbose)
