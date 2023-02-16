import sys
import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.pnp.denoiser import Denoiser
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.pnp.pnp import UnrolledPnP
from deepinv.training_utils import train,test
from torchvision import datasets, transforms
import os

num_workers = 4  # set to 0 if using small cpu
problem = 'deblur'
G = 1
denoiser_name = 'drunet'
ckpt_path = '../checkpoints/drunet_color.pth'
pnp_algo = 'HQS'
batch_size = 128
dataset = 'DRUNET'
dir = f'../datasets/{dataset}/{problem}/'
dataset_path = f'../datasets/{dataset}/'
noise_level_img = 0.03
lamb = 10
stepsize = 1.
sigma_k = 2.
sigma_denoiser = sigma_k*noise_level_img
max_iter = 10
im_size = 256
epochs = 2


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
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
 ])
train_transform = transforms.Compose([
                transforms.RandomCrop(im_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])
if not os.path.exists(f'{dir}/dinv_dataset0.h5'):
    dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
    dinv.datasets.generate_dataset(train_dataset=dataset, test_dataset=None,
                               physics=p, device=dinv.device, save_dir=dir, max_datapoints=1000,
                               num_workers=num_workers)
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_channels=3, pretrain=True, ckpt_path=ckpt_path, train=True)

pnp = UnrolledPnP(backbone_net=denoiser, sigma_denoiser=sigma_denoiser, algo_name=pnp_algo, data_fidelity=data_fidelity, max_iter=max_iter, stepsize=stepsize, device=dinv.device)

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))

# choose optimizer and scheduler
optimizer = torch.optim.Adam(pnp.parameters(), lr=1e-4, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))

train(model=pnp,
        train_dataloader=dataloader,
        epochs=epochs,
        scheduler=scheduler,
        loss_closure=losses,
        physics=p,
        optimizer=optimizer,
        device=dinv.device,
        ckp_interval=250,
        save_path=f'{dir}/dinv_moi_demo',
        plot=False,
        verbose=True)
