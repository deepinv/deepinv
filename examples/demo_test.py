import os
import deepinv as dinv
import torch
from torch.utils.data import DataLoader


G = 1  # number of operators
epochs = 2  # number of training epochs
num_workers = 4  # set to 0 if using small cpu
batch_size = 128  # choose if using small cpu/gpu
plot = True
dataset = 'MNIST'
problem = 'denoising'
ckp = 1  # saved epoch
trained_net = 'dinv_moi_demo'
dir = f'../datasets/{dataset}/{problem}/G{G}/'

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


ckp_path = ''
for dir in os.walk(dir + trained_net):
    ckp_path = dir[0]
    ckp_path = ckp_path + f'/ckp_{ckp}.pth.tar'
    if os.path.exists(ckp_path):
        break

backbone = dinv.models.UNet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)

model = dinv.models.ArtifactRemoval(backbone)

model.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
model.eval()

dinv.test(model=model,
          test_dataloader=dataloader,
          physics=physics,
          plot=plot,
          device=dinv.device)
