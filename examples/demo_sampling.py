import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from utils.plotting import plot_debug

G = 1  # number of operators
epochs = 2  # number of training epochs
num_workers = 0  # set to 0 if using small cpu
images = 4
plot = True
dataset = 'MNIST'
problem = 'super_resolution'
dir = f'../datasets/MNIST/{problem}/G{G}/'

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
    elif problem == 'super_resolution':
        p = dinv.physics.Downsampling(factor=4)
    elif problem == 'deblur':
        p = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(1, .5)), device=dinv.device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=False)
    dataloader.append(DataLoader(dataset, batch_size=images, num_workers=num_workers, shuffle=False))

# FNE DnCNN from Terris et al.
denoiser = dinv.models.dncnn(in_channels=1,
                         out_channels=1).to(dinv.device)

ckp_path = '../saved_models/DNCNN_nch_1_sigma_2.0_ljr_0.001.ckpt'

u = torch.randn((1, 28, 28), device=dinv.device) #.type(dtype)

lip = physics[0].power_method(u.unsqueeze(0), tol=1e-5)
print('Lip cte = ', lip)

testadj = physics[0].adjointness_test(u.unsqueeze(0))
print('Adj test : ', testadj)

denoiser.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
denoiser = denoiser.eval()

f = dinv.sampling.PnPULA(denoiser, sigma=1, max_iter=1000, step_size=0.1/lip, verbose=True)

for g in range(G):
    iterator = iter(dataloader[g])
    x, y = next(iterator)
    y = y.to(dinv.device)

    mean, var = f(y, physics[g])

    imgs = []
    for i in range(images):
        imgs.append(y[i, :, :, :].unsqueeze(0))
        imgs.append(x[i, :, :, :].unsqueeze(0))
        imgs.append(mean[i, :, :, :].unsqueeze(0))
        imgs.append(var[i, :, :, :].unsqueeze(0))

    plot_debug(imgs, shape=(images, 4), titles=['measurement', 'ground truth', 'mean', 'variance'])
