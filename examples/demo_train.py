import deepinv as dinv
from torch.utils.data import DataLoader
import torch

G = 1  # number of operators
epochs = 2  # number of training epochs
num_workers = 4  # set to 0 if using small cpu
batch_size = 128  # choose if using small cpu/gpu
plot = False
dataset = 'MNIST'
problem = 'denoising'
physics = []
dataloader = []
dir = f'../datasets/{dataset}/{problem}/G{G}/'

# choose training losses
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))
#losses.append(dinv.loss.MCLoss(metric=dinv.metric.mse()))
#losses.append(dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=1)))
#losses.append(dinv.loss.MOILoss(metric=dinv.metric.mse())

# choose backbone denoiser
backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)
# choose unrolled architecture
model = dinv.models.ArtifactRemoval(backbone)

# choose optimizer and scheduler
optimizer = dinv.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))


for g in range(G):
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=True)
    dataloader.append(DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True))

    x = dataset[0]
    im_size = x[0].shape if isinstance(x, list) or isinstance(x, tuple) else x.shape

    if problem == 'CS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=im_size, device=dinv.device)
    elif problem == 'onebitCS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=im_size, device=dinv.device)
        p.sensor_model = lambda x: torch.sign(x)
    elif problem == 'inpainting':
        p = dinv.physics.Inpainting(tensor_size=im_size, mask=.5, device=dinv.device)
    elif problem == 'denoising':
        p = dinv.physics.Denoising(sigma=.2)
    elif problem == 'blind_deblur':
        p = dinv.physics.BlindBlur(kernel_size=11)
    elif problem == 'CT':
        p = dinv.physics.CT(img_width=im_size[-1], views=30)
    elif problem == 'deblur':
        p = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)


dinv.train(model=model,
           train_dataloader=dataloader,
           epochs=epochs,
           scheduler=scheduler,
           loss_closure=losses,
           physics=physics,
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path=f'{dir}/dinv_moi_demo',
           plot=plot,
           verbose=True)
