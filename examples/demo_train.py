import deepinv as dinv
from torch.utils.data import DataLoader
import torch

# choose training epochs
epochs = 10

# choose training losses
losses = []
losses.append(dinv.loss.MCLoss(metric=dinv.metric.mse()))  # self-supervised loss
losses.append(dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=1)))

# choose backbone denoiser
backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=3).to(dinv.device)

# choose a reconstruction architecture
model = dinv.models.ArtifactRemoval(backbone)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))

# load dataset
dir = f'../datasets/MNIST/Inpainting/'  # folder containing deepinv.datasets.HDF5Dataset
num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
batch_size = 64  # choose if using small cpu/gpu
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

# choose a forward operator
physics = dinv.physics.Inpainting(tensor_size=(1, 28, 28), mask=.5, device=dinv.device)
physics.noise_model = dinv.physics.GaussianNoise()  # add Gaussian Noise module
# loads the random mask used to generate dataset
physics.load_state_dict(torch.load(f'{dir}/physics0.pt'))

# train the network
dinv.train(model=model,
           train_dataloader=dataloader,
           epochs=epochs,
           scheduler=scheduler,
           losses=losses,
           physics=physics,
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=int(epochs/2),
           save_path=f'{dir}/dinv_demo',
           plot=False,
           verbose=False)
