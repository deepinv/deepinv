import sys
sys.path.append('../deepinv')
import deepinv as dinv
from torch.utils.data import DataLoader
import torch


G = 1
m = 100
physics = []
dataloader = []
dir = f'../datasets/MNIST/G_{G}_m{m}/'

for g in range(G):
    p = dinv.physics.CompressedSensing(m=m, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
    p.sensor_model = lambda x: torch.sign(x)
    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=True)
    dataloader.append(DataLoader(dataset, batch_size=10, num_workers=0, shuffle=True))

backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)

model = dinv.models.ArtifactRemoval(backbone, pinv=False)

loss_sup = dinv.loss.SupLoss(metric=dinv.metric.mse())
# loss_mc = dinv.loss.MCLoss(metric=dinv.metric.mse)
# loss_moi = dinv.loss.MOILoss(metric=dinv.metric.mse)

optimizer = dinv.optim.Adam(model.parameters(),
                            lr=5e-4,
                            weight_decay=1e-8)

dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           epochs=1,
           schedule=[300],
           loss_closure=[loss_sup],
           physics=physics,
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path='dinv_moi_demo',
           plot=False,
           verbose=True)
