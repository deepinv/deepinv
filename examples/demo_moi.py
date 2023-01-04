import sys
sys.path.append('../deepinv')
import deepinv as dinv
from torch.utils.data import DataLoader
import torch


G = 10
physics = []
dataloader = []
dir = '../datasets/mnistInp'

for g in range(G):
    p = dinv.physics.Inpainting(mask=.3, tensor_size=(1, 28, 28)).to(dinv.device)
    p.load_state_dict(torch.load(f'{dir}/physics{g}.pt', map_location=dinv.device))
    physics.append(p)
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=True)
    dataloader.append(DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True))

backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)

model = dinv.models.ArtifactRemoval(backbone, pinv=False)

loss_mc = dinv.loss.MCLoss(metric=dinv.metric.mse(dinv.device))
loss_moi = dinv.loss.MOILoss(metric=dinv.metric.mse(dinv.device))

optimizer = dinv.optim.Adam(model.parameters(),
                            lr=5e-4,
                            weight_decay=1e-8)

dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           epochs=400,
           schedule=[300],
           loss_closure=[loss_mc, loss_moi],
           physics=physics,
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path='dinv_moi_demo',
           plot=False,
           verbose=True)
