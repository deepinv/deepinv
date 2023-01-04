import sys
sys.path.append('../deepinv')
import deepinv as dinv
from torch.utils.data import DataLoader
import torch


physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28)).to(dinv.device)
physics.load_state_dict(torch.load('../datasets/mnistCS/physics0.pt', map_location=dinv.device))

dataset = dinv.datasets.HDF5Dataset(path='../datasets/mnistCS/dinv_dataset.h5', train=True)
dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True)

#dataloader = dinv.datasets.mnist_dataloader(train=True, batch_size=128, num_workers=4, shuffle=True)

backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)

model = dinv.models.FBPNet(backbone)

loss_mc = dinv.loss.MCLoss(metric=dinv.metric.mse(dinv.device))
loss_sup = dinv.loss.SupLoss(metric=dinv.metric.mse(dinv.device))

loss_ei = dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=2),
                           metric=dinv.metric.mse(dinv.device))

loss_mcsure = dinv.loss.SureMCLoss(sigma=.2)

loss_ms = dinv.loss.MeaSplitLoss(metric=dinv.metric.mse(dinv.device),
                                 split_ratio=0.9)

optimizer = dinv.optim.Adam(model.parameters(),
                            lr=5e-4,
                            weight_decay=1e-8)

dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           epochs=500,
           schedule=[400],
           loss_closure=[loss_ei, loss_sup],
           loss_weight=[1, 1],
           physics=physics,
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path='dinv_mcsure_ei',
           plot=False,
           verbose=True)
