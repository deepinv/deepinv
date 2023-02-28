import sys
import os

import deepinv as dinv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deepinv.training_utils import test, train
from torchvision import datasets, transforms
from deepinv.diffops.models.pd_modules import Toy

num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4


# Setting variables
problem = 'CS'
dataset = 'MNIST'
G = 1
dataset_path = f'../../datasets/{dataset}/'
dir = f'../datasets/{dataset}/{problem}/'
batch_size = 1


# Generate physical problem
physics = []
for g in range(G):
    p = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
    p.sensor_model = lambda x: torch.sign(x)

    p.load_state_dict(torch.load(f'{dir}/G{G}/physics{g}.pt', map_location=dinv.device))
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/G{G}/dinv_dataset0.h5', train=True)
    physics.append(p)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)



# We define a dummy model
class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.toy_model = Toy(in_channels=1, out_channels=1)

    def forward(self, x, physics):
        out = self.toy_model(physics.A_adjoint(x))
        return out

model = Dummy()



# Let's check that the parameters of the dummy model are trainable
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, ' is trainable')



# Setting up the training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000)
losses = []
losses.append(dinv.loss.SupLoss(metric=dinv.metric.mse()))


# Train
train(model=model,
        train_dataloader=dataloader,
        epochs=100,
        scheduler=scheduler,
        loss_closure=losses,
        physics=p,
        optimizer=optimizer,
        device=dinv.device,
        ckp_interval=1000,
        save_path=f'{dir}/dinv_moi_demo',
        plot=False,
        verbose=True)
