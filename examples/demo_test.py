import deepinv as dinv
import torch
from torch.utils.data import DataLoader

# folder containing deepinv.datasets.HDF5Dataset
dir = f'../datasets/MNIST/Inpainting/'

# load test dataset
num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
batch_size = 128  # choose if using small cpu/gpu
dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=False)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# load the forward operator
physics = dinv.physics.Inpainting(tensor_size=(1, 28, 28), mask=.5, device=dinv.device)
# loads the random mask used for training
physics.load_state_dict(torch.load(f'{dir}/physics0.pt', map_location=dinv.device))

# choose the reconstruction architecture
backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=3).to(dinv.device)
model = dinv.models.ArtifactRemoval(backbone)

# load the pretrained weights
# change the string with the training date and epoch number accordingly
ckp_path = f'{dir}/dinv_demo/yy-mm-dd-hh:mm:ss/ckp_x.pth.tar'
model.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
model.eval()

# test the model
dinv.test(model=model, test_dataloader=dataloader, physics=physics, plot=True, device=dinv.device)
