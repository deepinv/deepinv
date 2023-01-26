import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch
from torch.utils.data import DataLoader


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
    dataset = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset{g}.h5', train=False)
    dataloader.append(DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False))

folder = '23-01-26-16:43:34_dinv_moi_demo'
ckp = 0
ckp_path = 'ckp/' + folder + '/ckp_' + str(ckp) + '.pth.tar'

backbone = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         scales=3).to(dinv.device)
model = dinv.models.ArtifactRemoval(backbone, pinv=False)

model.load_state_dict(torch.load(ckp_path, map_location=dinv.device)['state_dict'])
model.eval()

dinv.test(model=model,
          test_dataloader=dataloader,
          physics=physics,
          device=dinv.device,
          save_dir='results/results_CS_MNIST.png')