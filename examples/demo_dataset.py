import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch
from torchvision import datasets, transforms

# base train dataset
# transform_data = transforms.Compose([transforms.ToTensor(), transforms.Resize(128), transforms.Pad(64, padding_mode='symmetric'),
#                                      transforms.RandomHorizontalFlip(), transforms.RandomAffine(degrees=180, translate=(0.2, 0.2)),
#                                      transforms.CenterCrop(128)])
transform_data = transforms.Compose([transforms.ToTensor()])

data_train = datasets.MNIST(root='../datasets/', train=True, transform=transform_data, download=True)
#data_train = datasets.CelebA(root='../datasets/', split='train', transform=transform_data, download=True)
# data_train = datasets.Flowers102(root='../datasets/', split='test', transform=transform_data, download=True)
#data_train = datasets.FashionMNIST(root='../datasets/', train=True, transform=transform_data, download=True)

# base test dataset
data_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)
#data_test = datasets.CelebA(root='../datasets/', split='test', transform=transform_data)
# data_test = datasets.Flowers102(root='../datasets/', split='train', transform=transform_data)
#data_test = datasets.FashionMNIST(root='../datasets/', train=False, transform=transform_data)

m = 100
G = 1
max_datapoints = 3000

# physics
dir = f'../datasets/MNIST/G_{G}_m{m}/'

physics = []
for g in range(G):
    # p = dinv.physics.CompressedSensing(m=m, fast=True, img_shape=(1, 28, 28)).to(dinv.device)
    print('Device = ', dinv.device)
    p = dinv.physics.CompressedSensing(m=m, img_shape=(1, 28, 28), device=dinv.device).to(dinv.device)
    p.sensor_model = lambda x: torch.sign(x)
    physics.append(p)

# generate paired dataset
dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                               physics=physics, device=dinv.device, save_dir=dir, max_datapoints=max_datapoints,
                               num_workers=0)
