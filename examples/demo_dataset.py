import sys
sys.path.append('../deepinv')
import deepinv as dinv
import torch
from torchvision import datasets, transforms

# base train dataset
transform_data = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='../datasets/', train=True, transform=transform_data)
# base test dataset
mnist_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)

for m in [50, 100, 200, 300, 400, 784, 1000]:
    # physics
    physics = dinv.physics.CompressedSensing(m=m, img_shape=(1, 28, 28)).to(dinv.device)
    physics.sensor_model = lambda x: torch.sign(x)

    # generate paired dataset
    dir = f'../datasets/onebitCS/{m}/'
    dinv.datasets.generate_dataset(train_dataset=mnist_train, test_dataset=mnist_test,
                                   physics=physics, device=dinv.device, save_dir=dir)