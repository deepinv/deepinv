import torch
import deepinv as dinv
from torchvision import datasets, transforms

# choose base test and train datasets
transform_data = transforms.Compose([transforms.ToTensor()])
data_train = datasets.MNIST(root='../datasets/', train=True, transform=transform_data, download=True)
data_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)

# choose forward operator
physics = dinv.physics.Inpainting(tensor_size=(1, 28, 28), mask=.5, device=dinv.device)
physics.noise_model = dinv.physics.GaussianNoise(sigma=.1)  # add Gaussian Noise

# generate paired dataset
max_datapoints = 100
num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
dir = f'../datasets/MNIST/Inpainting/'  # save directory
dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                               physics=physics, device=dinv.device, save_dir=dir,
                               max_datapoints=max_datapoints, num_workers=num_workers)
