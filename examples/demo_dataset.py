import torch
import deepinv as dinv
from torchvision import datasets, transforms

# base train dataset
transform_data = transforms.Compose([transforms.ToTensor()])

G = 1  # number of operators
max_datapoints = 1e7
num_workers = 0  # set to 0 if using cpu

# # problem
# problem = 'denoising'
# dataset = 'set3c'
# dir = f'../datasets/{dataset}/{problem}/G{G}/'
problem = 'CS'
dataset = 'MNIST'
max_datapoints = 10
dir = f'../datasets/{dataset}/{problem}/G{G}/'


if dataset == 'MNIST':
    data_train = datasets.MNIST(root='../datasets/', train=True, transform=transform_data, download=True)
    data_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)

elif dataset == 'set3c':

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_train = datasets.ImageFolder(root='../datasets/set3c/', transform=val_transform)
    data_test = datasets.ImageFolder(root='../datasets/set3c/', transform=val_transform)

elif dataset =='CelebA':
    data_train = datasets.CelebA(root='../datasets/', split='train', transform=transform_data, download=True)
    data_test = datasets.CelebA(root='../datasets/', split='test', transform=transform_data)

elif dataset =='FashionMNIST':
    data_train = datasets.FashionMNIST(root='../datasets/', train=True, transform=transform_data, download=True)
    data_test = datasets.FashionMNIST(root='../datasets/', train=False, transform=transform_data)

else :
    raise Exception("The dataset chosen doesn't exist")


x = data_train[0]
im_size = x[0].shape if isinstance(x, list) or isinstance(x, tuple) else x.shape

physics = []
for g in range(G):
    if problem == 'CS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=im_size, device=dinv.device)
    elif problem == 'onebitCS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=im_size, device=dinv.device)
        p.sensor_model = lambda x: torch.sign(x)
    elif problem == 'inpainting':
        p = dinv.physics.Inpainting(tensor_size=im_size, mask=.5, device=dinv.device)
    elif problem == 'blind_deblur': # TODO
        p = dinv.physics.BlindBlur(kernel_size=11)
    elif problem == 'super_resolution':
        p = dinv.physics.Downsampling(factor=4)
    elif problem == 'denoising':
        p = dinv.physics.Denoising()
    elif problem == 'CT': # TODO
        p = dinv.physics.CT(img_width=im_size[-1], views=30)
    elif problem == 'deblur':
        p = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    p.noise_model = dinv.physics.GaussianNoise(sigma=.1)
    physics.append(p)

# generate paired dataset
dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                               physics=physics, device=dinv.device, save_dir=dir, max_datapoints=max_datapoints,
                               num_workers=num_workers)
