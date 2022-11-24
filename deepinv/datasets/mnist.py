import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def mnist_dataloader(mode='train', batch_size=1, shuffle=True, num_workers=1):
    transform_data = transforms.Compose([transforms.ToTensor()])
    data_set = datasets.MNIST(root='../datasets/',
                              train=True if mode=='train' else False,
                              download=True,
                              transform=transform_data)
    return DataLoader(data_set,batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)


# generate_dataset(physics)
#   self.physics = physics
#
#   def __getitem__():
#       return x, y, self.physics

if __name__ == '__main__':
    # data = mnist_dataloader('train')
    # print(len(data)) # 60000
    # data = mnist_dataloader('test')
    # print(len(data)) #10000

    # device = torch.device(f'cuda:0')
    # dtype = torch.float
    # from deepinv.diffops.physics.inpainting import Inpainting
    # physics = Inpainting(28,28,0.3).to(device)

    import deepinv as dinv
    physics = dinv.physics.inpainting(28,28,0.3, device=dinv.device)
    dataloader = dinv.datasets.mnist_dataloader('train')


    device = torch.device(f'cuda:0')
    for i, x in enumerate(dataloader):
        x = x[0] if isinstance(x, list) else x
        x = x.type(dinv.dtype).to(dinv.device)

        print(x.shape)
        break