import torch
#from torchvision.transforms.functional import rotate
from kornia.geometry.transform import rotate


class Rotate(torch.nn.Module):
    r'''
        2D Rotations.

        Generates n_transf randomly rotated versions of 2D images with zero padding.
        :param n_trans: number of rotated versions generated per input image.
        :param degrees: images are rotated in the range of angles (-degrees, degrees)
    '''
    def __init__(self, n_trans, degrees=360):
        super(Rotate, self).__init__()
        self.n_trans, self.group_size = n_trans, degrees

    def forward(self, data):
        if self.group_size == 360:
            theta = torch.arange(0, 360)[1:][torch.randperm(359)]
            theta = theta[:self.n_trans].type_as(data)
        else:
            theta = torch.arange(0, 360, int(360 / (self.group_size+1)))[1:]
            theta = theta[torch.randperm(self.group_size)][:self.n_trans].type_as(data)
        return torch.cat([rotate(data, _theta) for _theta in theta])