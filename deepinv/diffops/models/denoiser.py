import copy
import torch
import torch.nn as nn
try:
    import bm3d
except:
    print('WARNING: could not load BM3D')
import numpy as np


models = {}

# EXPERIMENTAL
# TAKEN FROM https://github.com/jaewon-lee-b/lte/blob/main/models/models.py
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
# END EXPERIMENTAL

def tensor2array(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)

class Denoiser(nn.Module):
    def __init__(self, denoiser_name, device, n_channels=3, pretrain=True, ckpt_path=None, train=False, n_it_max=1000,
                 verbose=False, model_spec=None):
        '''
        '''
        super(Denoiser, self).__init__()
        self.denoiser_name = denoiser_name
        self.device = device
        if 'drunet' in self.denoiser_name:
            from deepinv.diffops.models.drunet import UNetRes
            if 'tiny' in self.denoiser_name:
                self.model = UNetRes(in_channels=n_channels+1, out_channels=n_channels, nb = 2, nc=[16, 32, 64, 64])
            else :
                self.model = UNetRes(in_channels=n_channels+1, out_channels=n_channels, nb = 4, nc=[64, 128, 256, 512])
            if pretrain and ckpt_path is not None:
                self.model.load_state_dict(torch.load(ckpt_path), strict=True)
            if not train:
                self.model.eval()
                for _, v in self.model.named_parameters():
                    v.requires_grad = False
            self.model = self.model.to(device)
        elif self.denoiser_name == 'BM3D':
            self.model = lambda x,sigma : torch.cat([array2tensor(bm3d.bm3d(tensor2array(xi), sigma)) for xi in x])
        else:
            if model_spec is not None:
                self.model = make(model_spec)
            else:
                raise Exception("The denoiser chosen doesn't exist or model_spec not set")
        
    def forward(self, x, sigma):
        return self.model(x, sigma)