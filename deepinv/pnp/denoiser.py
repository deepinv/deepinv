import torch
import torch.nn as nn
try:
    import bm3d
except:
    print('WARNING: could not load BM3D')
import numpy as np

def tensor2array(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1)

class Denoiser(nn.Module):
    def __init__(self, denoiser_name, device, n_channels=3, pretrain=True, ckpt_path=None, train=False, n_it_max=1000,
                 verbose=False):
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
                self.model = UNetRes(in_channels=n_channels+1, out_channels=n_channels, nb = 2, nc=[64, 128, 256, 512])
            if pretrain and ckpt_path is not None:
                self.model.load_state_dict(torch.load(ckpt_path), strict=True)
            if not train:
                self.model.eval()
                for _, v in self.model.named_parameters():
                    v.requires_grad = False
            self.model = self.model.to(device)
        elif self.denoiser_name == 'TGV':
            from deepinv.diffops.models.tgv import TGV
            self.model = TGV(n_it_max=n_it_max, verbose=verbose)
        else: 
            raise Exception("The denoiser chosen doesn't exist")
        
    def forward(self, x, sigma):
        if self.denoiser_name == 'BM3D':            #x = torch.cat((x, torch.tensor([sigma]).to(self.device).repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
            return torch.cat([array2tensor(bm3d.bm3d(tensor2array(xi), sigma)) for xi in x])
        elif 'drunet' in self.denoiser_name:
            noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(self.device)
            x = torch.cat((x, noise_level_map), 1)
            x = self.model(x)
            return x
        elif self.denoiser_name == 'TGV':
            return self.model(x, sigma)