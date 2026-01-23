"""
FBI-Denoiser: Fast Blind Image Denoiser for Poisson-Gaussian Noise (CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

def conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1)

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=True):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2) if pooling else None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return (self.pool(x), x) if self.pool else (x, x)

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, merge_mode='add'):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = conv3x3(2*out_ch if merge_mode == 'concat' else out_ch, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.merge_mode = merge_mode

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat([from_up, from_down], 1) if self.merge_mode == 'concat' else from_up + from_down
        return F.relu(self.conv2(F.relu(self.conv1(x))))

class PGENet(nn.Module):
    def __init__(self, num_classes=2, in_ch=1, depth=3, start_filts=64, merge_mode='add', pretrained=None):
        super().__init__()
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        outs = in_ch
        for i in range(depth):
            ins = in_ch if i == 0 else outs
            outs = start_filts * (2 ** i)
            self.down_convs.append(DownConv(ins, outs, pooling=(i < depth - 1)))
        
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            self.up_convs.append(UpConv(ins, outs, merge_mode))
        
        self.conv_final = conv1x1(outs, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))

    def forward(self, x):
        encoder_outs = []
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            x = module(encoder_outs[-(i + 2)], x)
        return self.conv_final(x)



if __name__ == '__main__':
    import deepinv as dinv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pge = PGENet().to(device)

    ckpt_pth = '/pth/to/weights/synthetic_noise/211127_PGE_Net_RawRGB_random_noise_cropsize_200.w'
    state_dict = torch.load(ckpt_pth, map_location=device)
    del state_dict['noiseSTD']
    pge.load_state_dict(state_dict)

    x = dinv.utils.load_example("butterfly.png", device=device, grayscale=True)
    sigma_true = 0.1
    gain_true = 0.001

    physics = dinv.physics.Denoising(noise_model=dinv.physics.PoissonGaussianNoise(gain=gain_true, sigma=sigma_true, clip_positive=True)).to(device)
    y = physics(x)

    with torch.no_grad():
        out = pge(y)

    print('Out = ', out[:, 0].mean().item(), out[:, 1].mean().item())
