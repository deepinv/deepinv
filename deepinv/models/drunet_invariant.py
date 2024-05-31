import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import *

# Code inspired from https://github.com/cszn/DPIR/blob/master/models/network_unet.py

class DRUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, channels=[64, 128, 256, 512], num_blocks=4, blindness=False, mode='scale-equiv'):
        super().__init__()

        bias = mode == 'ordinary'
        self.blind = blindness
        if not blindness: input_channels += 1

        self.head_layer = conv2d(input_channels, channels[0], 3, stride=1, padding=1, bias=bias, mode=mode)
        
        self.downsampling_layers = nn.ModuleList([nn.Sequential(
             *[ResBlock(channels[i], channels[i], bias=bias, mode=mode) for _ in range(num_blocks)],
            conv2d(channels[i], channels[i+1], 2, stride=2, padding=0, bias=bias, mode=mode))
            for i in range(len(channels)-1)])

        self.body = nn.Sequential(*[ResBlock(channels[-1], channels[-1], bias=bias, mode=mode) for _ in range(num_blocks)])

        self.upsampling_layers = nn.ModuleList([nn.Sequential(
            upscale2(channels[i], channels[i-1], bias=bias, mode=mode),
             *[ResBlock(channels[i-1], channels[i-1], bias=bias, mode=mode) for _ in range(num_blocks)])
            for i in range(len(channels)-1, 0, -1)])

        self.tail_layer = conv2d(channels[0], output_channels, 3, stride=1, padding=1, bias=bias, mode=mode)

        self.residual_connections = nn.ModuleList([ResidualConnection(mode) for _ in range(len(channels))])
        

    def forward(self, x, noise_level=None) -> torch.Tensor:
        _, _, height, width = x.size()
        downsampling_scale = len(self.downsampling_layers)
        downsample = 2**downsampling_scale
        remainder1, remainder2 = height % downsample, width % downsample
        x = F.pad(x, pad=(0, downsample-remainder2 if remainder2 >0 else 0, 0, downsample-remainder1 if remainder1 >0 else 0), mode='constant', value=float(x.mean()))
        
        if not self.blind: 
            assert noise_level is not None
            noisemap = noise_level * torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            x = torch.cat((x, noisemap), dim=1)

        layers = [self.head_layer(x)]
        for i in range(downsampling_scale): layers.append(self.downsampling_layers[i](layers[-1]))
        x = self.body(layers[-1])
        for i in range(downsampling_scale): x = self.upsampling_layers[i](self.residual_connections[i](x, layers[-(1+i)]))
        x = self.tail_layer(self.residual_connections[-1](x, layers[0]))
        
        return x[..., :height, :width]