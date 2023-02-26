# Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
# TODO: missing the eval functions
import numpy as np
import torch
import torch.nn as nn

from .denoiser import register

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

@register('drunet')
class UNetRes(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                 downsample_mode='strideconv', upsample_mode='convtranspose', pretrain=False, ckpt_path=None,
                 device=None):
        super(UNetRes, self).__init__()

        self.m_head = conv(in_channels, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(*[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = sequential(*[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = sequential(*[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = sequential(*[ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = conv(nc[0], out_channels, bias=False, mode='C')

        if pretrain or ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path), strict=True)

        if device is not None:
            self.to(device)

    def forward_unet(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)
        return x

    def forward(self, x, sigma):
        noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(x.device)
        x = torch.cat((x, noise_level_map), 1)
        x = self.forward_unet(x)
        return x


class DRUNet(UNetRes):  # not needed?
    def __init__(self, **kwargs):
        super(UNetRes, self).__init__(**kwargs)

    def forward(self, x, sigma):
        apply_DRUNet(model, x, noise_level_model=sigma, x8=False)

'''
Functional blocks below
'''
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


'''
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
'''


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        #res = self.res(x)
        return x + self.res(x)


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


'''
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
'''


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


'''
Helpers for test time
'''

def apply_DRUNet(model, img_L, noise_level_model=0.01, x8=False):

    img_L = torch.cat(
            (img_L, torch.FloatTensor([noise_level_model]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)

    # ------------------------------------
    # (2) img_E
    # ------------------------------------

    if not x8 and img_L.size(2) // 8 == 0 and img_L.size(3) // 8 == 0:
        img_E = model(img_L)
    elif not x8 and (img_L.size(2) // 8 != 0 or img_L.size(3) // 8 != 0):
        img_E = test_mode(model, img_L, refield=64, mode=5)
    elif x8:
        raise NotImplementedError('x8 test mode not implemented!')

    return img_E

'''
Copyright (c) 2020 Kai Zhang (cskaizhang@gmail.com)
'''

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


# def single2tensor4(img):
#     return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
def single2tensor4(img):
    imgn = torch.from_numpy(img)
    init_shape = imgn.shape
    if len(init_shape) == 2:
        imgn.unsqueeze_(0)
        imgn.unsqueeze_(0)
    elif len(init_shape) == 3:
        imgn.unsqueeze_(0)
    return imgn.type(Tensor)


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        # img = np.transpose(img, (1, 2, 0))
        img = np.moveaxis(img, 0, -1)
    # return img*255
    return np.uint8((img*255.0).round())


def tensor2np(img):
    return img.data.squeeze().float().clamp_(0, 1).cpu().numpy()


def test_mode(model, L, mode=0, refield=32, min_size=256, sf=1, modulo=1):
    '''
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some testing modes
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (0) normal: test(model, L)
    # (1) pad: test_pad(model, L, modulo=16)
    # (2) split: test_split(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (3) x8: test_x8(model, L, modulo=1)
    # (4) split and x8: test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (5) split only once: test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # ---------------------------------------
    '''
    if mode == 0:
        E = test(model, L)
    elif mode == 1:
        E = test_pad(model, L, modulo)
    elif mode == 5:
        E = test_onesplit(model, L, refield, min_size, sf, modulo)
    return E


'''
# ---------------------------------------
# normal (0)
# ---------------------------------------
'''

def test(model, L):
    E = model(L)
    return E

'''
# ---------------------------------------
# pad (1)
# ---------------------------------------
'''


def test_pad(model, L, modulo=16):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L)
    E = E[..., :h, :w]
    return E


'''
# ---------------------------------------
# split (function)
# ---------------------------------------
'''


def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]
    if h*w <= min_size**2:
        L = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(L)
        E = model(L)
        E = E[..., :h*sf, :w*sf]
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)
        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]

        if h * w <= 4*(min_size**2):
            Es = [model(Ls[i]) for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E



'''
# ---------------------------------------
# split (2)
# ---------------------------------------
'''

def test_split(model, L, refield=32, min_size=256, sf=1, modulo=1):
    E = test_split_fn(model, L, refield=refield, min_size=min_size, sf=sf, modulo=modulo)
    return E
