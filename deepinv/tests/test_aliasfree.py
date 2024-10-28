from deepinv.models import UNet
from deepinv.models.aliasfree import AliasFreeDenoiser

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# a translation-equivariant model
afc = AliasFreeDenoiser(size="tiny")
# a model neither translation-equivariant nor shift-equivariant
unet = UNet(in_channels=3, out_channels=3)

def shift_equivariance_error(model, x, displacement):
    y1 = model(x).roll(displacement, dims=(2, 3))
    y2 = model(x.roll(displacement, dims=(2, 3)))
    return y1 - y2


def translation_equivariance_error(model, x, displacement):
    B, C, H, W = x.shape
    displacement_w, displacement_h = displacement

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )

    grid_x = grid_x + (displacement_w / W * 2)
    grid_y = grid_y + (displacement_h / H * 2)

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    kwargs = {
        'mode': 'bilinear',
        'padding_mode': 'border',
        'align_corners': False
    }

    y1 = model(F.grid_sample(x, grid, **kwargs))
    y2 = F.grid_sample(model(x), grid, **kwargs)
    return y1 - y2

def test_shift_equivariant():
    x = torch.randn(1, 3, 256, 256)
    # for multiple displacements
    for s in [0, 16, 32, 64, 128]:
        err = shift_equivariance_error(afc, x, (s, s))
    assert torch.allclose(err, torch.zeros_like(err))

def test_not_shift_equivariant():
    x = torch.randn(1, 3, 256, 256)
    # for multiple displacements
    for s in [0, 16, 32, 64, 128]:
        err = shift_equivariance_error(unet, x, (s, s))
    assert not torch.allclose(err, torch.zeros_like(err))

def test_translation_equivariant():
    x = torch.randn(1, 3, 256, 256)
    # for multiple displacements
    for s in [0, 1/(2 * 256), .5 + 1/(2 * 256)]:
        err = translation_equivariance_error(afc, x, (s, s))
    assert torch.allclose(err, torch.zeros_like(err))

def test_not_translation_equivariant():
    x = torch.randn(1, 3, 256, 256)
    # for multiple displacements
    for s in [0, 1/(2 * 256), .5 + 1/(2 * 256)]:
        err = translation_equivariance_error(unet, x, (s, s))
    assert not torch.allclose(err, torch.zeros_like(err))
