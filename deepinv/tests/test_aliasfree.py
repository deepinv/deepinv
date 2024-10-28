from deepinv.models import UNet
from deepinv.models.aliasfree import AliasFreeDenoiser
from deepinv.transform.utils import (
    shift_equivariance_error,
    translation_equivariance_error,
)

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# a translation-equivariant model
afc = AliasFreeDenoiser(size="tiny")
# a model neither translation-equivariant nor shift-equivariant
unet = UNet(in_channels=3, out_channels=3)


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
    for s in [0, 1 / (2 * 256), 0.5 + 1 / (2 * 256)]:
        err = translation_equivariance_error(afc, x, (s, s))
    assert torch.allclose(err, torch.zeros_like(err))


def test_not_translation_equivariant():
    x = torch.randn(1, 3, 256, 256)
    # for multiple displacements
    for s in [0, 1 / (2 * 256), 0.5 + 1 / (2 * 256)]:
        err = translation_equivariance_error(unet, x, (s, s))
    assert not torch.allclose(err, torch.zeros_like(err))
