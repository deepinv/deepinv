from deepinv.models import AliasFreeDenoiser, UNet
from deepinv.transform import Translate, Shift
from deepinv.tests.dummy_datasets.datasets import DummyCircles

import torch
import torch.nn.functional as F


torch.manual_seed(0)

# a translation-equivariant model
afc = AliasFreeDenoiser(in_channels=3, out_channels=3)
# a model neither translation-equivariant nor shift-equivariant
unet = UNet(in_channels=3, out_channels=3)

x = DummyCircles(1, imsize=(3, 256, 256))[0].unsqueeze(0)
linf_metric = lambda x, y: (x - y).abs().max()


def test_shift_equivariant():
    err = Shift().equivariance_error(afc, x, metric=linf_metric)
    assert err < 1e-5


def test_not_shift_equivariant():
    err = Shift().equivariance_error(unet, x, metric=linf_metric)
    assert err >= 1e0


def test_translation_equivariant():
    err = Translate().equivariance_error(afc, x, metric=linf_metric)
    assert err < 1e-4


def test_not_translation_equivariant():
    err = Translate().equivariance_error(unet, x, metric=linf_metric)
    assert err >= 1e0
