from deepinv.models import AliasFreeUNet, UNet
from deepinv.transform import Translate, Shift, Rotate
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.physics import BlurFFT, Inpainting
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.generator import BernoulliSplittingMaskGenerator
from deepinv.loss.metric import PSNR

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode


torch.manual_seed(0)

# a translation-equivariant model
afc = AliasFreeUNet(in_channels=1, out_channels=1, scales=2)
# a translation- and rotation-equivariant model
afc_rotation_equivariant = AliasFreeUNet(in_channels=1, out_channels=1, scales=2, rotation_equivariant=True)
# a model neither translation-equivariant nor shift-equivariant
unet = UNet(in_channels=1, out_channels=1, scales=2)

x = DummyCircles(1, imsize=(1, 32, 32))[0].unsqueeze(0)
metric = PSNR()

rotate_kwargs = {
    "interpolation_mode": InterpolationMode.BILINEAR,
    "padding": "circular",
}

rotate_params = {"theta": [15]}


def test_nonsquare_input():
    x = DummyCircles(1, imsize=(1, 32, 16))[0].unsqueeze(0)
    y = afc(x)
    assert y.shape == x.shape
    y = afc_rotation_equivariant(x)
    assert y.shape == x.shape


def test_shift_equivariant():
    err = Shift().equivariance_test(afc, x, metric=metric)
    assert err >= 75

    err = Shift().equivariance_test(afc_rotation_equivariant, x, metric=metric)
    assert err >= 35


def test_not_shift_equivariant():
    err = Shift().equivariance_test(unet, x, metric=metric)
    assert err < 15


def test_translation_equivariant():
    err = Translate().equivariance_test(afc, x, metric=metric)
    assert err >= 65

    err = Translate().equivariance_test(afc_rotation_equivariant, x, metric=metric)
    assert err >= 45


def test_not_translation_equivariant():
    err = Translate().equivariance_test(unet, x, metric=metric)
    assert err < 10


def test_rotation_equivariant():
    psnr = Rotate(**rotate_kwargs).equivariance_test(
        afc_rotation_equivariant, x, params=rotate_params, metric=metric
    )
    assert psnr >= 25


def test_not_rotation_equivariant():
    psnr = Rotate(**rotate_kwargs).equivariance_test(
        unet, x, params=rotate_params, metric=metric
    )
    assert psnr <= 15
