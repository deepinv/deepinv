from deepinv.models import AliasFreeDenoiser, UNet
from deepinv.transform import Translate, Shift, Rotate
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.physics import BlurFFT, Inpainting
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.generator import BernoulliSplittingMaskGenerator

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
    err = Shift().equivariance_test(afc, x, metric=linf_metric)
    assert err < 1e-5


def test_not_shift_equivariant():
    err = Shift().equivariance_test(unet, x, metric=linf_metric)
    assert err >= 1e0


def test_translation_equivariant():
    err = Translate().equivariance_test(afc, x, metric=linf_metric)
    assert err < 1e-4


def test_not_translation_equivariant():
    err = Translate().equivariance_test(unet, x, metric=linf_metric)
    assert err >= 1e0


def test_forward_operator_equivariance():
    physics = BlurFFT(filter=gaussian_blur(sigma=1), img_size=x.shape[-3:])

    err = Shift().equivariance_test(physics, x, metric=linf_metric)
    assert err < 1e-6

    err = Translate().equivariance_test(physics, x, metric=linf_metric)
    assert err < 1e-6

    gen = BernoulliSplittingMaskGenerator(x.shape[-3:], split_ratio=0.7)
    params = gen.step(batch_size=1, seed=0)
    physics = Inpainting(tensor_size=x.shape[-3:])
    physics.update_parameters(**params)

    err = Shift().equivariance_test(physics, x, metric=linf_metric)
    assert err >= 1e-1

    err = Translate().equivariance_test(physics, x, metric=linf_metric)
    assert err >= 1e0

    err = Rotate().equivariance_test(physics, x, metric=linf_metric)
    assert err >= 1e-1
