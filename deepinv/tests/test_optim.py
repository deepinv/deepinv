import pytest
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles

from torch.utils.data import DataLoader


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w

# TODO: use a DummyCircle as dataset and check convergence of optim algorithms (maybe with TV denoiser)
def test_generate_dataset(tmp_path, imsize, device):
    N = 10
    max_N = 10
    train_dataset = DummyCircles(samples=N, imsize=imsize)
    test_dataset = DummyCircles(samples=N, imsize=imsize)

    physics = dinv.physics.Inpainting(mask=.5, tensor_size=imsize, device=device)
    dinv.datasets.generate_dataset(train_dataset, physics, tmp_path, test_dataset=test_dataset, device=device,
                     dataset_filename='dinv_dataset', max_datapoints=max_N)

    dataset = dinv.datasets.HDF5Dataset(path=f'{tmp_path}/dinv_dataset0.h5', train=True)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

    return physics, dataloader


def test_prox_tgv(imsize, device):
    tmp_path = 'deepinv/datasets/'
    physics, dataloader = test_generate_dataset(tmp_path, imsize, device)

    backbone = dinv.models.tgvprox(reg=3, n_it_max=1000, crit=1e-4, convergence_test=True)
    model = dinv.models.ArtifactRemoval(backbone)

    kwargs = {'sigma': 0.2}

    dinv.training_utils.test(model=model,
         test_dataloader=dataloader,
         physics=physics,
         plot=False,
         device=dinv.device,
         **kwargs)

