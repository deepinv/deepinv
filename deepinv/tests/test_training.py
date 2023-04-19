import pytest
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w


def test_generate_dataset(tmp_path, imsize, device):
    N = 10
    max_N = 10
    train_dataset = DummyCircles(samples=N, imsize=imsize)
    test_dataset = DummyCircles(samples=N, imsize=imsize)

    physics = dinv.physics.Inpainting(mask=0.5, tensor_size=imsize, device=device)

    dinv.datasets.generate_dataset(
        train_dataset,
        physics,
        tmp_path,
        test_dataset=test_dataset,
        device=device,
        dataset_filename="dinv_dataset",
        max_datapoints=max_N,
    )

    dataset = dinv.datasets.HDF5Dataset(path=f"{tmp_path}/dinv_dataset0.h5", train=True)

    assert len(dataset) == min(max_N, N)

    x, y = dataset[0]
    assert x.shape == imsize
