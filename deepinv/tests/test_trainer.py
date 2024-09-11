import pytest
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles
import torch


NO_LEARNING = ["A_dagger", "A_adjoint", "prox_l2", "y"]


@pytest.fixture
def imsize():
    return (3, 5, 3)


@pytest.fixture
def physics(imsize, device):
    # choose a forward operator
    filter = torch.ones((1, 1, 3, 3), device=device) / 9
    return dinv.physics.BlurFFT(img_size=imsize, filter=filter, device=device)


@pytest.mark.parametrize("no_learning", NO_LEARNING)
def test_nolearning(imsize, physics, no_learning, device):
    model = lambda x, physics: x
    y = torch.ones((1,) + imsize, device=device)
    trainer = dinv.Trainer(
        model=model,
        train_dataloader=[],
        optimizer=None,
        losses=[],
        physics=physics,
        compare_no_learning=True,
        no_learning_method=no_learning,
    )
    x_hat = trainer.no_learning_inference(y, physics)
    assert (physics.A(x_hat) - y).pow(2).mean() < 0.1
