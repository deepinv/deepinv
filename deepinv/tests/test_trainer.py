import pytest
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles
import torch
from torch.utils.data import DataLoader, Dataset


NO_LEARNING = ["A_dagger", "A_adjoint", "prox_l2", "y"]


@pytest.fixture
def imsize():
    return (3, 37, 31)


@pytest.fixture
def model():
    return torch.nn.Module()


@pytest.fixture
def physics(imsize, device):
    # choose a forward operator
    filter = torch.ones((1, 1, 3, 3), device=device) / 9
    return dinv.physics.BlurFFT(img_size=imsize, filter=filter, device=device)


@pytest.mark.parametrize("no_learning", NO_LEARNING)
def test_nolearning(imsize, physics, model, no_learning, device):
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


@pytest.mark.parametrize("use_physics_generator", [True, False])
@pytest.mark.parametrize("online_measurements", [True, False])
def test_get_samples(
    tmp_path,
    physics: dinv.physics.BlurFFT,
    model,
    device,
    dummy_dataset,
    use_physics_generator,
    online_measurements,
):
    class DummyDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return dummy_dataset[0]

    physics_generator = dinv.physics.generator.DiffractionBlurGenerator(psf_size=(5, 5))

    dataset_path = dinv.datasets.generate_dataset(
        DummyDataset(),
        physics=physics,
        physics_generator=physics_generator if use_physics_generator else None,
        save_dir=tmp_path / "dataset",
        device=device,
    )

    dataloader = DataLoader(
        dinv.datasets.HDF5Dataset(
            dataset_path,
            train=True,
            load_physics_generator_params=use_physics_generator,
        )
    )
    iterator = iter(dataloader)

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=dataloader,
        optimizer=None,
        online_measurements=online_measurements,
        physics_generator=(
            physics_generator if online_measurements and use_physics_generator else None
        ),
    )

    trainer.setup_train(train=True)
    x1, y1, physics1 = trainer.get_samples([iterator], g=0)
    param1 = (
        physics1.filter
    )  # take this out now as otherwise physics gets modified in place by next get_samples
    x2, y2, physics2 = trainer.get_samples([iterator], g=0)
    param2 = physics2.filter

    assert torch.all(x1 == x2)

    if not use_physics_generator:
        assert torch.all(y1 == y2)
        assert torch.all(param1 == param2)
    else:
        assert not torch.all(y1 == y2)
        assert not torch.all(param1 == param2)
