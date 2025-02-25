import pytest
import torch
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles

@pytest.fixture
def dummy_dataset(imsize, device):
    return DummyCircles(samples=10, imsize=imsize)

@pytest.fixture
def image(imsize):
    x = torch.zeros(imsize).unsqueeze(0)
    x[..., x.shape[-2] // 2 - 2:x.shape[-2] // 2 + 2, x.shape[-1] // 2 - 2:x.shape[-1] // 2 + 2] = 1
    return x

def test_update_parameters(dummy_dataset, image, imsize, device, rng):
    physics = dinv.physics.Inpainting(imsize, device=device, rng=rng)
    physics.set_noise_model(dinv.physics.GaussianNoise())
    
    mask_generator  = dinv.physics.generator.GaussianSplittingMaskGenerator(imsize, 0.6, rng=rng, device=device)
    sigma_generator = dinv.physics.generator.SigmaGenerator(0.01, 0.5, rng=rng, device=device)
    physics_generator = mask_generator + sigma_generator

    pth = dinv.datasets.generate_dataset(
        dummy_dataset,
        physics=physics,
        save_dir=".",
        dataset_filename="test_update_parameters",
        physics_generator=physics_generator,
        batch_size=1,
    )
    train_dataset = dinv.datasets.HDF5Dataset(pth, train=True, load_physics_generator_params=True)
    test_dataset  = dinv.datasets.HDF5Dataset(pth, train=False, load_physics_generator_params=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, rng=rng)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    # Basics: physics generator returns random masks + sigmas
    x0, y0, params0 = next(iter(train_dataloader))
    x1, y1, params1 = next(next(iter(train_dataloader)))
    assert not torch.all(params0["mask"] == params1["mask"])
    assert not torch.all(params0["sigma"] == params1["sigma"])

    