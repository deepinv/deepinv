import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.training.trainer import Trainer
from deepinv.physics.generator.base import PhysicsGenerator
from deepinv.physics.forward import Physics
from deepinv.physics.noise import GaussianNoise, PoissonNoise

from conftest import no_plot

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


@pytest.mark.parametrize(
    "use_physics_generator", [None, "param", "noise", "param+noise"]
)
@pytest.mark.parametrize("online_measurements", [True, False])
@pytest.mark.parametrize("physics_type", ["blur", "inpainting"])
def test_get_samples(
    tmp_path,
    imsize,
    physics_type,
    model,
    device,
    dummy_dataset,
    use_physics_generator,
    online_measurements,
    rng,
):
    # Dummy constant GT dataset
    class DummyDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return dummy_dataset[0]

    # Define physics
    if physics_type == "blur":
        physics = dinv.physics.BlurFFT(
            img_size=imsize,
            filter=torch.ones((1, 1, 3, 3), device=device) / 9,
            device=device,
        )
        param_name = "filter"
    elif physics_type == "inpainting":
        physics = dinv.physics.Inpainting(tensor_size=imsize, device=device, rng=rng)
        param_name = "mask"

    # Define physics generator
    if use_physics_generator is None:
        physics_generator = None
    else:
        if "param" in use_physics_generator:
            if physics_type == "blur":
                param_generator = dinv.physics.generator.DiffractionBlurGenerator(
                    psf_size=(5, 5), rng=rng, device=device
                )
            elif physics_type == "inpainting":
                param_generator = dinv.physics.generator.GaussianSplittingMaskGenerator(
                    imsize, 0.6, device=device, rng=rng
                )
            physics_generator = param_generator
        if "noise" in use_physics_generator:
            noise_generator = dinv.physics.generator.SigmaGenerator(
                rng=rng, device=device
            )
            physics_generator = noise_generator
        if use_physics_generator == "param+noise":
            physics_generator = param_generator + noise_generator

    # Add noise to physics
    physics.set_noise_model(dinv.physics.GaussianNoise(sigma=0.1))

    # Generate dataset
    dataset_path = dinv.datasets.generate_dataset(
        DummyDataset(),
        physics=physics,
        physics_generator=physics_generator,
        save_dir=tmp_path / "dataset",
        device=device,
    )

    dataloader = DataLoader(
        dinv.datasets.HDF5Dataset(
            dataset_path,
            train=True,
            load_physics_generator_params=physics_generator is not None,
        )
    )

    iterator = iter(dataloader)

    if not online_measurements:
        if physics_generator is not None:
            # Test phys gen params change in offline dataset
            x1, y1, params1 = next(iterator)
            x2, y2, params2 = next(iterator)
            if "param" in use_physics_generator:
                assert not torch.all(params1[param_name] == params2[param_name])
            if "noise" in use_physics_generator:
                assert not torch.all(params1["sigma"] == params2["sigma"])
        else:
            # Test params don't exist in offline dataset
            assert len(next(iterator)) == 2  # (x, y)

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        train_dataloader=dataloader,
        optimizer=None,
        online_measurements=online_measurements,
        physics_generator=(
            physics_generator
            if online_measurements and physics_generator is not None
            else None
        ),
    )

    iterator = iter(dataloader)

    trainer.setup_train(train=True)
    x1, y1, physics1 = trainer.get_samples([iterator], g=0)
    # take this out now as otherwise physics gets modified in place by next get_samples
    param1 = getattr(physics1, param_name)
    sigma1 = physics1.noise_model.sigma
    x2, y2, physics2 = trainer.get_samples([iterator], g=0)
    param2 = getattr(physics2, param_name)
    sigma2 = physics2.noise_model.sigma

    # Test GT same in our dummy dataset
    assert torch.all(x1 == x2)

    if physics_generator is None:
        # Test params don't change when no phys gen
        assert torch.all(param1 == param2)
        assert torch.all(sigma1 == sigma2)
    else:
        # Test phys gen params change in both offline and online datasets
        assert not torch.all(y1 == y2)
        if use_physics_generator == "param":
            assert not torch.all(param1 == param2)
            assert torch.all(sigma1 == sigma2)
        elif use_physics_generator == "noise":
            assert torch.all(param1 == param2)
            assert not torch.all(sigma1 == sigma2)
        elif use_physics_generator == "param+noise":
            assert not torch.all(param1 == param2)
            assert not torch.all(sigma1 == sigma2)


@pytest.mark.parametrize("loop_random_online_physics", [True, False])
@pytest.mark.parametrize("noise", [None, "gaussian", "poisson"])
def test_trainer_physics_generator_params(
    imsize, loop_random_online_physics, noise, rng, device
):
    N = 10
    rng1 = rng
    rng2 = torch.Generator().manual_seed(0)

    class DummyDataset(Dataset):
        # Dummy dataset that returns equal blank images
        def __getitem__(self, i):
            return torch.ones(imsize)

        def __len__(self):
            return N

    class DummyPhysics(Physics):
        # Dummy physics which sums images, and multiplies by a parameter f
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.f = 1

        def A(self, x: torch.Tensor, f: float = None, **kwargs) -> float:
            # NOTE for training with get_samples_online, this following line is technically redundant
            self.update_parameters(f=f)
            return x.sum() * self.f

        def update_parameters(self, f=None, **kwargs):
            self.f = f if f is not None else self.f

    physics = DummyPhysics()
    if noise == "gaussian":
        physics.set_noise_model(GaussianNoise(rng=rng1))
    elif noise == "poisson":
        physics.set_noise_model(PoissonNoise(rng=rng1))

    class DummyPhysicsGenerator(PhysicsGenerator):
        # Dummy generator that outputs random factors
        def step(self, batch_size=1, seed=None, **kwargs):
            self.rng_manual_seed(seed)
            return {
                "f": torch.rand((batch_size,), generator=self.rng, device=device).item()
            }

    class SkeletonTrainer(Trainer):
        # hijack the step method to output samples to list
        ys = []
        fs = []

        def step(self, *args, **kwargs):
            x, y, physics_cur = self.get_samples(self.current_train_iterators, 0)
            self.ys += [y.item()]
            self.fs += [physics_cur.f]

    trainer = SkeletonTrainer(
        model=torch.nn.Module().to(device),
        physics=physics,
        optimizer=None,
        train_dataloader=DataLoader(DummyDataset()),  # NO SHUFFLE
        online_measurements=True,
        physics_generator=DummyPhysicsGenerator(rng=rng2),
        loop_random_online_physics=loop_random_online_physics,  # IMPORTANT
        epochs=2,
        device=device,
        save_path=None,
        verbose=False,
        show_progress_bar=False,
    )

    trainer.train()

    if loop_random_online_physics:
        # Test measurements random but repeat every epoch
        assert len(set(trainer.ys)) == len(set(trainer.fs)) == N
        assert all([a == b for (a, b) in zip(trainer.ys[:N], trainer.ys[N:])])
        assert all([a == b for (a, b) in zip(trainer.fs[:N], trainer.fs[N:])])
    else:
        # Test measurements random but don't repeat
        # This is ok for supervised training but not self-supervised!
        assert len(set(trainer.ys)) == len(set(trainer.fs)) == N * 2
        assert all([a != b for (a, b) in zip(trainer.ys[:N], trainer.ys[N:])])
        assert all([a != b for (a, b) in zip(trainer.fs[:N], trainer.fs[N:])])


def test_trainer_load_model(tmp_path):
    class TempModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.a = torch.nn.Parameter(torch.Tensor([1]), requires_grad=False)

    trainer = dinv.Trainer(
        model=TempModel(),
        physics=dinv.physics.Physics(),
        optimizer=None,
        train_dataloader=None,
    )

    torch.save({"state_dict": trainer.model.state_dict()}, tmp_path / "temp.pth")
    trainer.model.a *= 3
    trainer.load_model(tmp_path / "temp.pth")
    assert trainer.model.a == 1


def test_trainer_test_metrics(device, rng):
    N = 10
    dataloader = torch.utils.data.DataLoader(DummyCircles(N), batch_size=2)
    trainer = dinv.Trainer(
        model=dinv.models.MedianFilter().to(device),
        physics=dinv.physics.Inpainting((3, 32, 28), mask=0.8, device=device, rng=rng),
        train_dataloader=torch.utils.data.DataLoader(DummyCircles(3)),
        eval_dataloader=dataloader,
        optimizer=None,
        epochs=0,
        losses=dinv.loss.SupLoss(),
        save_path=None,
        verbose=False,
        show_progress_bar=False,
        device=device,
        online_measurements=True,
        plot_images=True,
    )
    with no_plot():
        _ = trainer.train()
        results = trainer.test(dataloader, log_raw_metrics=True)

    assert len(results["PSNR_vals"]) == len(results["PSNR no learning_vals"]) == N
    assert np.isclose(np.mean(results["PSNR_vals"]), results["PSNR"])
    assert np.isclose(np.std(results["PSNR_vals"]), results["PSNR_std"])
    assert np.isclose(
        np.mean(results["PSNR no learning_vals"]), results["PSNR no learning"]
    )
    assert np.isclose(
        np.std(results["PSNR no learning_vals"]), results["PSNR no learning_std"]
    )


@pytest.fixture
def dummy_model(device):
    class DummyModel(dinv.models.Reconstructor):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(1), requires_grad=True)
            self.median = dinv.models.MedianFilter()

        def forward(self, y, physics, **kwargs):
            x = physics.A_adjoint(y)
            return (x + self.param * self.median(x)) / 2.0

    return DummyModel().to(device)


def test_measurements_only(dummy_dataset, imsize, device, dummy_model):
    # Measurements only dataset
    class DummyDataset(Dataset):
        def __len__(self):
            return len(dummy_dataset)

        def __getitem__(self, i):
            return torch.nan, dummy_dataset[i]

    model = dummy_model
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1)
    physics = dinv.physics.Inpainting(tensor_size=imsize, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    losses = dinv.loss.MCLoss()

    trainer = dinv.Trainer(
        model=model,
        losses=losses,
        plot_images=True,
        epochs=1,
        physics=physics,
        metrics=dinv.loss.MCLoss(),
        online_measurements=False,
        train_dataloader=dataloader,
        optimizer=optimizer,
    )
    with no_plot():
        # Check that the model is trained without errors
        trainer.train()


@pytest.mark.parametrize("early_stop", [True, False])
@pytest.mark.parametrize("max_batch_steps", [3, 100000])
def test_early_stop(
    dummy_dataset, imsize, device, dummy_model, early_stop, max_batch_steps
):
    torch.manual_seed(0)
    model = dummy_model
    # split dataset
    epochs = 100 if early_stop else 4
    train_data, eval_data = dummy_dataset, dummy_dataset
    dataloader = DataLoader(train_data, batch_size=2)
    eval_dataloader = DataLoader(eval_data, batch_size=2)
    physics = dinv.physics.Inpainting(tensor_size=imsize, device=device, mask=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    losses = dinv.loss.MCLoss()
    trainer = dinv.Trainer(
        model=model,
        losses=losses,
        early_stop=early_stop,
        epochs=epochs,
        physics=physics,
        max_batch_steps=max_batch_steps,
        train_dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        online_measurements=True,
        optimizer=optimizer,
        verbose=False,
        plot_images=True,
    )
    with no_plot():
        trainer.train()

        metrics_history = trainer.eval_metrics_history["PSNR"]
        if max_batch_steps == 3:
            assert len(metrics_history) <= len(dataloader) * epochs
        elif early_stop:
            assert len(metrics_history) < epochs
            last = metrics_history[-1]
            best = max(metrics_history)
            metrics = trainer.test(eval_dataloader)
            assert metrics["PSNR"] < best and metrics["PSNR"] == last
        else:
            assert len(metrics_history) == epochs
