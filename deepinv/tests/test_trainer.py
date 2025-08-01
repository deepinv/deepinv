import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import deepinv as dinv
from deepinv.utils import get_timestamp
from dummy import DummyCircles, DummyModel
from deepinv.training.trainer import Trainer
from deepinv.physics.generator.base import PhysicsGenerator
from deepinv.physics.forward import Physics
from deepinv.physics.noise import GaussianNoise, PoissonNoise

from unittest.mock import patch
import math
import io
import contextlib
import re

# NOTE: It's used as a fixture.
from conftest import non_blocking_plots  # noqa: F401


NO_LEARNING = ["A_dagger", "A_adjoint", "prox_l2", "y"]


@pytest.fixture
def imsize():
    return (3, 37, 31)


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def physics(imsize, device):
    # choose a forward operator
    filter = torch.ones((1, 1, 3, 3), device=device) / 9
    return dinv.physics.BlurFFT(img_size=imsize, filter=filter, device=device)


@pytest.mark.parametrize("no_learning", NO_LEARNING)
def test_nolearning(imsize, physics, model, no_learning, device, tmpdir):
    y = torch.ones((1,) + imsize, device=device)
    trainer = dinv.Trainer(
        model=model,
        train_dataloader=[],
        optimizer=None,
        losses=[],
        physics=physics,
        compare_no_learning=True,
        no_learning_method=no_learning,
        save_path=tmpdir,
    )
    x_hat = trainer.no_learning_inference(y, physics)
    assert (physics.A(x_hat) - y).pow(2).mean() < 0.1


def get_dummy_dataset(imsize, N, value):

    class DummyDataset(Dataset):
        r"""
        Defines a constant value image dataset
        """

        def __init__(self, value=1.0):
            self.value = value

        def __getitem__(self, i):
            return torch.ones(imsize) * self.value

        def __len__(self):
            return N

    return DummyDataset(value=value)


def get_dummy_physics(rng):
    r"""
    Returns a physics object with a Gaussian noise model
    """

    class DummyPhysics(Physics):
        # Dummy physics which sums images, and multiplies by a parameter f
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.f = 1

        def A(self, x: torch.Tensor, f: float = None, **kwargs) -> float:
            # NOTE for training with get_samples_online, this following line is technically redundant
            self.update_parameters(f=f)
            return x

        def update_parameters(self, f=None, **kwargs):
            self.f = f if f is not None else self.f

    physics = DummyPhysics()
    physics.set_noise_model(GaussianNoise(rng=rng, sigma=1e-4))
    return physics


def get_dummy_physics_generator(rng, device):
    class DummyPhysicsGenerator(PhysicsGenerator):
        # Dummy generator that outputs random factors
        def step(self, batch_size=1, seed=None, **kwargs):
            self.rng_manual_seed(seed)
            return {
                "f": torch.rand((batch_size,), generator=self.rng, device=device).item()
            }

    return DummyPhysicsGenerator(rng=rng, device=device)


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
    tmpdir,
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
        physics = dinv.physics.Inpainting(
            img_size=imsize, device=device, rng=rng, mask=0.1
        )
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
        save_path=tmpdir,
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
    imsize, loop_random_online_physics, noise, rng, device, model
):
    N = 10
    rng1 = rng
    rng2 = torch.Generator(device).manual_seed(0)

    class DummyPhysics(Physics):
        # Dummy physics which sums images, and multiplies by a parameter f
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.f = 1

        def A(self, x: torch.Tensor, f: float = None, **kwargs) -> float:
            # NOTE for training with get_samples_online, this following line is technically redundant
            self.update_parameters(f=f)
            return x.sum() * self.f

        def update_parameters(self, f: float, **kwargs):
            self.f = f
            super().update_parameters(**kwargs)

    physics = DummyPhysics()
    if noise == "gaussian":
        physics.set_noise_model(GaussianNoise(rng=rng1))
    elif noise == "poisson":
        physics.set_noise_model(PoissonNoise(rng=rng1))

    class SkeletonTrainer(Trainer):
        # hijack the step method to output samples to list
        ys = []
        fs = []

        def step(self, *args, **kwargs):
            x, y, physics_cur = self.get_samples(self.current_train_iterators, 0)
            self.ys += [y.item()]
            self.fs += [physics_cur.f]

    trainer = SkeletonTrainer(
        model=model.to(device),
        physics=physics,
        optimizer=None,
        train_dataloader=DataLoader(
            get_dummy_dataset(imsize=imsize, N=N, value=1.0)
        ),  # NO SHUFFLE
        online_measurements=True,
        physics_generator=get_dummy_physics_generator(rng=rng2, device=device),
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
        assert all(
            [a == b for (a, b) in zip(trainer.ys[:N], trainer.ys[N:], strict=True)]
        )
        assert all(
            [a == b for (a, b) in zip(trainer.fs[:N], trainer.fs[N:], strict=True)]
        )
    else:
        # Test measurements random but don't repeat
        # This is ok for supervised training but not self-supervised!
        assert len(set(trainer.ys)) == len(set(trainer.fs)) == N * 2
        assert all(
            [a != b for (a, b) in zip(trainer.ys[:N], trainer.ys[N:], strict=True)]
        )
        assert all(
            [a != b for (a, b) in zip(trainer.fs[:N], trainer.fs[N:], strict=True)]
        )


def test_trainer_identity(imsize, rng, device):
    r"""
    A simple test to check that the trainer manages to learn specific functions.

    We follow the setup from above with added noise and custom physics to check the behaviour with physics generators.

    In this test, we check that a model can learn the identity function on several datasets simultaneously.
    """
    N = 10

    mean_value_dataset_0 = -0.4
    mean_value_dataset_1 = 1.9

    list_physics = [get_dummy_physics(rng=rng), get_dummy_physics(rng=rng)]
    list_generators = [
        get_dummy_physics_generator(rng=rng, device=device),
        get_dummy_physics_generator(rng=rng, device=device),
    ]
    list_dataloaders = [
        DataLoader(
            get_dummy_dataset(imsize=imsize, N=N, value=mean_value_dataset_0),
            batch_size=1,
        ),
        DataLoader(
            get_dummy_dataset(imsize=imsize, N=N, value=mean_value_dataset_1),
            batch_size=1,
        ),
    ]

    class DummyModel(torch.nn.Module):
        r"""
        If physics = Identity, then this model outputs A(x)=x * param.
        """

        def __init__(self) -> None:
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        def forward(self, y=0.0, physics=None, **kwargs):
            return self.dummy_param * y

    dummy_model = DummyModel()
    dummy_model.to(device)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-2, weight_decay=0.0)

    trainer = Trainer(
        model=dummy_model,
        physics=list_physics,
        optimizer=optimizer,
        train_dataloader=list_dataloaders,  # NO SHUFFLE
        online_measurements=True,
        physics_generator=list_generators,
        loop_random_online_physics=True,  # IMPORTANT
        optimizer_step_multi_dataset=True,  # this is what we test in this function
        epochs=100,
        device=device,
        save_path=None,
        verbose=False,
        show_progress_bar=False,
    )

    trainer.train()

    # the model should learn the identity, i.e. dummy_parm = 1.0
    assert torch.isclose(dummy_model.dummy_param, torch.tensor(1.0), atol=1e-6)


def test_trainer_multidatasets(imsize, rng, device):
    r"""
    A simple test to check that the trainer manages to learn specific functions.

    We follow the setup from above with added noise and custom physics to check the behaviour with physics generators.

    In this test, we train a model to learn the average of two datasets.
    """
    N = 10

    mean_value_dataset_0 = -0.4
    mean_value_dataset_1 = 1.9

    list_physics = [get_dummy_physics(rng=rng), get_dummy_physics(rng=rng)]
    list_generators = [
        get_dummy_physics_generator(rng=rng, device=device),
        get_dummy_physics_generator(rng=rng, device=device),
    ]
    list_dataloaders = [
        DataLoader(
            get_dummy_dataset(imsize=imsize, N=N, value=mean_value_dataset_0),
            batch_size=1,
        ),
        DataLoader(
            get_dummy_dataset(imsize=imsize, N=N, value=mean_value_dataset_1),
            batch_size=1,
        ),
    ]

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        def forward(self, y=0.0, physics=None, **kwargs):
            return self.dummy_param * torch.ones_like(y)

    dummy_model = DummyModel()
    dummy_model.to(device)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-2, weight_decay=0.0)

    trainer = Trainer(
        model=dummy_model,
        physics=list_physics,
        optimizer=optimizer,
        train_dataloader=list_dataloaders,  # NO SHUFFLE
        online_measurements=True,
        physics_generator=list_generators,
        loop_random_online_physics=True,  # IMPORTANT
        optimizer_step_multi_dataset=True,  # this is what we test in this function
        epochs=100,
        device=device,
        save_path=None,
        verbose=False,
        show_progress_bar=False,
    )

    trainer.train()

    avg_value = (mean_value_dataset_0 + mean_value_dataset_1) / 2.0

    assert torch.isclose(dummy_model.dummy_param, torch.tensor(avg_value), atol=1e-6)


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


def test_trainer_test_metrics(non_blocking_plots, device, rng):
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


@pytest.mark.parametrize("ground_truth", [True, False])
@pytest.mark.parametrize("measurements", [True, False])
@pytest.mark.parametrize("online_measurements", [True, False])
@pytest.mark.parametrize("generate_params", [True, False])
def test_dataloader_formats(
    non_blocking_plots,
    imsize,
    device,
    dummy_model,
    generate_params,
    ground_truth,
    measurements,
    online_measurements,
    rng,
    tmpdir,
):
    """Test dataloader return formats

    :param bool ground_truth: whether dataset return x
    :param bool measurements: whether dataset return y
    :param bool generate_params: whether dataset return params
    :param bool online_measurements: whether trainer overrides measurements online
    """
    if not ground_truth and not measurements:
        pytest.skip("Must be some data returned")

    if online_measurements and not ground_truth:
        pytest.skip("Online measurements require ground truth.")

    if not measurements and not online_measurements:
        pytest.skip("Measurements are neither loaded nor generated online")

    # Offline generator at low split ratio
    generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        img_size=imsize, split_ratio=0.1, rng=rng, device=device
    )

    class DummyDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, i):
            params = generator.step(1)
            # NOTE: The test relies on changing params in place.
            params["mask"] = params["mask"].squeeze(0)
            mask = params["mask"]
            x = torch.ones(imsize, device=mask.device, dtype=mask.dtype)
            y = x * mask
            if ground_truth:
                if measurements:
                    if generate_params:
                        return x, y, params
                    else:
                        return x, y
                else:
                    if generate_params:
                        return x, params
                    else:
                        return x
            else:
                if measurements:
                    if generate_params:
                        return torch.nan, y, params
                    else:
                        return torch.nan, y
                else:
                    raise ValueError("Some data must be returned")

    model = dummy_model
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1)
    physics = dinv.physics.Inpainting(img_size=imsize, mask=1.0, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    losses = dinv.loss.MCLoss() if not ground_truth else dinv.loss.SupLoss()

    # Online generator at higher split ratio
    generator2 = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        img_size=imsize, split_ratio=0.9, rng=rng, device=device
    )

    trainer = dinv.Trainer(
        model=model,
        losses=losses,
        plot_images=True,
        epochs=1,
        physics=physics,
        physics_generator=generator2,
        metrics=dinv.loss.MCLoss(),
        online_measurements=online_measurements,
        train_dataloader=dataloader,
        optimizer=optimizer,
        save_path=tmpdir,
    )
    trainer.setup_train()
    x, y, physics = trainer.get_samples([iter(dataloader)], 0)

    # fmt: off
    def assert_x_none(x): assert x is None
    def assert_x_full(x): assert math.isclose(x.mean(), 1.0, abs_tol=1e-7)
    def assert_physics_unchanged(physics): assert math.isclose(physics.mask.mean(), 1.0, abs_tol=1e-7) # params not loaded
    def assert_physics_offline(physics): assert physics.mask.mean() < .2
    def assert_physics_online(physics): assert physics.mask.mean() > .8
    def assert_y_offline(y): assert y.mean() < .2
    def assert_y_online(y): assert y.mean() > .8

    if ground_truth:
        if online_measurements:
            if measurements:
                if generate_params: # x, y, params online, both y and physics ignored
                    assert_x_full(x); assert_y_online(y); assert_physics_online(physics)
                else: # x, y online, y ignored
                    assert_x_full(x); assert_y_online(y); assert_physics_online(physics)
            else:
                if generate_params: # x, params online, params ignored
                    assert_x_full(x); assert_y_online(y); assert_physics_online(physics)
                else: # x online
                    assert_x_full(x); assert_y_online(y); assert_physics_online(physics)
        else:
            if measurements:
                if generate_params: # x, y, params offline
                    assert_x_full(x); assert_y_offline(y); assert_physics_offline(physics)
                else: # x, y offline
                    assert_x_full(x); assert_y_offline(y); assert_physics_unchanged(physics)
            else:
                raise ValueError("measurements are neither loaded nor generated")
    else:
        if online_measurements:
            raise ValueError("online measurements requires GT")
        if not measurements:
            raise ValueError("some data must be returned")
        if generate_params: # y, params
            assert_x_none(x); assert_y_offline(y); assert_physics_offline(physics)
        else: # y
            assert_x_none(x); assert_y_offline(y); assert_physics_unchanged(physics)
    # fmt: off

    # Check that the model is trained without errors
    trainer.train()


@pytest.mark.parametrize("early_stop", [True, False])
@pytest.mark.parametrize("max_batch_steps", [3, 100000])
def test_early_stop(
    non_blocking_plots,
    dummy_dataset,
    imsize,
    device,
    dummy_model,
    early_stop,
    max_batch_steps,
    tmpdir,
):
    torch.manual_seed(0)
    model = dummy_model
    # split dataset
    epochs = 100 if early_stop else 4
    train_data, eval_data = dummy_dataset, dummy_dataset
    dataloader = DataLoader(train_data, batch_size=2)
    eval_dataloader = DataLoader(eval_data, batch_size=2)
    physics = dinv.physics.Inpainting(img_size=imsize, device=device, mask=0.5)
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
        save_path=tmpdir,
    )
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


class ConstantLoss(dinv.loss.Loss):
    def __init__(self, value, device):
        super().__init__()
        self.value = value
        self.device = device

    def forward(self, *args, **kwargs):
        return torch.tensor(
            self.value, device=self.device, dtype=torch.float32, requires_grad=True
        )


def test_total_loss(dummy_dataset, imsize, device, dummy_model, tmpdir):
    train_data, eval_data = dummy_dataset, dummy_dataset
    dataloader = DataLoader(train_data, batch_size=2)
    eval_dataloader = DataLoader(eval_data, batch_size=2)
    physics = dinv.physics.Inpainting(img_size=imsize, device=device, mask=0.5)

    losses = [
        ConstantLoss(1 / 2, device),
        ConstantLoss(1 / 3, device),
    ]

    trainer = dinv.Trainer(
        model=dummy_model,
        losses=losses,
        epochs=2,
        physics=physics,
        train_dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=torch.optim.AdamW(dummy_model.parameters(), lr=1),
        verbose=False,
        online_measurements=True,
        save_path=tmpdir,
    )

    trainer.train()

    loss_history = trainer.loss_history
    assert all(
        [abs(value - sum([l.value for l in losses])) < 1e-6 for value in loss_history]
    )


# We test that the gradient norm is correctly computed and printed to the
# standard output. To do that, we mock backprop to control the gradient norms.
# More precisely, we make it so the gradient norm is 1.0 for epoch 1, 2.0 for
# epoch 2, and so on. Then, we run the trainer while capturing the standard
# output to get # the reported values for the gradient norms and compare them
# to the expected values.
def test_gradient_norm(dummy_dataset, imsize, device, dummy_model, tmpdir):
    train_data, eval_data = dummy_dataset, dummy_dataset
    dataloader = DataLoader(train_data, batch_size=2)
    physics = dinv.physics.Inpainting(img_size=imsize, device=device, mask=0.5)

    backbone = dinv.models.UNet(in_channels=3, out_channels=3, scales=2)
    model = dinv.models.ArtifactRemoval(backbone).to(device)

    trainer = dinv.Trainer(
        model,
        device=device,
        save_path=tmpdir,
        verbose=True,
        show_progress_bar=False,
        physics=physics,
        epochs=10,
        losses=dinv.loss.SupLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_dataloader=dataloader,
        online_measurements=True,
        check_grad=True,
    )

    call_count = 0
    calls_per_epoch = math.ceil(len(train_data) / dataloader.batch_size)

    def mock_fn(self):
        nonlocal call_count
        epoch = 1 + call_count // calls_per_epoch
        call_count += 1

        # 1. Fill in the gradients with random values.
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p, dtype=p.dtype, device=p.device)

        # 2. Compute the norm of the gradients.
        # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/7
        grads = [
            p.grad.detach().flatten() for p in model.parameters() if p.grad is not None
        ]
        grads = torch.cat(grads)
        norm = grads.norm()
        norm = norm.item()

        # 3. Rescale the gradients so that the gradient norm is equal to 1.0 for
        # the 1st epoch, to 2.0 for the 2nd, and so on.
        for p in model.parameters():
            if p.requires_grad:
                p.grad = epoch * p.grad / norm

    # Capture the standard output for future testing
    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        with patch.object(torch.Tensor, "backward", mock_fn):
            trainer.train()

    stdout_value = stdout_buf.getvalue()

    gradient_norms = re.findall(r"gradient_norm=(\d+(\.\d+)?)", stdout_value)
    gradient_norms = [float(norm[0]) for norm in gradient_norms]
    gradient_norms = torch.tensor(gradient_norms)
    expected_gradient_norms = [float(epoch) for epoch in range(1, trainer.epochs + 1)]
    expected_gradient_norms = torch.tensor(expected_gradient_norms)
    assert torch.allclose(gradient_norms, expected_gradient_norms, atol=1e-2)


# Test output directory collision detection
# It is difficult to deterministically trigger actual collisions so we mock the
# get_timestamp function used in the implementation to make it return the same
# value every time it is called. This forces a collision to occur and we make
# sure that it is detected as it should.
def test_out_dir_collision_detection(
    dummy_dataset, imsize, device, dummy_model, tmpdir
):
    train_data, eval_data = dummy_dataset, dummy_dataset
    dataloader = DataLoader(train_data, batch_size=2)
    physics = dinv.physics.Inpainting(img_size=imsize, device=device, mask=0.5)

    backbone = dinv.models.UNet(in_channels=3, out_channels=3, scales=2)
    model = dinv.models.ArtifactRemoval(backbone).to(device)

    timestamp = get_timestamp()

    # NOTE: Due to the way it's imported in the trainer module we need to patch
    # the importing module instead of the imported module.
    with patch.object(dinv.training.trainer, "get_timestamp", return_value=timestamp):
        with pytest.raises(FileExistsError, match=re.escape(timestamp)):
            # Train twice
            for _ in range(2):
                trainer = dinv.Trainer(
                    model,
                    device=device,
                    save_path=tmpdir,
                    verbose=True,
                    show_progress_bar=False,
                    physics=physics,
                    epochs=2,
                    losses=dinv.loss.SupLoss(),
                    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                    train_dataloader=dataloader,
                    online_measurements=True,
                    check_grad=True,
                )

                trainer.train()
