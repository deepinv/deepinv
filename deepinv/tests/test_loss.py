import pytest

import math
import numpy as np
import torch

import deepinv
from dummy import DummyCircles, DummyModel
from torch.utils.data import DataLoader
import deepinv as dinv
from deepinv.loss.regularisers import JacobianSpectralNorm, FNEJacobianSpectralNorm
from deepinv.loss.scheduler import RandomLossScheduler, InterleavedLossScheduler

# NOTE: It's used as a fixture.
from conftest import non_blocking_plots  # noqa: F401

LOSSES = [
    "sup",
    "sup_log_train_batch",
    "mcei",
    "mcei-scale",
    "mcei-homography",
    "r2r",
    "vortex",
]

LIST_SURE = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "GaussianUnknown",
    "PoissonGaussianUnknown",
]

LIST_R2R = [
    "Gaussian",
    "Poisson",
    "Gamma",
]


def test_jacobian_spectral_values(toymatrix):
    # Define the Jacobian regularisers we want to check
    reg_l2 = JacobianSpectralNorm(max_iter=100, tol=1e-4, eval_mode=False, verbose=True)
    reg_FNE_l2 = FNEJacobianSpectralNorm(
        max_iter=100, tol=1e-4, eval_mode=False, verbose=True
    )

    # Setup our toy example; here y = A@x
    x_detached = torch.randn((1, toymatrix.shape[0])).requires_grad_()
    out = x_detached @ toymatrix

    def model(x):
        return x @ toymatrix

    regl2 = reg_l2(out, x_detached)
    regfnel2 = reg_FNE_l2(out, x_detached, model, interpolation=False)

    assert math.isclose(regl2.item(), toymatrix.size(0), rel_tol=1e-3)
    assert math.isclose(regfnel2.item(), 2 * toymatrix.size(0) - 1, rel_tol=1e-3)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum", "max"])
def test_jacobian_spectral_values(toymatrix, reduction):
    ### Test reduction types on batches of images
    B, C, H, W = 5, 3, 8, 8
    toy_operators = torch.Tensor([1, 2, 3, 4, 5])[:, None, None, None]

    x_detached = torch.randn(B, C, H, W).requires_grad_()
    out = toy_operators * x_detached

    def model(x):
        return toy_operators * x

    # Nonec -> return all spectral norms
    reg_l2 = JacobianSpectralNorm(
        max_iter=100, tol=1e-4, eval_mode=False, verbose=True, reduction=reduction
    )
    reg_FNE_l2 = FNEJacobianSpectralNorm(
        max_iter=100, tol=1e-4, eval_mode=False, verbose=True, reduction=reduction
    )

    regl2 = reg_l2(out, x_detached)
    regfnel2 = reg_FNE_l2(out, x_detached, model, interpolation=False)

    if reduction == "none":
        reg_l2_target = toy_operators.squeeze()
        reg_fne_target = 2 * toy_operators.squeeze() - 1
    elif reduction == "mean":
        reg_l2_target = toy_operators.mean()
        reg_fne_target = 2 * toy_operators.sum() / toy_operators.shape[0] - 1
    elif reduction == "sum":
        reg_l2_target = toy_operators.sum()
        reg_fne_target = 2 * toy_operators.sum() - toy_operators.shape[0]
    elif reduction == "max":
        reg_l2_target = toy_operators.max()
        reg_fne_target = 2 * toy_operators.max() - 1

    assert torch.allclose(regl2, reg_l2_target, rtol=1e-3)
    assert torch.allclose(regfnel2, reg_fne_target, rtol=1e-3)


def choose_loss(loss_name, rng=None, device="cpu"):
    loss = []
    if loss_name == "mcei":
        loss.append(dinv.loss.MCLoss())
        loss.append(dinv.loss.EILoss(dinv.transform.Shift()))
    elif loss_name == "mcei-scale":
        loss.append(dinv.loss.MCLoss())
        loss.append(dinv.loss.EILoss(dinv.transform.Scale(rng=rng)))
    elif loss_name == "mcei-homography":
        pytest.importorskip(
            "kornia",
            reason="This test requires kornia. It should be "
            "installed with `pip install kornia`",
        )
        loss.append(dinv.loss.MCLoss())
        loss.append(dinv.loss.EILoss(dinv.transform.Homography(device=device)))
    elif loss_name == "splittv":
        loss.append(dinv.loss.SplittingLoss(split_ratio=0.25))
        loss.append(dinv.loss.TVLoss())
    elif loss_name == "score":
        loss.append(dinv.loss.ScoreLoss(dinv.physics.GaussianNoise(0.1), 100))
    elif loss_name in ("sup", "sup_log_train_batch"):
        loss.append(dinv.loss.SupLoss())
    elif loss_name == "r2r":
        loss.append(dinv.loss.R2RLoss(noise_model=dinv.physics.GaussianNoise(0.1)))
    elif loss_name == "vortex":
        loss.append(
            dinv.loss.AugmentConsistencyLoss(
                T_i=dinv.transform.RandomNoise(), T_e=dinv.transform.Shift()
            )
        )
    else:
        raise Exception("The loss doesnt exist")

    return loss


def choose_sure(noise_type):
    gain = 0.1
    sigma = 0.1
    if noise_type == "PoissonGaussian":
        loss = dinv.loss.SurePGLoss(sigma=sigma, gain=gain)
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "PoissonGaussianUnknown":
        loss = dinv.loss.SurePGLoss(sigma=sigma, gain=gain, unsure=True)
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "Gaussian":
        loss = dinv.loss.SureGaussianLoss(sigma=sigma)
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "GaussianUnknown":
        loss = dinv.loss.SureGaussianLoss(sigma=sigma, unsure=True)
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "Poisson":
        loss = dinv.loss.SurePoissonLoss(gain=gain)
        noise_model = dinv.physics.PoissonNoise(gain)
    else:
        raise Exception("The SURE loss doesnt exist")

    return loss, noise_model


@pytest.mark.parametrize("noise_type", LIST_SURE)
def test_sure(noise_type, device):
    imsize = (3, 256, 256)  # a bigger image reduces the error
    # choose backbone denoiser
    backbone = dinv.models.MedianFilter()

    # choose a reconstruction architecture
    f = dinv.models.ArtifactRemoval(backbone)

    # choose training losses
    loss, noise = choose_sure(noise_type)

    # choose noise
    torch.manual_seed(0)  # for reproducibility
    physics = dinv.physics.Denoising(noise)

    batch_size = 1
    x = torch.ones((batch_size,) + imsize, device=device)
    y = physics(x)

    print("sizes = ", x.size(), y.size())

    x_net = f(y, physics)
    mse = deepinv.metric.MSE()(x, x_net)
    sure = loss(y=y, x_net=x_net, physics=physics, model=f)

    rel_error = (sure - mse).abs() / mse
    assert rel_error < 0.9


def choose_r2r(noise_type):
    gain = 1.0
    sigma = 0.1
    l = 10.0

    if noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain)
        loss = dinv.loss.R2RLoss(alpha=0.9999)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma)
        loss = dinv.loss.R2RLoss(alpha=0.999)
    elif noise_type == "Gamma":
        noise_model = dinv.physics.GammaNoise(l)
        loss = dinv.loss.R2RLoss(alpha=0.999)
    else:
        raise Exception("The R2R loss doesnt exist")

    return loss, noise_model


@pytest.mark.parametrize("noise_type", LIST_R2R)
def test_r2r(noise_type, device):
    imsize = (3, 256, 256)  # a bigger image reduces the error
    # choose backbone denoiser
    backbone = dinv.models.MedianFilter()

    # choose a reconstruction architecture
    f = dinv.models.ArtifactRemoval(backbone)

    # choose training losses
    loss, noise = choose_r2r(noise_type)
    f = loss.adapt_model(f)

    # choose noise
    torch.manual_seed(0)  # for reproducibility
    physics = dinv.physics.Denoising(noise)

    batch_size = 1
    x = torch.ones((batch_size,) + imsize, device=device)
    y = physics(x)

    x_net = f(y, physics, update_parameters=True)
    mse = deepinv.metric.MSE()(x, x_net)
    r2r = loss(y=y, x_net=x_net, physics=physics, model=f)

    rel_error = (r2r - mse).abs() / mse
    rel_error = rel_error.item()
    print(rel_error)
    assert rel_error < 1.0


@pytest.fixture
def imsize():
    return (3, 15, 10)


@pytest.fixture
def physics(imsize, device):
    # choose a forward operator
    return dinv.physics.Inpainting(img_size=imsize, mask=0.5, device=device)


@pytest.fixture
def dataset(physics, tmp_path, imsize, device):
    # load dummy dataset
    save_dir = tmp_path / "dataset"
    dinv.datasets.generate_dataset(
        train_dataset=DummyCircles(samples=50, imsize=imsize),
        test_dataset=DummyCircles(samples=10, imsize=imsize),
        physics=physics,
        save_dir=save_dir,
        device=device,
    )

    return (
        dinv.datasets.HDF5Dataset(save_dir / "dinv_dataset0.h5", train=True),
        dinv.datasets.HDF5Dataset(save_dir / "dinv_dataset0.h5", train=False),
    )


def test_notraining(physics, tmp_path, imsize, device):
    # load dummy dataset
    save_dir = tmp_path / "dataset"

    dinv.datasets.generate_dataset(
        train_dataset=None,
        test_dataset=DummyCircles(samples=10, imsize=imsize),
        physics=physics,
        save_dir=save_dir,
        device=device,
    )

    dataset = dinv.datasets.HDF5Dataset(save_dir / "dinv_dataset0.h5", train=False)

    assert dataset[0][0].shape == imsize


@pytest.mark.parametrize("loss_name", LOSSES)
def test_losses(
    non_blocking_plots, loss_name, tmp_path, dataset, physics, imsize, device, rng
):
    # choose training losses
    loss = choose_loss(loss_name, rng, device=device)

    save_dir = tmp_path / "dataset"
    # choose backbone denoiser
    backbone = dinv.models.AutoEncoder(
        dim_input=imsize[0] * imsize[1] * imsize[2], dim_mid=64, dim_hid=16
    ).to(device)

    # choose a reconstruction architecture
    model = dinv.models.ArtifactRemoval(backbone, device=device)

    # choose optimizer and scheduler
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

    dataloader = DataLoader(dataset[0], batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset[1], batch_size=2, shuffle=False, num_workers=0)

    trainer = dinv.Trainer(
        model=model,
        train_dataloader=dataloader,
        eval_dataloader=test_dataloader,
        epochs=epochs,
        scheduler=scheduler,
        losses=loss,
        physics=physics,
        optimizer=optimizer,
        device=device,
        ckp_interval=int(epochs / 2),
        save_path=save_dir / "dinv_test",
        plot_images=True,
        verbose=False,
        log_train_batch=(loss_name == "sup_log_train_batch"),
    )

    # test the untrained model
    initial_test = trainer.test(test_dataloader=test_dataloader)

    # train the network
    trainer.train()
    final_test = trainer.test(test_dataloader=test_dataloader)

    assert final_test["PSNR"] > initial_test["PSNR"]


def test_sure_losses(device):
    f = dinv.models.ArtifactRemoval(dinv.models.MedianFilter())
    # test divergence

    x = torch.ones((1, 3, 16, 16), device=device) * 0.5
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1))
    y = physics(x)

    y1 = f(y, physics)
    tau = 1e-4

    exact = dinv.loss.sure.exact_div(y, physics, f)

    error_h = 0
    error_mc = 0

    num_it = 100

    for i in range(num_it):
        h = dinv.loss.sure.hutch_div(y, physics, f)
        mc = dinv.loss.sure.mc_div(y1, y, f, physics, tau)

        error_h += torch.abs(h - exact)
        error_mc += torch.abs(mc - exact)

    error_mc /= num_it
    error_h /= num_it

    assert error_h < 5e-2
    assert error_mc < 5e-2


@pytest.mark.parametrize(
    "loss_name",
    [
        "splitting",
        "weighted-splitting",
        "robust-splitting",
        "n2n",
        "splitting_eval_split_input",
        "splitting_eval_split_input_output",
    ],
)
def test_measplit(device, loss_name, rng):
    # Larger, even imsize to reduce effect of randomness
    imsize = (2, 64, 64)

    if loss_name == "n2n":
        physics = dinv.physics.Denoising()
        physics.noise_model = dinv.physics.GaussianNoise(sigma=0.1)
        backbone = dinv.models.MedianFilter()
    elif "splitting" in loss_name:
        physics = dinv.physics.Inpainting(imsize, mask=0.6, device=device, rng=rng)
        if loss_name == "robust-splitting":
            physics.noise_model = dinv.physics.GaussianNoise(0.1)
        backbone = DummyModel()

    f = dinv.models.ArtifactRemoval(backbone)

    batch_size = 1
    x = torch.ones((batch_size,) + imsize, device=device)
    y = physics(x)

    # Dummy metric to get both outputs before metric
    test_metric = lambda x, y: torch.stack([x, y])

    if loss_name == "n2n":
        loss = dinv.loss.Neighbor2Neighbor()
    elif loss_name == "splitting":
        loss = dinv.loss.SplittingLoss(
            split_ratio=0.7, metric=test_metric, eval_split_input=False
        )
    elif loss_name == "splitting_eval_split_input":
        eval_n_samples = 3
        loss = dinv.loss.SplittingLoss(
            split_ratio=0.7,
            metric=test_metric,
            eval_split_input=True,
            eval_n_samples=eval_n_samples,
        )
    elif loss_name == "splitting_eval_split_input_output":
        eval_n_samples = 1
        loss = dinv.loss.SplittingLoss(
            split_ratio=0.7,
            metric=test_metric,
            eval_split_input=True,
            eval_split_output=True,
            eval_n_samples=eval_n_samples,
        )
    elif loss_name == "weighted-splitting":
        gen = dinv.physics.generator.MultiplicativeSplittingMaskGenerator(
            imsize,
            dinv.physics.generator.BernoulliSplittingMaskGenerator(
                imsize, 0.5, device=device, rng=rng
            ),
            device=device,
        )
        loss = dinv.loss.mri.WeightedSplittingLoss(
            mask_generator=gen, physics_generator=physics.gen
        )
    elif loss_name == "robust-splitting":
        gen = dinv.physics.generator.MultiplicativeSplittingMaskGenerator(
            imsize,
            dinv.physics.generator.BernoulliSplittingMaskGenerator(
                imsize, 0.5, device=device, rng=rng
            ),
            device=device,
        )
        loss = dinv.loss.mri.RobustSplittingLoss(
            mask_generator=gen,
            physics_generator=physics.gen,
            noise_model=physics.noise_model,
        )
    else:
        raise ValueError("Loss name invalid.")

    f = loss.adapt_model(f)

    x_net = f(y, physics, update_parameters=True)
    l = loss(x_net=x_net, y=y, physics=physics, model=f)

    # Training recon + loss
    if loss_name in ("n2n", "weighted-splitting", "robust-splitting"):
        assert l > 0
    elif "splitting" in loss_name:
        y1 = x_net
        y2_hat, y2 = l.clamp(0, 1)  # remove normalisation
        # Splitting mask 1 has more samples than mask 2
        assert y2.mean() < y1.mean() < y.mean()
        # Union of splitting masks is original mask
        assert torch.all(y1 + y2 == y)
        # Splitting mask 1 and 2 are disjoint
        assert torch.all(y2_hat == 0)
    else:
        raise ValueError("Incorrect loss name.")

    f.eval()
    x_net = f(y, physics, update_parameters=True)
    y1_eval = x_net

    # Eval recon
    if loss_name == "splitting":
        # No splitting performed during eval
        assert torch.all(y == y1_eval)
        print(y.mean(), y1_eval.mean())
    elif loss_name == "splitting_eval_split_input":
        # Splits during eval
        assert y1_eval.mean() < y.mean()
        # Split data averaged across n samples so contains multiple values
        assert len(y1_eval.unique()) == eval_n_samples + 1
        # Split amount averages to amount during training
        assert y1_eval.mean() == y1.mean()
    elif loss_name == "splitting_eval_split_input_output":
        # Splits output with complement mask
        assert torch.all(y1_eval == 0)
    elif loss_name in ("weighted-splitting", "n2n", "robust-splitting"):
        pass
    else:
        raise ValueError("Incorrect loss name.")

    if loss_name == "weighted-splitting":
        assert loss.weight.shape == (1, imsize[-1])  # 1D in W dim


@pytest.mark.parametrize("mode", ["test_split_y", "test_split_physics"])
@pytest.mark.parametrize("img_size", [(1, 320, 320)])
@pytest.mark.parametrize("split_ratio", [0.6, 0.9])
def test_measplit_masking(mode, img_size, split_ratio):
    acc = 2

    class DummyModel(torch.nn.Module):
        def forward(self, y, physics, *args, **kwargs):
            return y

    class DummyModel2(torch.nn.Module):
        def forward(self, y, physics, *args, **kwargs):
            return physics.mask

    if mode == "test_split_y":
        model = DummyModel()
        dummy_metric = lambda y2_hat, y2: y2 * y2.mean()
    elif mode == "test_split_physics":
        model = DummyModel2()
        dummy_metric = lambda y2_hat, y2: y2_hat * y2.mean()

    physics = dinv.physics.Inpainting(
        img_size,
        mask=dinv.physics.generator.GaussianMaskGenerator(img_size, acc).step()["mask"],
    )
    loss = dinv.loss.SplittingLoss(
        eval_split_input=False,
        mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator(
            img_size, split_ratio=split_ratio
        ),
        metric=dummy_metric,
    )
    model.train()
    model = loss.adapt_model(model)

    x = torch.ones(img_size).unsqueeze(0)
    y = physics(x)
    with torch.no_grad():
        out = model(y, physics, update_parameters=True)

    assert torch.all(out == model.mask)
    assert np.allclose(model.mask.mean().item() * acc, split_ratio, atol=1e-4)

    if mode == "test_split_y":
        y1 = out
        y2 = loss(x, y, physics, model)
        assert torch.all(y1 + y2 == y)
    elif mode == "test_split_physics":
        physics1mask = out
        y2_hat = loss(x, y, physics, model)
        assert torch.all(physics1mask + y2_hat == y)


LOSS_SCHEDULERS = ["random", "interleaved", "random_weighted"]


@pytest.mark.parametrize("scheduler_name", LOSS_SCHEDULERS)
def test_loss_scheduler(scheduler_name):
    # Skeleton model
    class TestModel:
        def __init__(self):
            self.a = 0

    # Skeleton loss function
    class TestLoss(dinv.loss.Loss):
        def __init__(self, a=1):
            super().__init__()
            self.a = a

        def forward(self, x_net, x, y, physics, model, epoch, **kwargs):
            return self.a

        def adapt_model(self, model: TestModel, **kwargs):
            model.a += self.a
            return model

    rng = torch.Generator().manual_seed(0)

    if scheduler_name == "random":
        l = RandomLossScheduler(TestLoss(1), TestLoss(2), generator=rng)
    elif scheduler_name == "interleaved":
        l = InterleavedLossScheduler(TestLoss(1), TestLoss(2))
    elif scheduler_name == "random_weighted":
        l = RandomLossScheduler(
            [TestLoss(0), TestLoss(2)], TestLoss(1), generator=rng, weightings=[4, 1]
        )

    # Loss scheduler adapts all inside losses
    model = TestModel()
    l.adapt_model(model)
    assert model.a == 3

    # Scheduler calls all losses eventually
    loss_total = 0
    loss_log = []
    for _ in range(20):
        loss = l(None, None, None, None, None, None)
        loss_total += loss
        loss_log += [loss]
    assert loss_total > 20

    if scheduler_name == "random_weighted":
        assert loss_log.count(2) > loss_log.count(1)


def test_stacked_loss(device, imsize):
    # choose a reconstruction architecture
    backbone = dinv.models.MedianFilter()
    f = dinv.models.ArtifactRemoval(backbone)

    # choose training losses
    loss = dinv.loss.StackedPhysicsLoss(
        [dinv.loss.MCLoss(), dinv.loss.MCLoss(), dinv.loss.MCLoss()]
    )

    # choose noise
    noise = dinv.physics.GaussianNoise(0.1)
    physics = dinv.physics.StackedLinearPhysics(
        [
            dinv.physics.Denoising(noise),
            dinv.physics.Denoising(noise),
            dinv.physics.Denoising(noise),
        ]
    )

    # create a dummy image
    x = torch.ones((1,) + imsize, device=device)

    # apply the forward operator
    y = physics(x)

    # apply the denoiser
    x_net = f(y, physics)

    # calculate the loss
    loss_value = loss(x=x, y=y, x_net=x_net, physics=physics, model=f)

    assert loss_value > 0
