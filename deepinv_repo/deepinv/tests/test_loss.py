import pytest

import math
import torch

import deepinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from torch.utils.data import DataLoader
import deepinv as dinv
from deepinv.loss.regularisers import JacobianSpectralNorm, FNEJacobianSpectralNorm
from deepinv.loss.scheduler import RandomLossScheduler, InterleavedLossScheduler

LOSSES = ["sup", "sup_log_train_batch", "mcei", "mcei-scale", "mcei-homography", "r2r"]

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
    reg_l2 = JacobianSpectralNorm(max_iter=100, tol=1e-3, eval_mode=False, verbose=True)
    reg_FNE_l2 = FNEJacobianSpectralNorm(
        max_iter=100, tol=1e-3, eval_mode=False, verbose=True
    )

    # Setup our toy example; here y = A@x
    x_detached = torch.randn_like(toymatrix).requires_grad_()
    out = toymatrix @ x_detached

    def model(x):
        return toymatrix @ x

    regl2 = reg_l2(out, x_detached)
    regfnel2 = reg_FNE_l2(out, x_detached, model, interpolation=False)

    assert math.isclose(regl2.item(), toymatrix.size(0), rel_tol=1e-3)
    assert math.isclose(regfnel2.item(), 2 * toymatrix.size(0) - 1, rel_tol=1e-3)


def choose_loss(loss_name, rng=None):
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
        loss.append(dinv.loss.EILoss(dinv.transform.Homography()))
    elif loss_name == "splittv":
        loss.append(dinv.loss.SplittingLoss(split_ratio=0.25))
        loss.append(dinv.loss.TVLoss())
    elif loss_name == "score":
        loss.append(dinv.loss.ScoreLoss(dinv.physics.GaussianNoise(0.1), 100))
    elif loss_name in ("sup", "sup_log_train_batch"):
        loss.append(dinv.loss.SupLoss())
    elif loss_name == "r2r":
        loss.append(dinv.loss.R2RLoss())
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
        loss = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.9999)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma)
        loss = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.999)
    elif noise_type == "Gamma":
        noise_model = dinv.physics.GammaNoise(l)
        loss = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.999)
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
    return dinv.physics.Inpainting(tensor_size=imsize, mask=0.5, device=device)


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
def test_losses(loss_name, tmp_path, dataset, physics, imsize, device, rng):
    # choose training losses
    loss = choose_loss(loss_name, rng)

    save_dir = tmp_path / "dataset"
    # choose backbone denoiser
    backbone = dinv.models.AutoEncoder(
        dim_input=imsize[0] * imsize[1] * imsize[2], dim_mid=128, dim_hid=32
    ).to(device)

    # choose a reconstruction architecture
    model = dinv.models.ArtifactRemoval(backbone)

    # choose optimizer and scheduler
    epochs = 50
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
        plot_images=False,
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


def test_measplit(device):
    sigma = 0.1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)

    # choose a reconstruction architecture
    backbone = dinv.models.MedianFilter()
    f = dinv.models.ArtifactRemoval(backbone)
    batch_size = 1
    imsize = (3, 32, 32)

    # for split_ratio in np.linspace(0.7, 0.99, 10):
    x = torch.ones((batch_size,) + imsize, device=device)
    y = physics(x)

    # choose training losses
    loss = dinv.loss.Neighbor2Neighbor()
    n2n_loss = loss(y=y, physics=physics, model=f)

    loss = dinv.loss.SplittingLoss(split_ratio=0.5)
    f = loss.adapt_model(f)
    x_net = f(y, physics, update_parameters=True)
    split_loss = loss(x_net=x_net, y=y, physics=physics, model=f)
    f.eval()
    x_net2 = f(y, physics)

    assert split_loss > 0 and n2n_loss > 0


LOSS_SCHEDULERS = ["random", "interleaved"]


@pytest.mark.parametrize("scheduler_name", LOSS_SCHEDULERS)
def test_loss_scheduler(scheduler_name):
    # Skeleton loss function
    class TestLoss(dinv.loss.Loss):
        def __init__(self, a=1):
            super().__init__()
            self.adapted = False
            self.a = a

        def forward(self, x_net, x, y, physics, model, epoch, **kwargs):
            return self.a

        def adapt_model(self, model, **kwargs):
            self.adapted = True

    rng = torch.Generator().manual_seed(0)

    if scheduler_name == "random":
        l = RandomLossScheduler(TestLoss(1), TestLoss(2), generator=rng)
    elif scheduler_name == "interleaved":
        l = InterleavedLossScheduler(TestLoss(1), TestLoss(2))

    # Loss scheduler adapts all inside losses
    l.adapt_model(None)
    assert l.losses[0].adapted == True

    # Scheduler calls both losses eventually
    loss_total = 0
    for _ in range(20):
        loss_total += l(None, None, None, None, None, None)
    assert loss_total > 20


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
