import pytest

import numpy as np
import math
import torch

import deepinv
from deepinv.models import Denoiser
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from torch.utils.data import DataLoader
import deepinv as dinv
from deepinv.loss.regularisers import JacobianSpectralNorm, FNEJacobianSpectralNorm

list_losses = ["sup", "mcei"]
list_sure = ["Gaussian", "Poisson", "PoissonGaussian"]


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def toymatrix():
    w = 50
    A = torch.diag(torch.Tensor(range(1, w + 1)))
    return A


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


def choose_loss(loss_name):
    loss = []
    if loss_name == "mcei":
        loss.append(dinv.loss.MCLoss())
        loss.append(dinv.loss.EILoss(dinv.transform.Shift()))
    elif loss_name == "splittv":
        loss.append(dinv.loss.SplittingLoss(regular_mask=True, split_ratio=0.25))
        loss.append(dinv.loss.TVLoss())
    elif loss_name == "score":
        loss.append(dinv.loss.ScoreLoss(1.0))
    elif loss_name == "sup":
        loss.append(dinv.loss.SupLoss())
    else:
        raise Exception("The loss doesnt exist")

    return loss


def choose_sure(noise_type):
    gain = 0.1
    sigma = 0.1
    if noise_type == "PoissonGaussian":
        loss = dinv.loss.SurePGLoss(sigma=sigma, gain=gain)
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "Gaussian":
        loss = dinv.loss.SureGaussianLoss(sigma=sigma)
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "Poisson":
        loss = dinv.loss.SurePoissonLoss(gain=gain)
        noise_model = dinv.physics.PoissonNoise(gain)
    else:
        raise Exception("The SURE loss doesnt exist")

    return loss, noise_model


@pytest.mark.parametrize("noise_type", list_sure)
def test_sure(noise_type):
    imsize = (3, 256, 256)  # a bigger image reduces the error
    # choose backbone denoiser
    backbone = dinv.models.MedianFilter()

    # choose a reconstruction architecture
    f = dinv.models.ArtifactRemoval(backbone)

    # choose training losses
    loss, noise = choose_sure(noise_type)

    # choose noise
    physics = dinv.physics.Denoising(noise=noise)

    batch_size = 1
    x = torch.ones((batch_size,) + imsize, device=dinv.device)
    y = physics(x)

    x_net = f(y, physics)
    mse = deepinv.metric.mse()(x, x_net)
    sure = loss(y, x_net, physics, f)

    rel_error = (sure - mse).abs() / mse
    assert rel_error < 0.9


@pytest.fixture
def imsize():
    return (3, 15, 10)


@pytest.fixture
def physics(imsize):
    # choose a forward operator
    return dinv.physics.Inpainting(tensor_size=imsize, mask=0.5, device=dinv.device)


@pytest.fixture
def dataset(physics, tmp_path, imsize):
    # load dummy dataset
    save_dir = tmp_path / "dataset"
    dinv.datasets.generate_dataset(
        train_dataset=DummyCircles(samples=50, imsize=imsize),
        test_dataset=DummyCircles(samples=10, imsize=imsize),
        physics=physics,
        save_dir=save_dir,
        device=dinv.device,
    )

    return dinv.datasets.HDF5Dataset(
        save_dir / "dinv_dataset0.h5", train=True
    ), dinv.datasets.HDF5Dataset(save_dir / "dinv_dataset0.h5", train=False)


@pytest.mark.parametrize("loss_name", list_losses)
def test_losses(loss_name, tmp_path, dataset, physics, imsize):
    # choose training losses
    loss = choose_loss(loss_name)

    save_dir = tmp_path / "dataset"
    # choose backbone denoiser
    backbone = dinv.models.AutoEncoder(
        dim_input=imsize[0] * imsize[1] * imsize[2], dim_mid=128, dim_hid=32
    ).to(dinv.device)

    # choose a reconstruction architecture
    model = dinv.models.ArtifactRemoval(backbone)

    # choose optimizer and scheduler
    epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

    dataloader = DataLoader(dataset[0], batch_size=2, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset[1], batch_size=2, shuffle=False, num_workers=0)

    # test the untrained model
    initial_psnr = dinv.test(
        model=model,
        test_dataloader=test_dataloader,
        physics=physics,
        plot_images=False,
        device=dinv.device,
    )

    # train the network
    model = dinv.train(
        model=model,
        train_dataloader=dataloader,
        epochs=epochs,
        scheduler=scheduler,
        losses=loss,
        physics=physics,
        optimizer=optimizer,
        device=dinv.device,
        ckp_interval=int(epochs / 2),
        save_path=save_dir / "dinv_test",
        plot_images=False,
        verbose=False,
    )

    final_psnr = dinv.test(
        model=model,
        test_dataloader=test_dataloader,
        physics=physics,
        plot_images=False,
        device=dinv.device,
    )

    assert final_psnr[0] > initial_psnr[0]


def test_sure_losses():
    model_spec = {
        "name": "waveletprior",
        "args": {"wv": "db8", "level": 3, "device": dinv.device},
    }
    f = dinv.models.ArtifactRemoval(Denoiser(model_spec))
    # test divergence

    x = torch.ones((1, 3, 16, 16), device=dinv.device) * 0.5
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

    # print(f"error_h: {error_h}")
    # print(f"error_mc: {error_mc}")
    assert error_h < 5e-2
    assert error_mc < 5e-2


def test_measplit():
    sigma = 0.1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)

    # choose a reconstruction architecture
    backbone = dinv.models.MedianFilter()
    f = dinv.models.ArtifactRemoval(backbone)
    batch_size = 1
    imsize = (3, 128, 128)

    for split_ratio in np.linspace(0.7, 0.99, 10):
        x = torch.ones((batch_size,) + imsize, device=dinv.device)
        y = physics(x)

        # choose training losses
        loss = dinv.loss.SplittingLoss(split_ratio=split_ratio, regular_mask=True)
        x_net = f(y, physics)
        mse = dinv.metric.mse()(physics.A(x), physics.A(x_net))
        split_loss = loss(y, physics, f)

        rel_error = (split_loss - mse).abs() / mse
