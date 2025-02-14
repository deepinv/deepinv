from math import ceil
import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.unfolded import unfolded_builder
from deepinv.physics import Inpainting, GaussianNoise, Blur, Pansharpen
from deepinv.physics.generator import (
    BernoulliSplittingMaskGenerator,
    SigmaGenerator,
    DiffractionBlurGenerator,
)


@pytest.mark.parametrize("physics_name", ["inpainting", "pansharpen"])
def test_generate_dataset(tmp_path, imsize, device, physics_name):
    N = 10
    max_N = 10
    train_dataset = DummyCircles(samples=N, imsize=imsize)
    test_dataset = DummyCircles(samples=N, imsize=imsize)

    if physics_name == "inpainting":
        physics = Inpainting(mask=0.5, tensor_size=imsize, device=device)
        y_shape = imsize
    elif physics_name == "pansharpen":  # proxy for StackedPhysics
        physics = Pansharpen(img_size=imsize, factor=2, device=device)
        C, H, W = imsize
        y_shape = [(C, ceil(H / 2), ceil(W / 2)), (1, H, W)]
    else:
        raise ValueError(f"Unknown physics {physics_name}")

    dinv.datasets.generate_dataset(
        train_dataset,
        physics,
        tmp_path,
        test_dataset=test_dataset,
        device=device,
        dataset_filename="dinv_dataset",
        train_datapoints=max_N,
    )

    dataset = dinv.datasets.HDF5Dataset(path=f"{tmp_path}/dinv_dataset0.h5", train=True)

    assert len(dataset) == min(max_N, N)

    x, y = dataset[0]
    assert x.shape == imsize
    assert y.shape == y_shape


@pytest.mark.parametrize(
    "physics_combo",
    ["single_physics_no_gen", "single_physics_with_gen", "multi_physics_no_gen"],
)
@pytest.mark.parametrize("bsize", [1, 4])
@pytest.mark.parametrize(
    "phys_gen",
    [
        "bernoulli_mask",
        "sigma",
        "diffraction",
    ],
)
def test_generate_dataset_physics_generator(
    physics_combo, tmp_path, imsize, bsize, phys_gen
):
    N = 10

    class DummyDataset(Dataset):
        def __getitem__(self, i):
            return torch.ones(imsize)

        def __len__(self):
            return N

    x_dataset = DummyDataset()

    if physics_combo == "single_physics_no_gen":
        physics = Inpainting(tensor_size=imsize)
        physics_generator = None
    elif physics_combo == "single_physics_with_gen":
        if phys_gen == "bernoulli_mask":
            physics = Inpainting(tensor_size=imsize)
            physics_generator = BernoulliSplittingMaskGenerator(imsize, 0.6)
        elif phys_gen == "sigma":
            physics = GaussianNoise()
            physics_generator = SigmaGenerator()
        elif phys_gen == "diffraction":
            physics = Blur()
            physics_generator = DiffractionBlurGenerator((3, 3), num_channels=3)
    elif physics_combo == "multi_physics_no_gen":
        physics = [Inpainting(imsize, mask=0.1), Inpainting(imsize, mask=0.9)]
        physics_generator = None

    _ = dinv.datasets.generate_dataset(
        x_dataset,
        physics,
        physics_generator=physics_generator,
        save_physics_generator_params=True,
        test_dataset=x_dataset,
        save_dir=tmp_path,
        train_datapoints=N,
        test_datapoints=N,
        batch_size=bsize,
    )

    train_dataset = dinv.datasets.HDF5Dataset(
        path=f"{tmp_path}/dinv_dataset0.h5", train=True
    )
    test_dataset = dinv.datasets.HDF5Dataset(
        path=f"{tmp_path}/dinv_dataset0.h5", train=False
    )

    if physics_combo == "single_physics_no_gen":
        # test physics remains constant
        x0, y0 = train_dataset[0]
        x1, y1 = train_dataset[1]
        x9, y9 = train_dataset[9]
        assert torch.all(y0 == y1)
        assert torch.all(y0 == y9)

        x0t, y0t = test_dataset[0]
        x1t, y1t = test_dataset[1]
        x9t, y9t = test_dataset[9]
        assert torch.all(y0t == y1t)
        assert torch.all(y0t == y9t)
    elif physics_combo == "single_physics_with_gen":
        # test physics random generated
        x0, y0 = train_dataset[0]
        x1, y1 = train_dataset[1]

        x0t, y0t = test_dataset[0]
        x1t, y1t = test_dataset[1]

        if phys_gen != "diffraction":
            assert not torch.all(y0 == y1)
            assert not torch.all(y0t == y1t)
            assert not torch.all(y0 == y0t)

        # test load physics generator params
        d = dinv.datasets.HDF5Dataset(
            path=f"{tmp_path}/dinv_dataset0.h5",
            train=True,
            load_physics_generator_params=True,
        )
        x, y, params = d[0]
        if phys_gen == "bernoulli_mask":
            assert torch.all(y == params["mask"])
        elif phys_gen == "sigma":
            assert params["sigma"].ndim == 0
        elif phys_gen == "diffraction":
            _ = params["filter"], params["coeff"], params["pupil"]

    elif physics_combo == "multi_physics_no_gen":
        # test each dataset has different physics
        train_dataset1 = dinv.datasets.HDF5Dataset(
            path=f"{tmp_path}/dinv_dataset1.h5", train=True
        )
        x0, y0 = train_dataset[0]
        x1, y1 = train_dataset1[0]
        assert y0.mean() < 0.5
        assert y1.mean() > 0.5
        assert len(train_dataset) == len(train_dataset1) == N // 2

    # test dataloader
    b = 3
    batch = next(iter(DataLoader(train_dataset, batch_size=b)))
    x, y, params = (*batch, {}) if len(batch) == 2 else batch

    for t in [x, y] + list(params.values()):
        assert t.shape[0] == b


# optim_algos = [
#     "PGD",
#     "HQS",
#     "DRS",
#     "ADMM",
#     "CP",
# ]

optim_algos = ["PGD"]


@pytest.mark.parametrize("name_algo", optim_algos)
def test_optim_algo(name_algo, imsize, device):
    # This test uses WaveletDenoiser, which requires pytorch_wavelets
    # TODO: we could use a dummy trainable denoiser with a linear layer instead
    pytest.importorskip("ptwt")

    # pths
    BASE_DIR = Path(".")
    CKPT_DIR = BASE_DIR / "ckpts"

    # Select the data fidelity term
    data_fidelity = L2()

    # Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
    # If the prior is initialized with a list of length max_iter,
    # then a distinct weight is trained for each PGD iteration.
    # For fixed trained model prior across iterations, initialize with a single model.
    max_iter = 30 if torch.cuda.is_available() else 3  # Number of unrolled iterations
    level = 3
    prior = [
        PnP(denoiser=dinv.models.WaveletDenoiser(wv="db8", level=level, device=device))
        for i in range(max_iter)
    ]

    # Unrolled optimization algorithm parameters
    lamb = [
        1.0
    ] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
    stepsize = [
        1.0
    ] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.

    sigma_denoiser = [
        0.01
        * torch.ones(
            level,
        )
    ] * max_iter
    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "lambda": lamb,
    }

    # define which parameters from 'params_algo' are trainable
    trainable_params = ["g_param", "stepsize"]

    # Define the unfolded trainable model.

    # Because the CP algorithm uses more than 2 variables, we need to define a custom initialization.
    if name_algo == "CP":

        def custom_init(y, physics):
            x_init = physics.A_adjoint(y)
            u_init = y
            return {"est": (x_init, x_init, u_init)}

        params_algo["sigma"] = 1.0
    else:
        custom_init = None

    model_unfolded = unfolded_builder(
        name_algo,
        params_algo=params_algo,
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        custom_init=custom_init,
    )

    for idx, (name, param) in enumerate(model_unfolded.named_parameters()):
        assert param.requires_grad
        assert (trainable_params[0] in name) or (trainable_params[1] in name)

    N = 10
    train_dataset = DummyCircles(samples=N, imsize=imsize)
    test_dataset = DummyCircles(samples=N, imsize=imsize)

    physics = dinv.physics.Inpainting(mask=0.5, tensor_size=imsize, device=device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=2, num_workers=1, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, num_workers=1, shuffle=False
    )

    epochs = 1
    losses = [dinv.loss.SupLoss(metric=dinv.metric.MSE())]
    optimizer = torch.optim.Adam(model_unfolded.parameters(), lr=1e-3, weight_decay=0.0)

    trainer = dinv.Trainer(
        model=model_unfolded,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        epochs=epochs,
        losses=losses,
        physics=physics,
        optimizer=optimizer,
        device=device,
        save_path=str(CKPT_DIR),
        verbose=True,
        online_measurements=True,
    )
    trainer.train()
    trainer.test(test_dataloader)


def test_epll_parameter_estimation(imsize, dummy_dataset, device):
    from deepinv.datasets import PatchDataset

    torch.manual_seed(0)

    imgs = dummy_dataset.x
    patch_dataset = PatchDataset(imgs)
    patch_dataloader = torch.utils.data.DataLoader(
        patch_dataset, batch_size=2, shuffle=True, drop_last=False
    )
    epll = dinv.optim.EPLL(channels=imsize[0], pretrained=None, n_components=3)
    epll.GMM.fit(patch_dataloader, max_iters=10)

    assert not torch.any(torch.isnan(epll.GMM.mu))
    assert not torch.any(torch.isnan(epll.GMM.get_cov()))
    assert not torch.any(torch.isnan(epll.GMM.get_weights()))
