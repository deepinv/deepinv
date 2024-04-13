import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.unfolded import unfolded_builder
from deepinv.training import train
from deepinv.training import test as feature_test


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
        train_datapoints=max_N,
    )

    dataset = dinv.datasets.HDF5Dataset(path=f"{tmp_path}/dinv_dataset0.h5", train=True)

    assert len(dataset) == min(max_N, N)

    x, y = dataset[0]
    assert x.shape == imsize


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
    losses = [dinv.loss.SupLoss(metric=dinv.metric.mse())]
    optimizer = torch.optim.Adam(model_unfolded.parameters(), lr=1e-3, weight_decay=0.0)

    trained_unfolded_model = train(
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
        wandb_vis=False,
        online_measurements=True,
    )

    results = feature_test(
        model=trained_unfolded_model,
        test_dataloader=test_dataloader,
        physics=physics,
        device=device,
        plot_images=False,
        verbose=True,
        wandb_vis=False,
    )


def test_epll_parameter_estimation(imsize, dummy_dataset, device):
    from deepinv.datasets import PatchDataset

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
