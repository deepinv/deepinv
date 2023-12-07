import sys
import pytest
import torch

import deepinv as dinv


MODEL_LIST_1_CHANNEL = [
    "autoencoder",
    "drunet",
    "dncnn",
    "median",
    "tgv",
    "waveletprior",
    "waveletdict",
]
MODEL_LIST = MODEL_LIST_1_CHANNEL + [
    "bm3d",
    "gsdrunet",
    "scunet",
    "swinir",
    "tv",
    "unet",
    "waveletdict_hard",
    "waveletdict_topk",
]


def choose_denoiser(name, imsize):
    if name.startswith("waveletdict") or name == "waveletprior":
        pytest.importorskip(
            "pytorch_wavelets",
            reason="This test requires pytorch_wavelets. It should be "
            "installed with `pip install "
            "git+https://github.com/fbcotter/pytorch_wavelets.git`",
        )
    if name == "bm3d":
        pytest.importorskip(
            "bm3d",
            reason="This test requires bm3d. It should be "
            "installed with `pip install bm3d`",
        )
    if name in ("swinir", "scunet"):
        pytest.importorskip(
            "timm",
            reason="This test requires timm. It should be "
            "installed with `pip install timm`",
        )

    if name == "unet":
        out = dinv.models.UNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "drunet":
        out = dinv.models.DRUNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "scunet":
        out = dinv.models.SCUNet(in_nc=imsize[0])
    elif name == "gsdrunet":
        out = dinv.models.GSDRUNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "bm3d":
        out = dinv.models.BM3D()
    elif name == "dncnn":
        out = dinv.models.DnCNN(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "waveletprior":
        out = dinv.models.WaveletPrior()
    elif name == "waveletdict":
        out = dinv.models.WaveletDict()
    elif name == "waveletdict_hard":
        out = dinv.models.WaveletDict(non_linearity="hard")
    elif name == "waveletdict_topk":
        out = dinv.models.WaveletDict(non_linearity="topk")
    elif name == "tgv":
        out = dinv.models.TGV(n_it_max=10)
    elif name == "tv":
        out = dinv.models.TV(n_it_max=10)
    elif name == "median":
        out = dinv.models.MedianFilter()
    elif name == "autoencoder":
        out = dinv.models.AutoEncoder(dim_input=imsize[0] * imsize[1] * imsize[2])
    elif name == "swinir":
        out = dinv.models.SwinIR(in_chans=imsize[0])
    else:
        raise Exception("Unknown denoiser")

    return out


@pytest.mark.parametrize("denoiser", MODEL_LIST)
def test_denoiser(imsize, device, denoiser):
    model = choose_denoiser(denoiser, imsize).to(device)

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize, device=device).unsqueeze(0)
    y = physics(x)
    x_hat = model(y, sigma)

    assert x_hat.shape == x.shape


def test_equivariant(imsize, device):
    # 1. Check that the equivariance module is compatible with a denoiser
    model = dinv.models.DRUNet(in_channels=imsize[0], out_channels=imsize[0])

    model = dinv.models.EquivariantDenoiser(
        model, transform="rotoflips", random=True
    ).to(device)

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize, device=device).unsqueeze(0)
    y = physics(x)
    x_hat = model(y, sigma)

    assert x_hat.shape == x.shape

    # 2. Check that the equivariance module yields the identity when the denoiser is the identity
    class DummyIdentity(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, sigma):
            return x

    model_id = DummyIdentity()

    list_transforms = ["rotations", "flips", "rotoflips"]

    for transform in list_transforms:
        for random in [True, False]:
            model = dinv.models.EquivariantDenoiser(
                model_id, transform=transform, random=random
            ).to(device)

            x = torch.ones(imsize, device=device).unsqueeze(0)
            y = physics(x)
            y_hat = model(y, sigma)

            assert torch.allclose(y, y_hat)


@pytest.mark.parametrize("denoiser", MODEL_LIST_1_CHANNEL)
def test_denoiser_1_channel(imsize_1_channel, device, denoiser):
    model = choose_denoiser(denoiser, imsize_1_channel).to(device)

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize_1_channel, device=device).unsqueeze(0)
    y = physics(x)

    x_hat = model(y, sigma)

    assert x_hat.shape == x.shape


def test_drunet_inputs(imsize_1_channel, device):
    f = dinv.models.DRUNet(
        in_channels=imsize_1_channel[0], out_channels=imsize_1_channel[0], device=device
    )

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize_1_channel, device=device).unsqueeze(0)
    y = physics(x)

    # Case 1: sigma is a float
    x_hat = f(y, sigma)
    assert x_hat.shape == x.shape

    # Case 2: sigma is a torch tensor with batch dimension
    batch_size = 3
    x = torch.ones((batch_size, 1, 31, 37), device=device)
    y = physics(x)
    sigma_tensor = torch.tensor([sigma] * batch_size).to(device)
    x_hat = f(y, sigma_tensor)
    assert x_hat.shape == x.shape

    # Case 3: image has shape mulitple of 8
    x = torch.ones((3, 1, 32, 40), device=device)
    y = physics(x)
    x_hat = f(y, sigma_tensor)
    assert x_hat.shape == x.shape

    # Case 4: sigma is a tensor with no dimension
    sigma_tensor = torch.tensor(sigma).to(device)
    x_hat = f(y, sigma_tensor)
    assert x_hat.shape == x.shape


def test_diffunetmodel(imsize, device):
    # This model is a bit different from others as not strictly a denoiser as such.
    # The Ho et al. diffusion model only works for color, square image with powers of two in w, h.
    # Smallest size accepted so far is (3, 32, 32), but probably not meaningful at that size since trained at 256x256.

    from deepinv.models import DiffUNet

    model = DiffUNet().to(device)

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones((3, 32, 32), device=device).unsqueeze(
        0
    )  # Testing the smallest size possible
    y = physics(x)

    timestep = torch.tensor([1]).to(
        device
    )  # We pick a random timestep, goal is to check that model inference is ok.
    x_hat = model(y, timestep)
    x_hat = x_hat[:, :3, ...]

    assert x_hat.shape == x.shape

    # Now we check that the denoise_forward method works
    x_hat = model(y, sigma)
    assert x_hat.shape == x.shape

    with pytest.raises(Exception):
        # The following should raise an exception because type_t is not in ['noise_level', 'timestep'].
        x_hat = model(y, sigma, type_t="wrong_type")


def test_PDNet(imsize_1_channel, device):
    # Tests the PDNet algorithm - this is an unfolded algorithm so it is tested on its own here.
    from deepinv.optim.optimizers import CPIteration, fStep, gStep
    from deepinv.optim import Prior, DataFidelity
    from deepinv.models import PDNet_PrimalBlock, PDNet_DualBlock
    from deepinv.unfolded import unfolded_builder

    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize_1_channel, device=device).unsqueeze(0)
    y = physics(x)

    class PDNetIteration(CPIteration):
        r"""Single iteration of learned primal dual.
        We only redefine the fStep and gStep classes.
        The forward method is inherited from the CPIteration class.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.g_step = gStepPDNet(**kwargs)
            self.f_step = fStepPDNet(**kwargs)

    class fStepPDNet(fStep):
        r"""
        Dual update of the PDNet algorithm.
        We write it as a proximal operator of the data fidelity term.
        This proximal mapping is to be replaced by a trainable model.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, x, w, cur_data_fidelity, y, *args):
            r"""
            :param torch.Tensor x: Current first variable :math:`u`.
            :param torch.Tensor w: Current second variable :math:`A z`.
            :param deepinv.optim.data_fidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data fidelity term.
            :param torch.Tensor y: Input data.
            """
            return cur_data_fidelity.prox(x, w, y)

    class gStepPDNet(gStep):
        r"""
        Primal update of the PDNet algorithm.
        We write it as a proximal operator of the prior term.
        This proximal mapping is to be replaced by a trainable model.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def forward(self, x, w, cur_prior, *args):
            r"""
            :param torch.Tensor x: Current first variable :math:`x`.
            :param torch.Tensor w: Current second variable :math:`A^\top u`.
            :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
            """
            return cur_prior.prox(x, w)

    class PDNetPrior(Prior):
        def __init__(self, model, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model

        def prox(self, x, w):
            return self.model(x, w[:, 0:1, :, :])

    class PDNetDataFid(DataFidelity):
        def __init__(self, model, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model

        def prox(self, x, w, y):
            return self.model(x, w[:, 1:2, :, :], y)

    # Unrolled optimization algorithm parameters
    max_iter = 5  # number of unfolded layers

    # Set up the data fidelity term. Each layer has its own data fidelity module.
    data_fidelity = [
        PDNetDataFid(model=PDNet_DualBlock().to(device)) for i in range(max_iter)
    ]

    # Set up the trainable prior. Each layer has its own prior module.
    prior = [PDNetPrior(model=PDNet_PrimalBlock().to(device)) for i in range(max_iter)]

    n_primal = 5  # extend the primal space
    n_dual = 5  # extend the dual space

    def custom_init(y, physics):
        x0 = physics.A_dagger(y).repeat(1, n_primal, 1, 1)
        u0 = torch.zeros_like(y).repeat(1, n_dual, 1, 1)
        return {"est": (x0, x0, u0)}

    def custom_output(X):
        return X["est"][0][:, 1, :, :].unsqueeze(1)

    # Define the unfolded trainable model.
    model = unfolded_builder(
        iteration=PDNetIteration(),
        params_algo={"K": physics.A, "K_adjoint": physics.A_adjoint, "beta": 1.0},
        data_fidelity=data_fidelity,
        prior=prior,
        max_iter=max_iter,
        custom_init=custom_init,
        get_output=custom_output,
    )

    x_hat = model(y, physics)

    assert x_hat.shape == x.shape


@pytest.mark.parametrize(
    "denoiser, dep",
    [
        ("BM3D", "bm3d"),
        ("SCUNet", "timm"),
        ("SwinIR", "timm"),
        ("WaveletPrior", "pytorch_wavelets"),
        ("WaveletDict", "pytorch_wavelets"),
    ],
)
def test_optional_dependencies(denoiser, dep):
    # Skip the test if the optional dependency is installed
    if dep in sys.modules:
        pytest.skip(f"Optional dependency {dep} is installed.")

    klass = getattr(dinv.models, denoiser)
    with pytest.raises(ImportError, match=f"pip install .*{dep}"):
        klass()


# def test_dip(imsize, device): TODO: fix this test
#     torch.manual_seed(0)
#     channels = 64
#     physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.2))
#     f = dinv.models.DeepImagePrior(
#         generator=dinv.models.ConvDecoder(imsize, layers=3, channels=channels).to(
#             device
#         ),
#         input_size=(channels, imsize[1], imsize[2]),
#         iterations=30,
#     )
#     x = torch.ones(imsize, device=device).unsqueeze(0)
#     y = physics(x)
#     mse_in = (y - x).pow(2).mean()
#     x_net = f(y, physics)
#     mse_out = (x_net - x).pow(2).mean()
#
#     assert mse_out < mse_in
