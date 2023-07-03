import pytest
import deepinv as dinv
import torch


@pytest.fixture
def device():
    return dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


@pytest.fixture
def imsize():
    h = 37
    w = 31
    c = 3
    return c, h, w


@pytest.fixture
def imsize_1_channel():
    h = 37
    w = 31
    c = 1
    return c, h, w


model_list = [
    "unet",
    "drunet",
    "scunet",
    "dncnn",
    "waveletprior",
    "waveletdict",
    "waveletdict_hard",
    "waveletdict_topk",
    "tgv",
    "median",
    "autoencoder",
    "gsdrunet",
    "swinir"
]

try:  # install of BM3D may fail on some architectures (arm64)
    from dinv.models.bm3d import bm3d

    model_list.append("bm3d")
except ImportError:
    print("Could not find bm3d; not testing bm3d.")


def choose_denoiser(name, imsize):
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
    elif name == "median":
        out = dinv.models.MedianFilter()
    elif name == "autoencoder":
        out = dinv.models.AutoEncoder(dim_input=imsize[0] * imsize[1] * imsize[2])
    elif name == "swinir":
        out = dinv.models.SwinIR(img_size=(imsize[1], imsize[2]), in_chans=imsize[0])
    else:
        raise Exception("Unknown denoiser")

    return out


@pytest.mark.parametrize("denoiser", model_list)
def test_denoiser(imsize, device, denoiser):
    if denoiser in ("waveletprior", "waveletdict"):
        try:
            import pytorch_wavelets
        except ImportError:
            pytest.xfail(
                "This test requires pytorch_wavelets. "
                "It should be installed with `pip install"
                "git+https://github.com/fbcotter/pytorch_wavelets.git`"
            )
    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize, device=device).unsqueeze(0)
    y = physics(x)

    f = choose_denoiser(denoiser, imsize).to(device)

    x_hat = f(y, sigma)

    assert x_hat.shape == x.shape


model_list_1_channel = [
    "drunet",
    "dncnn",
    "waveletprior",
    "waveletdict",
    "tgv",
    "median",
    "autoencoder",
]


@pytest.mark.parametrize("denoiser", model_list_1_channel)
def test_denoiser_1_channel(imsize_1_channel, device, denoiser):
    if denoiser in ("waveletprior", "waveletdict"):
        try:
            import pytorch_wavelets
        except ImportError:
            pytest.xfail(
                "This test requires pytorch_wavelets. "
                "It should be installed with `pip install"
                "git+https://github.com/fbcotter/pytorch_wavelets.git`"
            )
    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize_1_channel, device=device).unsqueeze(0)
    y = physics(x)

    f = choose_denoiser(denoiser, imsize_1_channel).to(device)

    x_hat = f(y, sigma)

    assert x_hat.shape == x.shape


def test_PDNet(imsize, device):
    # Tests the PDNet algorithm - this is an unfolded algorithm so it is tested on its own here.
    from deepinv.optim.optimizers import CPIteration, fStep, gStep
    from deepinv.optim import Prior, DataFidelity
    from deepinv.models.PDNet import PrimalBlock, DualBlock
    from deepinv.unfolded import unfolded_builder

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize, device=device).unsqueeze(0)
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
            return self.model(x, w)

    class PDNetDataFid(DataFidelity):
        def __init__(self, model, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model

        def prox(self, x, w, y):
            return self.model(x, w, y)

    # Unrolled optimization algorithm parameters
    max_iter = 5  # number of unfolded layers

    # Set up the data fidelity term. Each layer has its own data fidelity module.
    data_fidelity = [
        PDNetDataFid(model=DualBlock(in_channels=9).to(device)) for i in range(max_iter)
    ]

    # Set up the trainable prior. Each layer has its own prior module.
    prior = [
        PDNetPrior(model=PrimalBlock(in_channels=6).to(device)) for i in range(max_iter)
    ]

    def custom_init(y, physics):
        z0 = physics.A_adjoint(y)
        x0 = physics.A_adjoint(y)
        u0 = y
        return {"est": (x0, z0, u0)}

    # Define the unfolded trainable model.
    model = unfolded_builder(
        iteration=PDNetIteration(),
        params_algo={"K": physics.A, "K_adjoint": physics.A_adjoint, "beta": 1.0},
        data_fidelity=data_fidelity,
        prior=prior,
        max_iter=max_iter,
        custom_init=custom_init,
    )

    x_hat = model(y, physics)

    assert x_hat.shape == x.shape


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
