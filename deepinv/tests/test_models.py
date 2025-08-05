import sys
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import deepinv as dinv
from deepinv.loss import PSNR

from dummy import DummyCircles, DummyModel

from deepinv.tests.test_physics import find_operator

MODEL_LIST_1_CHANNEL = [
    "autoencoder",
    "drunet",
    "dncnn",
    "median",
    "tgv",
    "waveletdenoiser",
    "waveletdict",
    "epll",
    "restormer",
    "ncsnpp",
    "adinv.modelsunet",
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
    "dsccp",
]

REST_MODEL_LIST = [
    "ram",
    "modl",
    "varnet",
    "pannet",
]

LINEAR_OPERATORS = [
    "inpainting",
    "demosaicing",
    "denoising",
    "fftdeblur",
    "deblur_valid",
    "super_resolution_circular",
    "MRI",
    "pansharpen_circular",
]  # this is a reduced list of linear operators for testing restoration models.

CHANNELS = [
    1,
    2,
    3,
]


@pytest.fixture
def rng(device):
    return torch.Generator(device).manual_seed(0)


def choose_denoiser(name, imsize):
    if name.startswith("waveletdict") or name == "waveletdenoiser":
        pytest.importorskip(
            "ptwt",
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
    elif name == "waveletdenoiser":
        out = dinv.models.WaveletDenoiser()
    elif name == "waveletdict":
        out = dinv.models.WaveletDictDenoiser()
    elif name == "waveletdict_hard":
        out = dinv.models.WaveletDictDenoiser(non_linearity="hard")
    elif name == "waveletdict_topk":
        out = dinv.models.WaveletDictDenoiser(non_linearity="topk")
    elif name == "tgv":
        out = dinv.models.TGVDenoiser(n_it_max=10)
    elif name == "tv":
        out = dinv.models.TVDenoiser(n_it_max=10)
    elif name == "median":
        out = dinv.models.MedianFilter()
    elif name == "autoencoder":
        out = dinv.models.AutoEncoder(dim_input=imsize[0] * imsize[1] * imsize[2])
    elif name == "swinir":
        out = dinv.models.SwinIR(in_chans=imsize[0])
    elif name == "epll":
        out = dinv.models.EPLLDenoiser(channels=imsize[0])
    elif name == "restormer":
        out = dinv.models.Restormer(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "ncsnpp":
        out = dinv.models.NCSNpp(
            in_channels=imsize[0],
            out_channels=imsize[0],
            img_resolution=imsize[1],
            pretrained=None,
        )
    elif name == "adinv.modelsunet":
        out = dinv.models.ADMUNet(
            in_channels=imsize[0],
            out_channels=imsize[0],
            img_resolution=imsize[1],
            pretrained=None,
        )
    elif name == "dsccp":
        out = dinv.models.DScCP()
    else:
        raise Exception("Unknown denoiser")

    return out.eval()


def choose_restoration_model(name, in_channels=3, out_channels=3, pretrained=None):
    if name == "ram":
        out = dinv.models.RAM(pretrained=pretrained)
    elif name == "modl" or name == "varnet":
        denoiser = dinv.models.DnCNN(
            in_channels, out_channels, 7, pretrained=pretrained
        )
        if name == "modl":
            out = dinv.models.MoDL(denoiser=denoiser, num_iter=3)
        else:
            out = dinv.models.VarNet(
                num_cascades=3,
                mode=name,
                denoiser=denoiser,
            )
    elif name == "pannet":
        hrms_shape = (8, 16, 16)  # manually adjust
        out = dinv.models.PanNet(hrms_shape=hrms_shape, scale_factor=4)
    else:
        raise Exception("Unknown restoration model")
    return out.eval()


def test_TVs_adjoint():
    r"""
    This tests the adjointness of the finite difference operator used in TV and TGV regularisation.
    """
    model = dinv.models.TVDenoiser(n_it_max=10)

    u = torch.randn((4, 3, 20, 20)).type(torch.DoubleTensor)
    Au = model.nabla(u)
    v = torch.randn(*Au.shape).type(Au.dtype)
    Atv = model.nabla_adjoint(v)
    e = v.flatten() @ Au.flatten() - Atv.flatten() @ u.flatten()

    assert torch.allclose(e, torch.tensor([0.0], dtype=torch.float64))

    model = dinv.models.TGVDenoiser(n_it_max=10)

    u = torch.randn((4, 3, 20, 20)).type(torch.DoubleTensor)
    Au = model.nabla(u)
    v = torch.randn(*Au.shape).type(Au.dtype)
    Atv = model.nabla_adjoint(v)
    e = v.flatten() @ Au.flatten() - Atv.flatten() @ u.flatten()

    assert torch.allclose(e, torch.tensor([0.0], dtype=torch.float64))

    u = torch.randn((2, 3, 20, 20, 2)).type(torch.DoubleTensor)
    Au = model.epsilon(u)
    v = torch.randn(*Au.shape).type(Au.dtype)
    Atv = model.epsilon_adjoint(v)
    e = v.flatten() @ Au.flatten() - Atv.flatten() @ u.flatten()

    assert torch.allclose(e, torch.tensor([0.0], dtype=torch.float64))


def test_wavelet_adjoints():
    pytest.importorskip(
        "ptwt",
        reason="This test requires ptwt. It should be "
        "installed with `pip install ptwt`",
    )

    torch.manual_seed(0)

    def gen_function(Au, wvdim=2):
        r"""
        A helper function to generate a random tensor of the same shape as a wavelet decomposition.
        """
        out = [torch.randn_like(Au[0])]

        if wvdim == 2:
            for l in range(1, len(Au)):
                out = out + [[torch.randn_like(Aul) for Aul in Au[l]]]

        elif wvdim == 3:
            for l in range(1, len(Au)):
                out = out + [{key: torch.randn_like(Au[l][key]) for key in Au[l]}]
        return out

    for dimension in ["2d", "3d"]:
        wvdim = 2 if dimension == "2d" else 3

        model = dinv.models.WaveletDenoiser(wvdim=wvdim)

        def A_f(x):
            dx = model.dwt(x)
            Ax = model.flatten_coeffs(dx)
            return dx, Ax

        def gen_rand(Au):
            v = gen_function(Au, wvdim=wvdim)
            v_flat = model.flatten_coeffs(v)
            return v, v_flat

        # Note: we only check the adjointness for square tensors as a reshape is performed in our proximal solvers
        # to handle non-square tensors.
        if wvdim == 2:
            u = torch.randn((2, 3, 20, 20)).type(torch.DoubleTensor)
        else:
            u = torch.randn((2, 3, 20, 20, 20)).type(torch.DoubleTensor)

        Au, Au_flat = A_f(u)
        v, v_flat = gen_rand(Au)
        Atv = model.iwt(v)
        e = v_flat.flatten() @ Au_flat.flatten() - Atv.flatten() @ u.flatten()

        assert torch.allclose(e, torch.tensor([0.0], dtype=torch.float64))


def test_wavelet_models_identity():
    # We check that variational models yield identity when regularization parameter is set to 0.

    pytest.importorskip(
        "ptwt",
        reason="This test requires pytorch_wavelets. It should be "
        "installed with `pip install "
        "git+https://github.com/fbcotter/pytorch_wavelets.git`",
    )

    # 1. Wavelet denoiser (single & dictionary)
    for dimension in ["2d", "3d"]:
        wvdim = 2 if dimension == "2d" else 3
        x = (
            torch.randn((4, 3, 31, 27))
            if dimension == "2d"
            else torch.randn((4, 3, 31, 27, 29))
        )
        for non_linearity in ["soft", "hard"]:
            model = dinv.models.WaveletDenoiser(
                non_linearity=non_linearity, wvdim=wvdim
            )
            ths = (
                0.0 if non_linearity in ["soft", "hard"] else 1.0
            )  # topk counts the number of coefficients to keep
            x_hat = model(x, ths)
            assert x_hat.shape == x.shape
            assert torch.allclose(
                x, x_hat, atol=1e-5
            )  # The model should be the identity

        model = dinv.models.WaveletDictDenoiser(
            list_wv=["haar", "db3", "db8"], non_linearity="soft", wvdim=wvdim
        )
        x_hat = model(x, 0.0)
        assert x_hat.shape == x.shape
        assert torch.allclose(x, x_hat, atol=1e-5)  # The model should be the identity

    # 2. Wavelet Prior
    for dimension in ["2d", "3d"]:
        wvdim = 2 if dimension == "2d" else 3
        x = (
            torch.ones((4, 3, 31, 27))
            if dimension == "2d"
            else torch.ones((4, 3, 31, 27, 29))
        )
        level = 3
        prior = dinv.optim.prior.WaveletPrior(wvdim=wvdim, p=1, level=level)
        g_nonflat = prior(x, reduce=False)
        g_flat = prior(x, reduce=True)
        assert g_nonflat.dim() > 0
        assert g_nonflat.size(1) == 3 * level if wvdim == 2 else 7 * level
        assert g_flat.dim() == 1  # Batch dimension

        assert torch.allclose(g_nonflat.abs().sum(dim=1), g_flat)


def test_TV_models_identity():
    # Next priors are checked for 2D only
    x = torch.randn((4, 3, 31, 27))

    # 3. TV
    model = dinv.models.TVDenoiser(n_it_max=10)
    x_hat = model(x, 0.0)
    assert x_hat.shape == x.shape
    assert torch.allclose(x, x_hat, atol=1e-5)  # The model should be the identity

    # 4. TGV
    model = dinv.models.TGVDenoiser(n_it_max=10)
    x_hat = model(
        x, 1e-6
    )  # There is some numerical instability when the regularization parameter is 0
    assert x_hat.shape == x.shape
    assert torch.allclose(x, x_hat, atol=1e-5)  # The model should be the identity


@pytest.mark.parametrize("denoiser", MODEL_LIST)
def test_denoiser_color(imsize, device, denoiser):
    # NCSNpp and Adinv.modelsUnet only support imsize that divisible by 8
    if denoiser in ["ncsnpp", "adinv.modelsunet"]:
        imsize = (imsize[0], (imsize[1] // 8) * 8, (imsize[2] // 8) * 8)

    model = choose_denoiser(denoiser, imsize).to(device)
    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize, device=device).unsqueeze(0)
    y = physics(x)
    x_hat = model(y, sigma)

    assert x_hat.shape == x.shape


@pytest.mark.parametrize("denoiser", MODEL_LIST)
def test_denoiser_gray(imsize_1_channel, device, denoiser):
    # except scunet and dsccp, all models support 1 channel
    if denoiser not in ["scunet", "dsccp"]:
        # NCSNpp and Adinv.modelsUnet only support imsize that divisible by 8
        if denoiser in ["ncsnpp", "adinv.modelsunet"]:
            imsize_1_channel = (
                imsize_1_channel[0],
                (imsize_1_channel[1] // 8) * 8,
                (imsize_1_channel[2] // 8) * 8,
            )

        model = choose_denoiser(denoiser, imsize_1_channel).to(device)

        torch.manual_seed(0)
        sigma = 0.2
        physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
        x = torch.ones(imsize_1_channel, device=device).unsqueeze(0)
        y = physics(x)
        x_hat = model(y, sigma)

        assert x_hat.shape == x.shape


@pytest.mark.parametrize("batch_size", [1, 2])
def test_equivariant(imsize, device, batch_size):
    # 1. Check that the equivariance module is compatible with a denoiser
    model = dinv.models.DRUNet(in_channels=imsize[0], out_channels=imsize[0])

    model = (
        dinv.models.EquivariantDenoiser(model, random=True)  # Roto-reflections
        .to(device)
        .eval()
    )

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones((batch_size, *imsize), device=device)
    y = physics(x)
    x_hat = model(y, sigma)

    assert x_hat.shape == x.shape

    # 2. Check that the equivariance module yields the identity when the denoiser is the identity
    model_id = DummyModel()

    list_transforms = [
        dinv.transform.Rotate(
            multiples=90, positive=True, n_trans=4, constant_shape=False
        ),  # full group
        dinv.transform.Rotate(
            multiples=90, positive=True, n_trans=1, constant_shape=False
        ),  # subsampled
        dinv.transform.Reflect(dim=[-1], n_trans=2),  # full group
        dinv.transform.Reflect(dim=[-1], n_trans=2)
        * dinv.transform.Rotate(
            multiples=90, positive=True, n_trans=3, constant_shape=False
        ),  # compound group
    ]

    # constant_shape False as imsize is rectangular. For square images, ignore constant_shape.

    for transform in list_transforms:
        model = dinv.models.EquivariantDenoiser(model_id, transform=transform).to(
            device
        )

        x = torch.ones((batch_size, *imsize), device=device)
        y = physics(x)
        y_hat = model(y, sigma)

        assert torch.allclose(y, y_hat)


@pytest.mark.parametrize("denoiser", MODEL_LIST_1_CHANNEL)
def test_denoiser_1_channel(imsize_1_channel, device, denoiser):
    if denoiser in ["ncsnpp", "adinv.modelsunet"]:
        imsize_1_channel = list(imsize_1_channel)
        if imsize_1_channel[1] % 8 > 0:
            imsize_1_channel[1] = 32
        if imsize_1_channel[2] % 8 > 0:
            imsize_1_channel[2] = 32
    model = choose_denoiser(denoiser, imsize_1_channel).to(device)

    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize_1_channel, device=device).unsqueeze(0)
    y = physics(x)

    x_hat = model(y, sigma)

    assert x_hat.shape == x.shape


@pytest.mark.parametrize("denoiser", MODEL_LIST_1_CHANNEL)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_denoiser_sigma_gray(batch_size, denoiser, device):
    img_size = (1, 64, 64)
    model = choose_denoiser(denoiser, img_size).to(device)
    noiser = dinv.physics.GaussianNoise()
    x = torch.ones((batch_size,) + img_size, device=device, dtype=torch.float32)
    # Same sigma for all image in the batch
    sigma = torch.tensor(0.1)
    y = noiser(x, sigma=sigma)
    with torch.no_grad():
        x_hat = model(y, sigma)
    assert x_hat.shape == x.shape

    # Same sigma but a float
    with torch.no_grad():
        x_hat = model(y, sigma)
    assert x_hat.shape == x.shape

    # Each sigma for each image in the batch
    if batch_size > 1:
        sigma = torch.linspace(0.1, 0.3, batch_size, device=device)
        with torch.no_grad():
            y = noiser(x, sigma=sigma)
            x_hat = model(y, sigma)
        assert x_hat.shape == x.shape


@pytest.mark.parametrize("denoiser", MODEL_LIST)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_denoiser_sigma_color(batch_size, denoiser, device):
    img_size = (3, 64, 64)
    model = choose_denoiser(denoiser, img_size).to(device)
    noiser = dinv.physics.GaussianNoise()
    x = torch.ones((batch_size,) + img_size, device=device, dtype=torch.float32)
    # Same sigma for all image in the batch
    sigma = torch.tensor(0.1)
    y = noiser(x, sigma=sigma)
    with torch.no_grad():
        x_hat = model(y, sigma)
    assert x_hat.shape == x.shape

    # Same sigma but a float
    with torch.no_grad():
        x_hat = model(y, sigma)
    assert x_hat.shape == x.shape

    # Each sigma for each image in the batch
    if batch_size > 1:
        sigma = torch.linspace(0.1, 0.3, batch_size, device=device)
        with torch.no_grad():
            y = noiser(x, sigma=sigma)
            x_hat = model(y, sigma)
        assert x_hat.shape == x.shape


@pytest.mark.parametrize("level", [3, 5, 6])
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("non_linearity", ["soft", "hard", "topk"])
def test_wavelet_denoiser_ths(
    level, channels, dimension, non_linearity, batch_size, device
):
    model = dinv.models.WaveletDenoiser(
        level=level, wvdim=dimension, non_linearity=non_linearity
    ).to(device)
    img_size = (batch_size, channels, 64, 64)
    y = torch.randn(img_size, dtype=torch.float32).to(device)

    # Test with a float threshold
    ths = 0.1
    x_hat = model(y, ths)
    assert x_hat.shape == y.shape
    # Test with a tensor threshold
    ths_tensor = torch.tensor([ths] * batch_size, device=device)
    x_hat = model(y, ths_tensor)
    assert x_hat.shape == y.shape
    # Test with a tensor for each level
    ths_tensor = torch.rand((batch_size, level), device=device)
    x_hat = model(y, ths_tensor)
    assert x_hat.shape == y.shape

    # Test with a tensor for batch, level and wavelet dimension
    numel = 3 if dimension == 2 else 7
    ths_tensor = torch.rand((batch_size, level, numel), device=device)
    x_hat = model(y, ths_tensor)
    assert x_hat.shape == y.shape


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("dimension", [2, 3])
def test_wavelet_decomposition(channels, dimension, batch_size, device):
    model = dinv.models.WaveletDenoiser(level=1, wvdim=dimension).to(device)
    img_size = (
        (batch_size, channels, 64, 64)
        if dimension == 2
        else (batch_size, channels, 8, 64, 64)
    )
    # Test the wavelet decomposition and reconstruction
    x = torch.randn(img_size, dtype=torch.float32).to(device)
    # 1 decomposition
    out = model.dwt(x)
    x_hat = model.iwt(out)

    # For some reason the precision is more than 100x lower on GPU.
    tol = 1e-3 if torch.device(device).type == "cuda" else 1e-5

    # NOTE: Tensors are broadcasted in torch.allclose so
    # they might pass the test even if they have different shapes. For this
    # reason we also check the shapes.
    assert x_hat.shape == x.shape
    assert torch.allclose(x, x_hat, rtol=tol, atol=tol)

    # 2 decomposition
    cA1, cD1 = model.dwt(x)
    cA2, cD2 = model.dwt(cA1)

    x_hat = model.iwt((cA2, cD2, cD1))
    assert torch.allclose(x, x_hat, rtol=tol, atol=tol)


def test_drunet_inputs(imsize_1_channel, device):
    f = dinv.models.DRUNet(
        in_channels=imsize_1_channel[0], out_channels=imsize_1_channel[0], device=device
    ).eval()

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
        ("WaveletDenoiser", "ptwt"),
        ("WaveletDictDenoiser", "ptwt"),
    ],
)
def test_optional_dependencies(denoiser, dep):
    # Skip the test if the optional dependency is installed
    if dep in sys.modules:
        pytest.skip(f"Optional dependency {dep} is installed.")

    klass = getattr(dinv.models, denoiser)
    with pytest.raises(ImportError, match=f"pip install .*{dep}"):
        klass()


def test_icnn(device, rng):
    from deepinv.models import ICNN

    model = ICNN(in_channels=3, device=device)
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1, rng=rng))
    x = torch.ones((1, 3, 128, 128), device=device)
    y = physics(x)
    potential = model(y)
    grad = model.grad(y)
    assert grad.shape == x.shape


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


def test_time_agnostic_net():
    backbone_net = dinv.models.UNet(scales=2)
    net = dinv.models.TimeAgnosticNet(backbone_net)
    y = torch.rand(1, 1, 2, 4, 4)  # B,C,T,H,W
    x_net = net(y, None)
    assert x_net.shape == y.shape


@pytest.mark.parametrize("varnet_type", ("varnet", "e2e-varnet", "modl"))
def test_varnet(varnet_type, device):

    def dummy_dataset(imsize):
        return DummyCircles(samples=1, imsize=imsize)

    x = dummy_dataset((2, 8, 8))[0].unsqueeze(0).to(device)
    physics = dinv.physics.MRI(
        mask=dinv.physics.generator.GaussianMaskGenerator(
            x.shape[1:], acceleration=2, device=device
        ).step()["mask"],
        device=device,
    )
    y = physics(x)

    class DummyMRIDataset(Dataset):
        def __getitem__(self, i):
            return x[0], y[0]

        def __len__(self):
            return 1

    denoiser = dinv.models.DnCNN(2, 2, 7, pretrained=None, device=device)
    if varnet_type == "modl":
        model = dinv.models.MoDL(denoiser=denoiser, num_iter=3).to(device)
    else:
        model = dinv.models.VarNet(
            num_cascades=3,
            mode=varnet_type,
            denoiser=denoiser,
        ).to(device)

    model = dinv.Trainer(
        model=model,
        physics=physics,
        optimizer=torch.optim.Adam(model.parameters()),
        train_dataloader=DataLoader(DummyMRIDataset()),
        epochs=50,
        save_path=None,
        plot_images=False,
        compare_no_learning=True,
        device=device,
        optimizer_step_multi_dataset=True,
    ).train()

    x_hat = model(y, physics)
    x_init = physics.A_adjoint(y)

    assert x_hat.shape == x_init.shape

    psnr = dinv.metric.PSNR()
    assert psnr(x_init, x) < psnr(x_hat, x)


LIST_IMAGE_WHSIZE = [(32, 37), (25, 129)]


@pytest.mark.parametrize("pretrained", [True, None])
@pytest.mark.parametrize("whsize", LIST_IMAGE_WHSIZE)
@pytest.mark.parametrize("model_name", REST_MODEL_LIST)
@pytest.mark.parametrize("physics_name", LINEAR_OPERATORS + [None])
@pytest.mark.parametrize("channels", CHANNELS)
def test_restoration_models(
    device, pretrained, model_name, physics_name, channels, rng, whsize
):

    if channels == 1 and physics_name in ["demosaicing", "MRI"]:
        pytest.skip(f"Skipping {model_name} with {physics_name} for 1 channel input.")

    if channels == 2 and physics_name == "demosaicing":
        pytest.skip(f"Skipping {model_name} with {physics_name} for 2 channel input.")

    if channels == 3 and physics_name == "MRI":
        pytest.skip(f"Skipping {model_name} with {physics_name} for 3 channel input.")

    if model_name == "varnet" or model_name == "modl" or model_name == "pannet":
        pytest.skip(f"Skipping {model_name} with {physics_name}. TODO: fix.")

    if model_name != "ram" and physics_name is None:
        pytest.skip(f"Skipping {model_name} with {physics_name}.")

    model = choose_restoration_model(
        model_name, in_channels=channels, out_channels=channels, pretrained=pretrained
    ).to(device)
    torch.manual_seed(0)

    imsize = (channels, whsize[0], whsize[1])

    if physics_name is not None:
        physics, imsize, _, dtype = find_operator(physics_name, device, imsize=imsize)
    else:
        physics = None

    if hasattr(physics, "noise_model"):
        if hasattr(physics.noise_model, "sigma"):
            physics.noise_model.sigma = torch.tensor(
                [max(physics.noise_model.sigma, 0.01)]
            )
        else:
            physics.noise_model = dinv.physics.GaussianNoise(0.01, rng=rng)
    else:
        if physics is not None:
            physics.noise_model = dinv.physics.GaussianNoise(0.01, rng=rng)

    x = DummyCircles(imsize=imsize, samples=1)[0].unsqueeze(0)

    if physics is not None:
        y = physics(x)
    else:
        y = torch.randn_like(x)

    if physics_name is not None:
        with torch.no_grad():
            x_hat = model(y, physics)
    else:
        if model_name == "ram":
            # ram model should output an error if no sigma and gain is provided
            with pytest.raises(ValueError):
                x_hat = model(y, physics)
            x_hat = model(y, sigma=0.01, gain=1.0)

    assert x_hat.shape == x.shape

    psnr_fn = PSNR(max_pixel=1)

    if (
        not (physics_name == "super_resolution_circular" and channels == 2)
        and model_name == "ram"
        and pretrained == True
        and physics is not None
    ):  # suboptimal performance in this case
        psnr_in = psnr_fn(physics.A_dagger(y), x)
        psnr_out = psnr_fn(x_hat, x)
        assert torch.all(psnr_out > psnr_in)
    else:
        pytest.skip(f"Skipping PSNR test for {model_name} with {physics_name}.")


def test_pannet():
    hrms_shape = (8, 16, 16)  # C,H,W

    physics = dinv.physics.Pansharpen(hrms_shape, factor=4)
    model = dinv.models.PanNet(hrms_shape=hrms_shape, scale_factor=4)

    x = torch.rand((1,) + hrms_shape)  # B,C,H,W
    y = physics(x)  # (MS, PAN)

    x_net = model(y, physics)

    assert x_net.shape == x.shape


@pytest.mark.parametrize("device", [torch.device("cpu")])
@pytest.mark.parametrize("image_size", [32, 64])
@pytest.mark.parametrize("n_channels", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("precond", [True, False])
@pytest.mark.parametrize("use_fp16", [True, False])
def test_ncsnpp_net(device, image_size, n_channels, batch_size, precond, use_fp16):
    # Load the pretrained model
    model = dinv.models.NCSNpp(
        img_resolution=image_size,
        in_channels=n_channels,
        out_channels=n_channels,
        pretrained=None,
    )
    if precond:
        model = dinv.models.EDMPrecond(model, use_fp16=use_fp16).to(device)
    else:
        model = model.to(device)
    x = torch.rand(batch_size, n_channels, image_size, image_size, device=device)

    y = model(x, 0.01)
    # Check the output tensor shape
    assert y.shape == x.shape

    y = model(x, torch.tensor(0.01))
    # Check the output tensor shape
    assert y.shape == x.shape

    y = model(x, torch.tensor([0.01]))
    # Check the output tensor shape
    assert y.shape == x.shape


@pytest.mark.parametrize("n_channels", [3])
def test_dsccp_net(device, n_channels):
    # Load the pretrained model

    image_size = (n_channels, 37, 28)
    model = dinv.models.DScCP().to(device)
    x = torch.rand(image_size, device=device).unsqueeze(0)

    y = model(x, 0.01)
    # Check the output tensor shape
    assert y.shape == x.shape

    y = model(x, torch.tensor(0.01))
    # Check the output tensor shape
    assert y.shape == x.shape

    # batch of sigma
    x = x.expand(4, -1, -1, -1)
    y = model(x, torch.linspace(0.01, 0.1, 4, device=device))
    assert y.shape == x.shape


def test_denoiser_perf(device):
    # Load 2 example images
    x1 = dinv.utils.load_example(
        "butterfly.png",
        img_size=64,
        resize_mode="resize",
    ).to(device)

    x2 = dinv.utils.load_example(
        "celeba_example.jpg",
        img_size=64,
        resize_mode="resize",
    ).to(device)

    x = torch.cat([x1, x2, x1], dim=0)
    # three different noise levels
    sigma = torch.tensor([0.05, 0.1, 0.2]).to(device)
    rng = torch.Generator(device=device).manual_seed(123)
    y = x + sigma.view(-1, 1, 1, 1) * torch.randn(x.shape, generator=rng, device=device)

    psnr_fn = PSNR(max_pixel=1)

    # Only test the trained denoisers and the correspinding expected performance
    learned_denoisers = [
        (dinv.models.DnCNN(pretrained="download").to(device), (0.1, 0.03, 0.001)),
        (dinv.models.DRUNet(pretrained="download").to(device), (7.0, 10.5, 11.0)),
        (dinv.models.GSDRUNet(pretrained="download").to(device), (6.5, 10.5, 10.5)),
        (dinv.models.SCUNet(pretrained="download").to(device), (3.5, 9.5, 8.5)),
        (dinv.models.SwinIR(pretrained="download").to(device), (7.5, 3.4, 1.0)),
        (dinv.models.DiffUNet(pretrained="download").to(device), (6.5, 10.5, 10.0)),
        (
            dinv.models.Restormer(
                in_channels=3, out_channels=3, pretrained="denoising"
            ).to(device),
            (7.0, 9.5, 10.0),
        ),
        (dinv.models.NCSNpp(pretrained="download").to(device), (7.0, 11.5, 10.5)),
        (dinv.models.ADMUNet(pretrained="download").to(device), (7.0, 11.5, 11.0)),
        (dinv.models.DScCP(pretrained="download").to(device), (4.5, 9.0, 3.0)),
    ]

    for denoiser, expected_perf in learned_denoisers:
        kwargs = {}

        # It's a conditional denoiser
        if isinstance(denoiser, dinv.models.ADMUNet):
            class_labels = torch.eye(1000, device=device)[
                0 : x.size(0)
            ]  # Take random class labels
            kwargs["class_labels"] = class_labels
        with torch.no_grad():
            x_hat = denoiser(y, sigma, **kwargs)
        assert torch.all(
            psnr_fn(x_hat, x) >= psnr_fn(y, x) + torch.tensor(expected_perf).to(device)
        )

    # Classical denoisers
    classical_denoisers = [
        (dinv.models.BM3D(), (2.75, 7.5, 6.5)),
        (dinv.models.MedianFilter(kernel_size=5), (-9, 4.25, 2.5)),
        (dinv.models.TVDenoiser(), (1.5, 6.0, 4.0)),
        (dinv.models.TGVDenoiser(), (1.75, 6, 5.0)),
        (
            dinv.models.WaveletDenoiser(non_linearity="hard", device=device),
            (0.05, 2.5, 2.5),
        ),
        (
            dinv.models.WaveletDictDenoiser(non_linearity="hard", device=device),
            (0.2, 4.0, 2.5),
        ),
        (
            dinv.models.EPLLDenoiser(pretrained="download", channels=3, device=device),
            (2.5, 6, 7.5),
        ),
    ]
    for denoiser, expected_perf in classical_denoisers:
        kwargs = {}

        # Some denoisers require specific sigma scaling
        if isinstance(denoiser, dinv.models.TVDenoiser):
            _sigma = sigma.clone() * torch.tensor([0.5, 1.0, 1.0]).to(device)
        elif isinstance(denoiser, dinv.models.TGVDenoiser):
            _sigma = sigma.clone() * torch.tensor([4.0, 5.0, 5.0]).to(device)
        elif isinstance(
            denoiser, (dinv.models.WaveletDenoiser, dinv.models.WaveletDictDenoiser)
        ):
            _sigma = sigma.clone() * torch.tensor([1.05, 2.0, 3.0]).to(device)
        else:
            _sigma = sigma
        with torch.no_grad():
            x_hat = denoiser(y, _sigma, **kwargs)
        assert torch.all(
            psnr_fn(x_hat, x) >= psnr_fn(y, x) + torch.tensor(expected_perf).to(device)
        )
