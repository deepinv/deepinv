import pytest
import torch
import math
import deepinv as dinv
from contextlib import nullcontext

# Noise model which has a `rng` attribute
NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "LogPoisson",
    "SaltPepper",
    "RicianNoise",
    "Laplace",
    "Gamma",
    "FisherTippett",
]


DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

DTYPES = [torch.float32, torch.float64]


def choose_noise(noise_type, rng):
    gain = 0.1
    sigma = 0.1
    mu = 0.2
    N0 = 1024.0
    p, s = 0.025, 0.025
    if noise_type == "PoissonGaussian":
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain, rng=rng)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma, rng=rng)
    elif noise_type == "UniformGaussian":
        noise_model = dinv.physics.UniformGaussianNoise(rng=rng)
    elif noise_type == "Uniform":
        noise_model = dinv.physics.UniformNoise(a=gain, rng=rng)
    elif noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain, rng=rng)
    elif noise_type == "LogPoisson":
        noise_model = dinv.physics.LogPoissonNoise(N0, mu, rng=rng)
    elif noise_type == "SaltPepper":
        noise_model = dinv.physics.SaltPepperNoise(p=p, s=s, rng=rng)
    elif noise_type == "RicianNoise":
        noise_model = dinv.physics.RicianNoise(sigma=sigma, rng=rng)
    elif noise_type == "Laplace":
        noise_model = dinv.physics.LaplaceNoise(b=sigma, rng=rng)
    elif noise_type == "Gamma":
        noise_model = dinv.physics.GammaNoise(l=gain, rng=rng)
    elif noise_type == "FisherTippett":
        noise_model = dinv.physics.FisherTippettNoise(l=gain, rng=rng)
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concatenation(device, rng, dtype):
    imsize = (1, 3, 7, 16)
    for name_1 in NOISES:
        for name_2 in NOISES:
            noise_model_1 = choose_noise(name_1, rng=rng)
            noise_model_2 = choose_noise(name_2, rng=rng)
            noise_model = noise_model_1 * noise_model_2
            x = torch.rand(imsize, device=device, dtype=dtype)
            y = noise_model(x)
            assert y.shape == torch.Size(imsize)


@pytest.mark.parametrize("name", NOISES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rng(name, device, rng, dtype):
    imsize = (1, 3, 7, 16)
    x = torch.rand(imsize, device=device, dtype=dtype)
    if name == "Gamma":
        # Gamma measurements must be strictly positive, including at RNG endpoints.
        x = x.clamp_min(torch.finfo(dtype).eps)
    noise_model = choose_noise(name, rng=rng)
    y_1 = noise_model(x, seed=0)
    y_2 = noise_model(x, seed=1)
    y_3 = noise_model(x, seed=0)
    assert torch.allclose(y_1, y_3)
    assert not torch.allclose(y_1, y_2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_laplace_noise_moments(device, dtype, rng):
    imsize = (1, 3, 7, 16)
    b = 0.1
    x = torch.zeros(imsize, device=device, dtype=dtype)
    noise_model = dinv.physics.LaplaceNoise(b=b, rng=rng)
    b = noise_model.b.item()
    y = noise_model(x, seed=0)
    noise = y - x

    true_mean = x.mean().item()
    true_var = 2 * (b**2)

    empirical_mean = noise.mean().item()
    empirical_var = noise.var().item()

    assert math.isclose(empirical_mean, true_mean, abs_tol=1e-1)
    assert math.isclose(empirical_var, true_var, abs_tol=1e-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gaussian_noise_arithmetics(device, rng, dtype):

    sigma_0 = 0.01
    sigma_1 = 0.05
    sigma_2 = 0.2

    multiplication_value = 0.5

    noise_model_0 = dinv.physics.GaussianNoise(sigma_0, rng=rng)
    noise_model_1 = dinv.physics.GaussianNoise(sigma_1, rng=rng)
    noise_model_2 = dinv.physics.GaussianNoise(sigma_2, rng=rng)

    # addition
    noise_model = noise_model_0 + noise_model_1 + noise_model_2
    assert math.isclose(
        noise_model.sigma.item(),
        (sigma_0**2 + sigma_1**2 + sigma_2**2) ** (0.5),
        abs_tol=1e-5,
    )

    # multiplication

    # right
    noise_model = noise_model_0 * multiplication_value
    assert math.isclose(
        noise_model.sigma.item(), (sigma_0 * multiplication_value), abs_tol=1e-5
    )

    # left
    noise_model = multiplication_value * noise_model_0
    assert math.isclose(
        noise_model.sigma.item(), (sigma_0 * multiplication_value), abs_tol=1e-5
    )

    # factorisation
    noise_model = (noise_model_0 + noise_model_1) * multiplication_value
    assert math.isclose(
        noise_model.sigma.item(),
        ((sigma_0**2 + sigma_1**2) ** (0.5)) * multiplication_value,
        abs_tol=1e-5,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_poisson_noise_params(device, rng, dtype):

    gain = 0.1

    def get_poisson_model(name, normalize=False, clip_positive=False):
        if name == "Poisson":
            return dinv.physics.PoissonNoise(
                gain, normalize=normalize, clip_positive=clip_positive, rng=rng
            )
        elif name == "PoissonGaussian":
            return dinv.physics.PoissonGaussianNoise(
                gain, sigma=0.0, rng=rng, clip_positive=clip_positive
            )  # we check that poissongaussian noise behaves like poisson noise when sigma=0.0

    for name in ["Poisson", "PoissonGaussian"]:

        noise_model_1 = get_poisson_model(name, normalize=False)
        noise_model_2 = get_poisson_model(name, normalize=True)
        noise_model_3 = get_poisson_model(name, normalize=True, clip_positive=True)

        imsize = (1, 3, 7, 16)

        # Positive entries
        x = torch.rand(imsize, device=device, dtype=dtype)
        y_1 = noise_model_1(x, seed=0)
        y_2 = noise_model_2(x, seed=0)

        assert y_1.shape == torch.Size(imsize)

        # Check that the Poisson noise model is normalized in the case of Poisson noise
        if name == "Poisson":
            assert torch.allclose(gain * y_1, y_2, atol=1e-6)

        # handling negative values
        # check that an entry with negative value raises an error
        x = torch.randn(imsize, device=device, dtype=dtype)
        with pytest.raises(Exception) as e_info:
            y2_bis = noise_model_2(x, seed=0)  # will fail because of negative values

        y_3 = noise_model_3(x, seed=0)

        # check that no negative values are present in y_3
        assert torch.all(y_3 >= 0)


# NOTE: This is a regression test.
@pytest.mark.parametrize("sigma_device", DEVICES)
@pytest.mark.parametrize("rng_kind", ["none", "consistent", "inconsistent"])
def test_gaussian_noise_device_inference(sigma_device, rng_kind):
    if not torch.cuda.is_available() and rng_kind == "inconsistent":
        pytest.skip("This test requires having at least one CUDA device available.")

    sigma = torch.tensor(1.0, device=sigma_device)

    if rng_kind != "none":
        if rng_kind == "consistent":
            rng_device = sigma.device
        elif rng_kind == "inconsistent":  # pragma: no cover
            if sigma_device.type == "cuda":
                rng_device = torch.device("cpu")
            elif sigma_device.type == "cpu":
                rng_device = torch.device("cuda:0")
            else:
                raise ValueError(f"Unknown device type: {sigma_device.type}")
        else:
            raise ValueError(f"Unknown rng_kind: {rng_kind}")

        rng = torch.Generator(device=rng_device)
    else:
        rng = None

    noise_model = None
    with pytest.raises(RuntimeError) if rng_kind == "inconsistent" else nullcontext():
        noise_model = dinv.physics.GaussianNoise(sigma=sigma, rng=rng)

    if noise_model is not None:
        assert noise_model.sigma.device == sigma.device
        if rng is not None:
            assert noise_model.sigma.device == rng.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["Gamma", "FisherTippett"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("l", [0.1, 0.5, 1, 8.0])
@pytest.mark.parametrize("intensity", [0.1, 1.0, 10.0])
def test_gamma_sampling_distribution(name, device, rng, dtype, l, intensity):
    stats = pytest.importorskip("scipy.stats")
    x = torch.full((30000,), intensity, device=device, dtype=dtype)
    model = getattr(dinv.physics, name + "Noise")(l=l, rng=rng)
    y = model(x if name == "Gamma" else x.log(), seed=0)
    samples = (y if name == "Gamma" else y.exp()).cpu().numpy()

    # Compare against the previous implementation using an independent seed.
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1)
        reference = torch.distributions.Gamma(l, l / x).sample().cpu().numpy()
    # A conservative significance level avoids rejecting valid random samples.
    assert stats.ks_2samp(samples, reference).pvalue > 1e-6
    assert stats.kstest(samples, "gamma", args=(l, 0, intensity / l)).pvalue > 1e-6
    shape, loc, scale = stats.gamma.fit(samples, floc=0)
    assert shape == pytest.approx(l, rel=0.05)
    assert loc == 0
    assert scale == pytest.approx(intensity / l, rel=0.05)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["Gamma", "FisherTippett"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gamma_integer_level_and_rng_state(name, device, rng, dtype):
    x = torch.ones((128,), device=device, dtype=dtype)
    model = getattr(dinv.physics, name + "Noise")(l=2, rng=rng)
    initial = model(x)
    assert not torch.equal(initial, model(x))
    model.reset_rng()
    assert torch.equal(initial, model(x))
    assert model.l.is_floating_point()
    assert torch.isfinite(model(x, l=3, seed=0)).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gamma_sampling_broadcast_and_gradients(device, rng, dtype):
    x = torch.ones((1, 4), device=device, dtype=dtype, requires_grad=True)
    l = torch.tensor([[0.5], [2.0]], device=device, dtype=dtype, requires_grad=True)
    y = dinv.physics.GammaNoise(l=l, rng=rng)(x)
    assert y.shape == (2, 4)
    assert not y.requires_grad
    assert (y > 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["Gamma", "FisherTippett"])
def test_gamma_sampling_underflow(name, device, rng):
    x = torch.full((128,), 1e-12, device=device)
    model = getattr(dinv.physics, name + "Noise")(l=0.001, rng=rng)
    y = model(x if name == "Gamma" else x.log(), seed=0)
    assert torch.isfinite(y).all()
    if name == "Gamma":
        assert (y > 0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["Gamma", "FisherTippett"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gamma_sampling_without_rng(name, device, dtype):
    x = torch.linspace(0.1, 2, 128, device=device, dtype=dtype)
    model = getattr(dinv.physics, name + "Noise")(l=2)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        y = model(x if name == "Gamma" else x.log())
        torch.manual_seed(0)
        expected = torch.distributions.Gamma(model.l, model.l / x).sample()
        if name == "FisherTippett":
            expected = expected.log()
    torch.testing.assert_close(y, expected)
