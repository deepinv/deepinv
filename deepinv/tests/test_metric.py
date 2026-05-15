import pytest
import torch
import deepinv as dinv
import deepinv.loss.metric as metric
from deepinv.utils import load_example
import math
import os
from PIL import Image
import numpy as np

FULL_REFERENCE_METRICS = [
    "MAE",
    "MSE",
    "MSE1",
    "MSE2",
    "NMSE",
    "PSNR",
    "SNR",
    "SSIM",
    "LpNorm",
    "L1L2",
    "LPIPS",
    "ERGAS",
    "SAM",
    "HaarPSI",
    "CosineSimilarity",
]
NO_REFERENCE_METRICS = [
    "BlurStrength",
    "SharpnessIndex",
    "SharpnessIndex1",
    "SharpnessIndex2",
    "NIQE",
]
FUNCTIONALS = ["cal_mse", "cal_mae", "cal_psnr", "signal_noise_ratio"]


def choose_full_reference_metric(metric_name, device, **kwargs) -> metric.Metric:
    if metric_name == "MSE":
        # Test importing from deepinv.loss.metric
        return metric.MSE(**kwargs)
    elif metric_name == "MSE1":
        # Test importing from deepinv.metric
        return dinv.metric.MSE(**kwargs)
    elif metric_name == "MSE2":
        # Test importing from deepinv.loss directly
        return dinv.loss.MSE(**kwargs)
    elif metric_name == "NMSE":
        return metric.NMSE(**kwargs)
    elif metric_name == "MAE":
        return metric.MAE(**kwargs)
    elif metric_name == "PSNR":
        return metric.PSNR(**kwargs)
    elif metric_name == "SNR":
        return metric.SNR(**kwargs)
    elif metric_name == "SSIM":
        return metric.SSIM(**kwargs)
    elif metric_name == "LpNorm":
        return metric.LpNorm(**kwargs)
    elif metric_name == "L1L2":
        return metric.L1L2(**kwargs)
    elif metric_name == "LPIPS":
        return metric.LPIPS(device=device, **kwargs)
    elif metric_name == "ERGAS":
        return metric.ERGAS(factor=4, **kwargs)
    elif metric_name == "SAM":
        return metric.SpectralAngleMapper(**kwargs)
    elif metric_name == "HaarPSI":
        kwargs.pop("norm_inputs")
        return metric.HaarPSI(norm_inputs="clip", **kwargs)
    elif metric_name == "CosineSimilarity":
        return metric.CosineSimilarity(**kwargs)
    else:
        raise ValueError("Incorrect metric name.")


def choose_no_reference_metric(metric_name, device, **kwargs) -> metric.Metric:
    if metric_name == "NIQE":
        return metric.NIQE(
            patch_size=64,
            patch_overlap=32,
            denominator=1 / 255,
            **kwargs,
            device=device,
        )
    elif metric_name == "QNR":
        return metric.QNR()
    elif metric_name == "BlurStrength":
        return metric.BlurStrength(**kwargs)
    elif metric_name == "SharpnessIndex":
        return metric.SharpnessIndex(**kwargs)
    elif metric_name == "SharpnessIndex1":
        return metric.SharpnessIndex(dequantize=False, **kwargs)
    elif metric_name == "SharpnessIndex2":
        return metric.SharpnessIndex(periodic_component=False, **kwargs)
    else:
        raise ValueError("Incorrect no-reference metric name.")


@pytest.fixture(scope="session")
def test_image():
    return load_example(
        "celeba_example.jpg",
        img_size=128,
        resize_mode="resize",
    )


@pytest.mark.parametrize("metric_name", FULL_REFERENCE_METRICS)
@pytest.mark.parametrize("train_loss", [True, False])
@pytest.mark.parametrize("norm_inputs", [None])
@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("max_pixel", [1, None])
@pytest.mark.parametrize("min_pixel", [0, None])
def test_full_reference_metrics(
    metric_name,
    train_loss,
    norm_inputs,
    rng,
    device,
    channels,
    test_image,
    max_pixel,
    min_pixel,
):
    metric_kwargs = {
        "complex_abs": channels == 2,
        "train_loss": train_loss,
        "norm_inputs": norm_inputs,
        "reduction": "mean",  # 'none' is tested below
    }
    if metric_name in ("SSIM", "PSNR"):
        metric_kwargs |= {"max_pixel": max_pixel, "min_pixel": min_pixel}
    elif max_pixel is None or min_pixel is None:
        pytest.skip("max_pixel or min_pixel set to None requires SSIM or PSNR.")

    m = choose_full_reference_metric(metric_name, device, **metric_kwargs)

    test_image = test_image.to(device)
    x = test_image.clone()

    x = x[:, :channels]

    if metric_name in ("SAM", "ERGAS") and channels < 3:
        pytest.skip("ERGAS or SAM must have multichannels.")

    x_hat = dinv.physics.GaussianNoise(sigma=0.1, rng=rng)(x)

    if metric_name == "LPIPS":
        x = x.clip(min=0.0, max=1.0)
        x_hat = x_hat.clip(min=0.0, max=1.0)
        if channels != 3:
            pytest.skip("LPIPS requires 3 channel input.")

    # Test metric worse when image worse
    # In general, metrics can be either lower or higher = better
    # However, if we set train_loss=True, all metrics become lower = better.
    if train_loss:
        assert m(x_hat, x).item() > m(x, x).item()

    # Test various args and kwargs which could be passed to metrics
    assert m(x_hat, x, None, model=None, some_other_kwarg=None) != 0
    assert m(x_net=x_hat, x=x, some_other_kwarg=None) != 0

    # Test summing metrics
    dummy_metric = metric.Metric(metric=lambda *a, **kw: 1)
    m2 = m + dummy_metric
    assert m2(x_hat, x) == m(x_hat, x) + 1

    # Test no reduce works
    B = 5
    x_hat = torch.cat([x_hat] * B)
    m = choose_full_reference_metric(
        metric_name,
        device,
        complex_abs=channels == 2,
        train_loss=train_loss,
        norm_inputs=norm_inputs,
        reduction="none",
    )
    assert len(m(x_hat, x_hat)) == B


@pytest.mark.parametrize("metric_name", NO_REFERENCE_METRICS)
@pytest.mark.parametrize("train_loss", [True, False])
@pytest.mark.parametrize("norm_inputs", [None])
@pytest.mark.parametrize("channels", [1, 2, 3])
def test_no_reference_metrics(
    metric_name,
    train_loss,
    norm_inputs,
    rng,
    device,
    channels,
    test_image,
):
    metric_kwargs = {
        "complex_abs": channels == 2,
        "train_loss": train_loss,
        "norm_inputs": norm_inputs,
        "reduction": "mean",
    }

    if (
        metric_name == "NIQE" and channels == 2
    ):  # NIQE only acts on 1- and 3-channel images.
        return

    m = choose_no_reference_metric(metric_name, device, **metric_kwargs)

    test_image = test_image.to(device)
    x = test_image.clone()

    if metric_name == "QNR":
        x_hat = x
        physics = dinv.physics.Pansharpen((3, 128, 128), device=device)
        y = physics(x)
        assert 0 < m(x_net=x_hat, y=y, physics=physics).item() < 1
        return

    x = x[:, :channels]

    # test noise
    x_hat = dinv.physics.GaussianNoise(sigma=0.1, rng=rng)(x)
    if metric_name not in ("BlurStrength",):  # BlurStrength not robust to noise.
        if not m.lower_better and not train_loss:
            assert m(x_hat).item() < m(x).item()
        else:
            assert m(x_hat).item() > m(x).item()

    # test blur
    x_hat = dinv.physics.BlurFFT(
        filter=dinv.physics.functional.gaussian_blur(sigma=(3, 3)),
        img_size=x.shape[1:],
        device=device,
    )(x)
    if not m.lower_better and not train_loss:
        assert m(x_hat).item() < m(x).item()
    else:
        assert m(x_hat).item() > m(x).item()

    # Test various args and kwargs which could be passed to metrics
    assert m(x_hat, None, model=None, some_other_kwarg=None) != 0
    assert m(x_net=x_hat, some_other_kwarg=None) != 0

    # Test summing metrics
    dummy_metric = metric.Metric(metric=lambda *a, **kw: 1)
    m2 = m + dummy_metric
    assert m2(x_hat) == m(x_hat) + 1

    # Test no reduce works
    B = 5
    x_hat = torch.cat([x_hat] * B)
    m = choose_no_reference_metric(
        metric_name,
        device,
        complex_abs=channels == 2,
        train_loss=train_loss,
        norm_inputs=norm_inputs,
        reduction="none",
    )
    assert len(m(x_hat)) == B


@pytest.mark.parametrize("functional_name", FUNCTIONALS)
def test_functional(functional_name, imsize_2_channel, device, rng):
    x = torch.rand((1, *imsize_2_channel), device=device, generator=rng)
    x_net = torch.rand((1, *imsize_2_channel), device=device, generator=rng)

    if functional_name == "cal_mse":
        # Note the torch losses average the batch so they are unsuitable for being used as metrics
        # However in these tests we set batch size to 1 so it doesn't matter
        assert metric.cal_mse(x_net, x) == torch.nn.MSELoss()(x_net, x)
        assert metric.MSE()(x_net, x) == torch.nn.MSELoss()(x_net, x)
        assert torch.allclose(
            metric.cal_mse(x_net, x), metric.LpNorm(p=2)(x_net, x) / x.numel()
        )

    elif functional_name == "cal_mae":
        assert metric.cal_mae(x_net, x) == torch.nn.L1Loss()(x_net, x)
        assert metric.MAE()(x_net, x) == torch.nn.L1Loss()(x_net, x)
        assert torch.allclose(
            metric.cal_mae(x_net, x), metric.LpNorm(p=1)(x_net, x) / x.numel()
        )

    elif functional_name == "cal_psnr":
        pytest.importorskip(
            "torchmetrics",
            reason="This test requires torchmetrics. It should be "
            "installed with `pip install torchmetrics`",
        )
        from torchmetrics.image import PeakSignalNoiseRatio

        torch_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        assert torch.allclose(metric.cal_psnr(x_net, x), torch_psnr(x_net, x))

    elif functional_name == "signal_noise_ratio":
        pytest.importorskip(
            "torchmetrics",
            reason="This test requires torchmetrics. It should be "
            "installed with `pip install torchmetrics`",
        )
        from torchmetrics.functional.audio import (
            signal_noise_ratio as signal_noise_ratio_ref,
        )

        # torchmetrics SNR only supports 1D inputs so we flatten the inputs
        x_net, x = x_net.flatten(1, -1), x.flatten(1, -1)

        assert torch.allclose(
            metric.signal_noise_ratio(x_net, x),
            signal_noise_ratio_ref(x_net, x, zero_mean=False),
        )


def test_metric_kwargs():
    # Test reduce
    x_hat = torch.tensor([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert torch.all(metric.MSE(reduction="mean")(x_hat, x) == torch.tensor(7.0))
    assert torch.all(metric.MSE(reduction="sum")(x_hat, x) == torch.tensor(21.0))
    assert torch.all(
        metric.MSE(reduction="none")(x_hat, x) == torch.tensor([1.0, 4.0, 16.0])
    )

    # Test norm_inputs
    x_hat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert torch.all(
        metric.MSE(norm_inputs="min_max")(x_hat, x)
        == torch.tensor([0.5000, 0.5000, 0.5000])
    )
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    assert torch.allclose(
        metric.MSE(norm_inputs="l2")(x_hat, x),
        torch.tensor([0.0083, 0.0011, 0.0011]),
        atol=0.0001,
    )

    # Test complex_abs
    x = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
    x = torch.complex(x[:, 0, :], x[:, 0, :])  # tensor([[1.+1.j, 2.+2.j]])
    assert torch.allclose(
        metric.MSE(complex_abs=True)(x, x * 0), torch.tensor([5.0000])
    )

    x_hat = torch.tensor([[[1.0], [1.0]]])
    x = x_hat * 0
    assert torch.allclose(
        metric.MSE(complex_abs=True)(x_hat, x), torch.tensor([2.0000])
    )
    assert torch.all(metric.MSE(complex_abs=False)(x_hat, x) == torch.tensor([1.0]))

    # Test train loss does nothing as MSE already lower_better=True
    assert torch.all(metric.MSE(train_loss=True)(x_hat, x) == torch.tensor([1.0]))


def test_center_crop():
    # Test center crop with positive int value
    x = torch.ones(2, 3, 32, 32)
    x_hat = torch.zeros(2, 3, 32, 32)

    # Test with int crop size (crops all spatial dimensions equally)
    m = metric.MSE(center_crop=16)
    result = m(x_hat, x)
    # After cropping to 16x16, each sample should have MSE of 1.0
    assert torch.all(result == torch.tensor([1.0, 1.0]))

    # Test with tuple crop size (crops last len(tuple) dimensions)
    m = metric.MSE(center_crop=(8, 8))
    result = m(x_hat, x)
    # After cropping to 8x8, each sample should have MSE of 1.0
    assert torch.all(result == torch.tensor([1.0, 1.0]))

    # Test with negative crop value (removes pixels from borders)
    x = torch.ones(2, 3, 32, 32)
    x_hat = x.clone()
    # Set borders to 0
    x_hat[:, :, :4, :] = 0  # top
    x_hat[:, :, -4:, :] = 0  # bottom
    x_hat[:, :, :, :4] = 0  # left
    x_hat[:, :, :, -4:] = 0  # right

    m = metric.MSE(center_crop=-4)  # Remove 4 pixels from each border
    result = m(x_hat, x)
    # After removing borders, we should have identical images
    assert torch.all(result == torch.tensor([0.0, 0.0]))

    # Test with zero crop value (removes 0 pixels from borders, i.e., no crop)
    m = metric.MSE(center_crop=0)
    result = m(x_hat, x)
    # With zero crop, borders are included and MSE should be > 0
    assert torch.all(result > 0)

    # Test that crop works with different metrics
    x = torch.ones(1, 1, 64, 64)
    x_hat = torch.zeros(1, 1, 64, 64)

    m_psnr = metric.PSNR(center_crop=32)
    result_psnr = m_psnr(x_hat, x)
    assert result_psnr.shape == torch.Size([1])

    m_mae = metric.MAE(center_crop=32)
    result_mae = m_mae(x_hat, x)
    assert torch.all(result_mae == torch.tensor([1.0]))

    # Test that None means no cropping
    m_none = metric.MSE(center_crop=None)
    x = torch.ones(1, 1, 32, 32)
    x_hat = torch.zeros(1, 1, 32, 32)
    result_none = m_none(x_hat, x)
    assert torch.all(result_none == torch.tensor([1.0]))

    # Test error handling: crop size larger than dimension
    m = metric.MSE(center_crop=64)
    x = torch.ones(1, 1, 32, 32)
    try:
        m(x, x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "larger than dimension size" in str(e)

    # Test error handling: too many border pixels to remove
    m = metric.MSE(center_crop=-20)
    x = torch.ones(1, 1, 32, 32)
    try:
        m(x, x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "remove entire dimension" in str(e)

    # Test if tuple has mixed signs
    try:
        m = metric.MSE(center_crop=(-8, 8))
        m(x, x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert (
            "If center_crop is a tuple, all values must be either positive or negative."
            in str(e)
        )


@pytest.mark.parametrize("power_signal", [0.0, 1.0, 10.0])
@pytest.mark.parametrize("power_noise", [0.0, 1.0, 10.0])
def test_snr(power_signal, power_noise):
    x = torch.full((1, 1, 16, 16), math.sqrt(power_signal))
    y = x + torch.full((1, 1, 16, 16), math.sqrt(power_noise))

    snr = metric.signal_noise_ratio(y, x)

    if power_noise == 0.0 and power_signal == 0.0:
        assert torch.isnan(snr), f"Expected NaN SNR, got {snr.item()}"
    elif power_noise == 0.0:
        assert torch.isposinf(snr), f"Expected infinite SNR, got {snr.item()}"
    elif power_signal == 0.0:
        assert torch.isneginf(snr), f"Expected -infinite SNR, got {snr.item()}"
    else:
        target_snr = 10 * math.log10(power_signal / power_noise)
        assert torch.isclose(
            snr,
            torch.tensor(target_snr),
        ), f"Expected SNR {target_snr}, got {snr.item()}"


class _MockDataset(torch.utils.data.Dataset):
    def __init__(
        self, num_samples: int, resolution: int, num_channels: int, dtype=torch.dtype
    ):
        super().__init__()
        self.resolution = resolution
        self.n_samples = num_samples
        self.num_channels = num_channels
        self.dtype = dtype

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return torch.rand(
            (self.num_channels, self.resolution, self.resolution), dtype=self.dtype
        )


@pytest.mark.parametrize("n_channels", (1, 3))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.float64))
def test_niqe_fit(n_channels: int, dtype: torch.dtype):
    # General Note: Testing whether weights created are useful is too complex here.
    # Therefore, we simply use dummy inputs to confirm
    # (i) Errors are raised on certain inputs
    # (ii) NIQE fit works as expected, without unexpected errors

    # Test if informative error raised if weights=None and metric called
    niqe = metric.NIQE(patch_size=32, patch_overlap=16, weights_path=None, dtype=dtype)
    test_tensor = torch.ones((1, n_channels, 128, 128))
    with pytest.raises(RuntimeError) as exc_info:
        result = niqe.metric(test_tensor)
    assert (
        str(exc_info.value)
        == "NIQE weights not loaded. Either pass weights_path at init or call create_weights first."
    )
    # Test fail on too low resolution
    low_res_ds = _MockDataset(
        num_samples=1, resolution=31, num_channels=n_channels, dtype=dtype
    )
    with pytest.raises(RuntimeError) as exc_info:
        mu, cov = niqe.create_weights(low_res_ds)
    assert (
        str(exc_info.value)
        == "No patches collected. Consider lowering sharpness_threshold or checking dataset."
    )

    ds = _MockDataset(
        num_samples=2, resolution=40, num_channels=n_channels, dtype=dtype
    )
    mu, cov = niqe.create_weights(ds, sharpness_threshold=0.1)
    assert mu.shape == torch.Size([36]) and cov.shape == torch.Size([36, 36])
    assert mu.dtype == cov.dtype and mu.dtype == dtype
    assert niqe.mu_p is not None and niqe.cov_p is not None
    assert torch.equal(niqe.mu_p, mu) and torch.equal(niqe.cov_p, cov)


def test_niqe_other_implementations():
    results_other_implementations = {
        "baboon": {"PyIQA": 5.738537788391113, "basicsr": 5.90808476428397},
        "barbara": {"PyIQA": 4.465195178985596, "basicsr": 4.621669738931946},
        "bridge": {"PyIQA": 2.5123531818389893, "basicsr": 2.5121616143607324},
        "coastguard": {"PyIQA": 5.468789577484131, "basicsr": 5.282670788211967},
        "comic": {"PyIQA": 4.254405498504639, "basicsr": 3.9421781727240037},
        "face": {"PyIQA": 8.72698974609375, "basicsr": 9.189687465717332},
        "flowers": {"PyIQA": 2.9371178150177, "basicsr": 3.094643463111649},
        "foreman": {"PyIQA": 5.505414009094238, "basicsr": 5.381376548613031},
        "lenna": {"PyIQA": 5.060755729675293, "basicsr": 5.484477923940196},
        "man": {"PyIQA": 3.76017427444458, "basicsr": 3.662171460656096},
        "monarch": {"PyIQA": 3.4054934978485107, "basicsr": 3.4319501014991114},
        "pepper": {"PyIQA": 7.261951446533203, "basicsr": 6.261429595525796},
        "ppt3": {"PyIQA": 5.532331466674805, "basicsr": 5.300301903526748},
        "zebra": {"PyIQA": 3.379080295562744, "basicsr": 3.339659586462774},
    }
    dinv.datasets.Set14HR(root="set14_niqe_test", download=True)
    niqe = dinv.loss.metric.NIQE()
    for f in os.listdir("set14_niqe_test/Set14_HR"):
        fname = f.split(".")[0]
        img = Image.open(f"set14_niqe_test/Set14_HR/{f}").convert("RGB")
        arr = np.asarray(img).transpose(2, 0, 1)
        t = torch.tensor(arr).unsqueeze(0).to(dtype=torch.float32)
        result_dinv = float(niqe.metric(t))
        refs = results_other_implementations[fname]
        assert abs(result_dinv - refs["PyIQA"]) <= 1.0, (
            f"{fname}: deepinv={result_dinv:.4f}, PyIQA={refs['PyIQA']:.4f}, "
            f"diff={abs(result_dinv - refs['PyIQA']):.4f}"
        )
        assert abs(result_dinv - refs["basicsr"]) <= 1.0, (
            f"{fname}: deepinv={result_dinv:.4f}, basicsr={refs['basicsr']:.4f}, "
            f"diff={abs(result_dinv - refs['basicsr']):.4f}"
        )
