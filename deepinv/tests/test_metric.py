import pytest
import torch
from deepinv.physics import GaussianNoise
import deepinv.loss.metric as metric
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils.demo import get_image_url, load_url_image

METRICS = ["MAE", "MSE", "NMSE", "PSNR", "SSIM", "LpNorm", "L1L2", "LPIPS", "NIQE"]
FUNCTIONALS = ["cal_mse", "cal_mae", "cal_psnr"]


def choose_metric(metric_name, **kwargs) -> metric.Metric:
    if metric_name in ("LPIPS", "NIQE"):
        pytest.importorskip(
            "pyiqa",
            reason="This test requires pyiqa. It should be "
            "installed with `pip install pyiqa`",
        )
    if metric_name == "MSE":
        return metric.MSE(**kwargs)
    elif metric_name == "NMSE":
        return metric.NMSE(**kwargs)
    elif metric_name == "MAE":
        return metric.MAE(**kwargs)
    elif metric_name == "PSNR":
        return metric.PSNR(**kwargs)
    elif metric_name == "SSIM":
        return metric.SSIM(**kwargs)
    elif metric_name == "LpNorm":
        return metric.LpNorm(**kwargs)
    elif metric_name == "L1L2":
        return metric.L1L2(**kwargs)
    elif metric_name == "LPIPS":
        return metric.LPIPS(**kwargs)
    elif metric_name == "NIQE":
        return metric.NIQE(**kwargs)


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("complex_abs", [True])
@pytest.mark.parametrize("train_loss", [True, False])
@pytest.mark.parametrize("norm_inputs", [None])
def test_metrics(metric_name, complex_abs, train_loss, norm_inputs, rng):
    m = choose_metric(
        metric_name,
        complex_abs=complex_abs,
        train_loss=train_loss,
        norm_inputs=norm_inputs,
        reduction="mean",
    )
    x = load_url_image(
        get_image_url("celeba_example.jpg"), img_size=128, resize_mode="resize"
    )

    if complex_abs:
        x = x[:, :2, ...]

    x_hat = GaussianNoise(sigma=0.1, rng=rng)(x)

    # Test metric worse when image worse
    if train_loss:  # All metrics lower = better
        assert m(x_hat, x).item() > m(x, x).item()

    # Test various args and kwargs which could be passed to metrics
    assert m(x_hat, x, None, model=None, some_other_kwarg=None) != 0

    # Test no reduce works
    x_hat = torch.cat([x_hat] * 3)
    m = choose_metric(
        metric_name,
        complex_abs=complex_abs,
        train_loss=train_loss,
        norm_inputs=norm_inputs,
        reduction="none",
    )
    assert len(m(x_hat, x_hat)) == 3


@pytest.mark.parametrize("functional_name", FUNCTIONALS)
def test_functional(functional_name, imsize_2_channel, rng):
    x = torch.rand((1, *imsize_2_channel), generator=rng)
    x_net = torch.rand((1, *imsize_2_channel), generator=rng)

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

        assert torch.allclose(
            metric.cal_psnr(x_net, x),
            PeakSignalNoiseRatio(data_range=1.0)(x_net, x),
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
