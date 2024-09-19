import pytest
import torch
import deepinv.loss.metric as metric

METRICS = ["MSE", "NMSE", "PSNR", "SSIM", "LpNorm", "L1L2", "LPIPS", "NIQE"]
FUNCTIONALS = ["cal_mse", "cal_psnr", "cal_angle"]

# TODO Test all metrics take kwargs and args appropriately
# skip pyiqa ones if not installed


def choose_metric(metric_name, complex_abs, train_loss) -> metric.Metric:
    if metric_name in ("LPIPS", "NIQE"):
        pytest.importorskip(
            "pyiqa",
            reason="This test requires pyiqa. It should be "
            "installed with `pip install pyiqa`",
        )
    if metric_name == "MSE":
        return metric.MSE(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "NMSE":
        return metric.NMSE(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "PSNR":
        return metric.PSNR(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "SSIM":
        return metric.SSIM(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "LpNorm":
        return metric.LpNorm(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "L1L2":
        return metric.L1L2(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "LPIPS":
        return metric.LPIPS(complex_abs=complex_abs, train_loss=train_loss)
    elif metric_name == "NIQE":
        return metric.NIQE(complex_abs=complex_abs, train_loss=train_loss)


@pytest.mark.parametrize("metric_name", METRICS)
@pytest.mark.parametrize("complex_abs", [True])
@pytest.mark.parametrize("train_loss", [True, False])
def test_metrics(metric_name, complex_abs, train_loss, imsize_2_channel, rng):
    m = choose_metric(metric_name, complex_abs, train_loss)
    x = torch.rand((1, *imsize_2_channel), generator=rng)
    x_hat = x * 0.0 + 0.01
    assert m(x_hat, x) != 0
    # Test various args and kwargs which could be passed to metrics
    assert m(x_hat, x, None, model=None, some_other_kwarg=None) != 0


@pytest.mark.parametrize("functional_name", FUNCTIONALS)
def test_functional(functional_name, imsize_2_channel, rng):
    x = torch.rand((1, *imsize_2_channel), generator=rng)
    x_net = torch.rand((1, *imsize_2_channel), generator=rng)

    if functional_name == "cal_mse":
        assert metric.cal_mse(x_net, x) == torch.nn.MSELoss()(x_net, x)
        assert metric.MSE()(x_net, x) == torch.nn.MSELoss()(x_net, x)
    elif functional_name == "cal_psnr":
        pytest.importorskip(
            "torchmetrics",
            reason="This test requires torchmetrics. It should be "
            "installed with `pip install torchmetrics`",
        )
        from torchmetrics.image import PeakSignalNoiseRatio

        assert torch.allclose(
            metric.cal_psnr(x_net, x, to_numpy=False),
            PeakSignalNoiseRatio(data_range=1.0)(x_net, x),
        )