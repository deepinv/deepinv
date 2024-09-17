import pytest
import deepinv as dinv
import deepinv.loss.metric as metric

METRICS = ["MSE", "NMSE", "PSNR", "SSIM", "LpNorm", "LPIPS", "NIQE"]

# TODO Test all metrics take kwargs and args appropriately
# skip pyiqa ones if not installed

def choose_metric(metric_name):
    if metric_name == "MSE":
        return metric.MSE()

# Test cal_mse gives torch.nn.MSELoss()

# Test cal_psnr gives torchmetrics one

