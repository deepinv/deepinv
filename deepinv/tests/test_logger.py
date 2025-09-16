import pytest

from deepinv.training.run_logger import LocalLogger
import os
import json


def test_localrunner_start_run(
    tmpdir,
):

    logger = LocalLogger(
        log_dir=tmpdir, project_name="test_project", run_name="test_run"
    )
    hyperparam_dict = {"lr": 0.001, "batch_size": 32}
    logger.init_logger(hyperparams=hyperparam_dict)
    logger.stdout_logger.setLevel("CRITICAL")

    expected_log_dir = tmpdir / "test_project" / "test_run"
    assert logger.log_dir == expected_log_dir
    assert expected_log_dir.exists()

    # Verify that 4 subdirectories were created
    subdirs = [entry for entry in os.scandir(expected_log_dir) if entry.is_dir()]
    assert len(subdirs) == 4
    expected_dirs = ["checkpoints", "images", "losses", "metrics"]
    actual_dirs = sorted([d.name for d in subdirs])
    assert actual_dirs == expected_dirs

    # Verify hyperparameters content
    assert (expected_log_dir / "hyperparams.json").exists()
    with open(expected_log_dir / "hyperparams.json", "r") as f:
        saved_hyperparams = json.load(f)
    assert saved_hyperparams == hyperparam_dict


@pytest.mark.parametrize("epoch", [None, 1])
@pytest.mark.parametrize("phase", ["train", "val"])
def test_localrunner_log_losses(epoch, phase, tmpdir):
    logger = LocalLogger(
        log_dir=tmpdir, project_name="test_project", run_name="test_run"
    )
    logger.init_logger()
    logger.stdout_logger.setLevel("CRITICAL")
    losses_dict = {"loss1": 0.5, "loss2": 0.3, "total_loss": 0.8}
    logger.log_losses(losses=losses_dict, step=1, epoch=epoch, phase=phase)
    logger.log_losses(losses=losses_dict, step=2, epoch=epoch, phase=phase)

    # Get the log file path for the losses
    log_dir = tmpdir / "test_project" / "test_run"
    log_file = log_dir / "losses" / "losses.json"

    assert log_file.exists(), f"Log file {log_file} not found"

    with open(log_file, "r") as f:
        logged_losses = json.load(f)

    assert logged_losses[phase][0]["losses"] == losses_dict
    assert logged_losses[phase][1]["losses"] == losses_dict


@pytest.mark.parametrize("epoch", [None, 1])
@pytest.mark.parametrize("phase", ["train", "val"])
def test_localrunner_log_metrics(epoch, phase, tmpdir):
    logger = LocalLogger(
        log_dir=tmpdir, project_name="test_project", run_name="test_run"
    )
    logger.init_logger()
    logger.stdout_logger.setLevel("CRITICAL")
    metrics_dict = {"psnr": 25.0, "ssim": 0.85}
    logger.log_metrics(metrics=metrics_dict, step=1, epoch=epoch, phase=phase)
    logger.log_metrics(metrics=metrics_dict, step=2, epoch=epoch, phase=phase)

    # Get the log file path for the metrics
    log_dir = tmpdir / "test_project" / "test_run"
    log_file = log_dir / "metrics" / "metrics.json"

    assert log_file.exists(), f"Log file {log_file} not found"

    with open(log_file, "r") as f:
        logged_metrics = json.load(f)

    assert logged_metrics[phase][0]["metrics"] == metrics_dict
    assert logged_metrics[phase][1]["metrics"] == metrics_dict
