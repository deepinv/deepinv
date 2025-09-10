import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.utils import get_timestamp
from dummy import DummyCircles, DummyModel
from deepinv.training.trainer import Trainer
from deepinv.physics.generator.base import PhysicsGenerator
from deepinv.physics.forward import Physics
from deepinv.physics.noise import GaussianNoise, PoissonNoise
from deepinv.datasets.base import ImageDataset
from deepinv.utils.compat import zip_strict
from unittest.mock import patch
import math
import io
import contextlib
import re
import typing
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
    logger.start_run(hyperparams=hyperparam_dict)
    logger.stdout_logger.setLevel("ERROR")

    expected_log_dir = tmpdir / "test_project" / "test_run"
    assert logger.log_dir == expected_log_dir
    assert expected_log_dir.exists()

    # Verify that 4 subdirectories were created
    subdirs = [entry for entry in os.scandir(expected_log_dir) if entry.is_dir()]
    assert len(subdirs) == 4
    expected_dirs = ["losses", "metrics", "images", "checkpoints"]
    actual_dirs = sorted([d.name for d in subdirs])
    assert actual_dirs == expected_dirs

    # Verify hyperparameters content
    assert (expected_log_dir / "configs" / "hyperparams.json").exists()
    with open(expected_log_dir / "configs" / "hyperparams.json", "r") as f:
        saved_hyperparams = json.load(f)
    assert saved_hyperparams == hyperparam_dict


@pytest.mark.parametrize("epoch", [None, 1])
@pytest.mark.parametrize("phase", ["train", "val"])
def test_localrunner_log_losses(epoch, phase, tmpdir):
    logger = LocalLogger(
        log_dir=tmpdir, project_name="test_project", run_name="test_run"
    )
    logger.start_run()
    losses_dict = {"loss1": 0.5, "loss2": 0.3, "total_loss": 0.8}
    logger.log_losses(losses=losses_dict, step=1, epoch=epoch, phase=phase)

    # Get the log file path for the losses
    log_dir = tmpdir / "test_project" / "test_run"
    file_pattern = (
        f"{phase}_losses" if epoch is None else f"{phase}_epoch{epoch:03d}_losses"
    )
    log_file = log_dir / f"{file_pattern}.json"

    assert log_file.exists(), f"Log file {log_file} not found"

    with open(log_file, "r") as f:
        logged_losses = json.load(f)

    assert logged_losses[phase] == losses_dict
