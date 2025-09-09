from abc import ABC, abstractmethod
import json
import csv
import logging
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import warnings
from typing import Any, Optional, Union
import warnings
import os
from logging import getLogger
from deepinv.utils import AverageMeter
from deepinv.loss import Loss, Metric


class RunLogger(ABC):
    """
    Abstract base class for logging training runs.

    Defines the interface for logging metrics, losses, images, and other
    training artifacts during model training and evaluation.
    """

    def __init__(
        self, run_name: Optional[str] = None, config: Optional[dict[str, Any]] = None
    ):
        """
        Initialize the logger.

        :param run_name: Optional name for the training run
        :param config: Configuration dictionary for the logger
        """
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.current_epoch = 0
        self.current_step = 0

    @abstractmethod
    def start_run(self, hyperparams: Optional[dict[str, Any]] = None):
        """
        Start a new training run.

        :param hyperparams: Dictionary of hyperparameters to log
        """
        pass

    @abstractmethod
    def log_losses(
        self,
        losses: dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log loss values for the current step/epoch.

        :param losses: Dictionary of loss_name -> value
        :param step: Current training step
        :param epoch: Current epoch
        :param phase: Training phase ('train', 'eval', 'test')
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log metric values for the current step/epoch.

        :param metrics: Dictionary of metric_name -> value
        :param step: Current training step
        :param epoch: Current epoch
        :param phase: Training phase ('train', 'eval', 'test')
        """
        pass

    @abstractmethod
    def log_images(
        self,
        images: dict[str, Union[torch.Tensor, np.ndarray]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log images for visualization.

        :param images: Dictionary of image_name -> tensor/array
        :param step: Current training step
        :param epoch: Current epoch
        :param phase: Training phase ('train', 'eval', 'test')
        """
        pass

    @abstractmethod
    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        metrics: Optional[dict[str, float]] = None,
        epoch: Optional[int] = None,
    ):
        """
        Log model checkpoint information.

        :param checkpoint_path: Path to the saved checkpoint
        :param metrics: Optional metrics associated with this checkpoint
        :param epoch: Epoch when checkpoint was saved
        """
        pass

    @abstractmethod
    def finish_run(self):
        """
        Finalize and close the training run.
        """
        pass

    def set_step(self, step: int):
        """Set the current step."""
        self.current_step = step

    def set_epoch(self, epoch: int):
        """Set the current epoch."""
        self.current_epoch = epoch


class LocalLogger(RunLogger):
    """
    Concrete implementation of RunLogger that logs to local files.

    """

    def __init__(
        self,
        log_dir: str = "logs",
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(run_name, config)
        self.log_dir = Path(log_dir) / Path(project_name) / self.run_name
        self.loss_dir = self.log_dir / "losses"
        self.metrics_dir = self.log_dir / "metrics"
        self.images_dir = self.log_dir / "images"
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.loss_history = []

    def start_run(self, hyperparams: Optional[dict[str, Any]] = None):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Save hyperparameters
        if hyperparams:
            with open(self.log_dir / "hyperparams.json", "w") as f:
                json.dump(hyperparams, f, indent=4)

        # Setup logging to file
        self.stdout_logger = getLogger("stdout_logger")
        self.stdout_logger.setLevel("INFO")
        fh = logging.FileHandler(self.log_dir / "training.log")
        fh.setLevel("INFO")
        self.stdout_logger.addHandler(fh)

        # Setup average meters for losses
        self.img_counter = 0  # Initialize image counter

        # Initialize dictionaries to hold meters for different phases
        self.losses_train = {}
        self.losses_val = {}
        self.losses_test = {}
        self.metrics_train = {}
        self.metrics_val = {}
        self.metrics_test = {}

        # Initialize total loss meters for each phase
        self.total_loss_train = AverageMeter("train total_loss", ":.6f")
        self.total_loss_val = AverageMeter("val total_loss", ":.6f")
        self.total_loss_test = AverageMeter("test total_loss", ":.6f")

        self.loss_history = []

        self.stdout_logger.info(f"Run started: {self.run_name}")

    def log_losses(
        self,
        losses: dict[str, float],
        step: int,
        epoch: int,
        phase: str = "train",
    ):

        if phase == "train":
            meters = self.losses_train
            total_meter = self.total_loss_train
        elif phase == "val":
            meters = self.losses_val
            total_meter = self.total_loss_val
        elif phase == "test":
            meters = self.losses_test
            total_meter = self.total_loss_test
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Initialize meters for each loss if this is the first time
        for name, value in losses.items():
            if name not in meters:
                meters[name] = AverageMeter(f"{phase} {name}", ":.6f")

        for name, value in losses.items():
            meters[name].update(value)

        total_meter.update(losses["total_loss"])

        if phase == "train":
            self.loss_history.append(total_meter.avg)

        # Human readable logging
        loss_str = "| ".join(
            [f"{meter.name}: {meter.avg:.6f}" for meter in meters.values()]
        )
        self.stdout_logger.info(
            f"{phase} - epoch: {epoch} | step: {step} | losses: {loss_str}"
        )

        # JSON logging
        log_file = self.loss_dir / "losses.json"

        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    all_logs = json.load(f)
            except json.JSONDecodeError:
                all_logs = {}
        else:
            all_logs = {}

        if phase not in all_logs:
            all_logs[phase] = {}

        all_logs[phase]["epoch"] = epoch
        all_logs[phase]["step"] = step
        all_logs[phase]["losses"] = {name: meter.avg for name, meter in meters.items()}

        with open(log_file, "w") as f:
            json.dump(all_logs, f, indent=2)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int,
        phase: str = "train",
    ):
        if phase == "train":
            meters = self.metrics_train
        elif phase == "eval":
            meters = self.metrics_val
        elif phase == "test":
            meters = self.metrics_test
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Initialize meters for each metric if this is the first time
        for name, value in metrics.items():
            if name not in meters:
                meters[name] = AverageMeter(f"{phase} {name}", ":.6f")

        for name, value in metrics.items():
            meters[name].update(value)

        # Human readable logging
        metric_str = "| ".join(
            [f"{meter.name}: {meter.avg:.6f}" for meter in meters.values()]
        )
        self.stdout_logger.info(
            f"{phase} - epoch: {epoch} | step: {step} | metrics: {metric_str}"
        )

        # JSON logging
        log_file = self.metrics_dir / "metrics.json"

        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    all_logs = json.load(f)
            except json.JSONDecodeError:
                all_logs = {}
        else:
            all_logs = {}

        if phase not in all_logs:
            all_logs[phase] = {}

        all_logs[phase]["epoch"] = epoch
        all_logs[phase]["step"] = step
        all_logs[phase]["metrics"] = {name: meter.avg for name, meter in meters.items()}

        with open(log_file, "w") as f:
            json.dump(all_logs, f, indent=2)

    def log_images(
        self,
        images: dict[str, Union[torch.Tensor, np.ndarray]],
        epoch: int = 0,
        step: Optional[int] = None,
        phase: str = "train",
    ):
        dir_path = self.images_dir / phase / f"epoch_{epoch}"
        if step is not None:
            dir_path = dir_path / f"step_{step}"
        os.makedirs(dir_path, exist_ok=True)

        for k, (name, img) in enumerate(images.items()):
            for i in range(img.size(0)):
                img_name = f"{dir_path}/{name}_{k}_{self.img_counter + i}.png"
                save_image(img[i], img_name)

    def log_checkpoint(
        self,
        epoch: Optional[int] = None,
        state: dict[str, Any] = {},
    ):
        ckpt_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch}.pth.tar"

        torch.save(
            state,
            ckpt_path,
        )
        self.stdout_logger.info(f"Checkpoint saved at epoch {epoch}: {ckpt_path}")

    def finish_run(self):
        self.stdout_logger.info(f"Run finished: {self.run_name}")
        handlers = self.stdout_logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.stdout_logger.removeHandler(handler)
