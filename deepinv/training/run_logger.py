from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any
import json
import logging
import os
import platform

from torchvision.utils import save_image
import torch
import wandb


def get_timestamp() -> str:
    """Get current timestamp string.

    :return str: timestamp, with separators determined by system.
    """
    # ":" is not allowed on Windows filenames
    sep = "_" if platform.system() == "Windows" else ":"
    return datetime.now().strftime(f"%y-%m-%d-%H{sep}%M{sep}%S")


@dataclass
class RunLogger(ABC):
    """
    Abstract base class for logging training runs.

    TODO
    """

    log_dir: str

    @abstractmethod
    def init_logger(self, hyperparams: dict[str, Any] | None = None) -> None:
        """
        Start a new training run.

        :param dict hyperparams: Dictionary of hyperparameters to log.
        """
        pass

    @abstractmethod
    def setLevel(self, level: str) -> None:
        """
        Set the logging level.

        :param str level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        pass

    @abstractmethod
    def log_losses(
        self,
        losses: dict[str, float],
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        Log loss values for the current step/epoch.

        :param dict losses: Dictionary of current losses values.
        :param int step: Current training step.
        :param int epoch: Current training epoch.
        :param str phase: Training phase ('train', 'val', 'test').
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        Log metrics for the current step/epoch.

        :param dict metrics: Dictionary of current metrics values.
        :param int step: Current training step.
        :param int epoch: Current training epoch.
        :param str phase: Training phase ('train', 'val', 'test').
        """
        pass

    @abstractmethod
    def log_images(
        self,
        images: dict[str, torch.Tensor],
        epoch: int,
        step: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        Log images for visualization.

        :param images: Dictionary of images to log.
        :param int step: Current training step.
        :param int epoch: Current training epoch.
        :param str phase: Training phase ('train', 'val', 'test').
        """
        pass

    @abstractmethod
    def load_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Resume the logger by restoring from a training checkpoint.

        :param dict state: Contains model weights, optimizer states, LR scheduler, etc.
        """
        pass

    @abstractmethod
    def log_checkpoint(
        self, epoch: int, state: dict[str, Any], name: str | None = None
    ) -> None:
        """
        Log training checkpoint (always save in a folder on the local machine).

        :param int epoch: Save checkpoint at the end of an epoch.
        :param dict state: Contains model weights, optimizer states, LR scheduler, etc.
        :param str name: Checkpoint filename.
        """
        pass

    @abstractmethod
    def finish_run(self) -> None:
        """
        Finalize and close the training run.
        """
        pass


class WandbLogger(RunLogger):
    """
    TODO
    """

    def __init__(
        self,
        local_checkpoint_dir: str,
        log_dir: str,
        project_name: str,
        run_name: str | None = None,
        logging_mode: str = "online",
        resume_id: str = None,
    ) -> None:
        """
        TODO
        """
        self.local_checkpoint_dir = Path(local_checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.proj_name = project_name
        if run_name is None:
            run_name = get_timestamp()
        self.run_name = run_name
        self.logging_mode = logging_mode
        self.resume_id = resume_id

    @classmethod
    def get_wandb_setup(
        cls,
        wandb_save_dir: str,
        wandb_proj_name: str,
        wandb_run_name: str,
        wandb_hp_config: dict[str, Any],
        wandb_logging_mode: str = "online",
        wandb_resume_id: str = None,
    ) -> dict[str, Any]:
        """
        TODO
        """
        if (
            wandb_resume_id is not None and wandb_logging_mode == "offline"
        ):  # https://github.com/wandb/wandb/issues/2423
            raise ValueError("Cannot resume wandb run with `wandb_logs_mode=offline`.")

        if wandb_resume_id is not None:  # setting to resume a wandb run
            wandb_setup = {
                "dir": wandb_save_dir,
                "mode": wandb_logging_mode,
                "project": wandb_proj_name,
                "id": wandb_resume_id,
                "resume": "must",
            }
        else:  # setting to create a new wandb run
            wandb_setup = {
                "dir": wandb_save_dir,
                "mode": wandb_logging_mode,
                "project": wandb_proj_name,
                "name": wandb_run_name,
                "config": wandb_hp_config,
            }
        return wandb_setup

    def init_logger(self, hyperparams: dict[str, Any] | None = None) -> None:
        """ """

        # Get a dict that contains wandb settings and experiment metadata, necessary to launch a Wandb run
        wandb_setup = self.get_wandb_setup(
            wandb_save_dir=self.log_dir,
            wandb_proj_name=self.proj_name,
            wandb_run_name=self.run_name,
            wandb_hp_config=hyperparams,
            wandb_logging_mode=self.logging_mode,
            wandb_resume_id=self.resume_id,
        )

        # Start Wandb run
        self.wandb_run = wandb.init(**wandb_setup)

    def setLevel(self, level: str) -> None:
        """
        Set the logging level.
        """
        # Wandb does not have a direct method to set logging level like standard loggers.
        # However, you can control the verbosity of the output using the `wandb.settings` module.

        if level.upper() == "DEBUG":
            wandb.settings().set("verbose", True)
        elif level.upper() in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            wandb.settings().set("verbose", False)
        else:
            raise ValueError(f"Unsupported logging level: {level}")

    def log_losses(
        self,
        losses: dict[str, Any],
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        TODO
        """
        epoch = None

        # {loss_name_1: loss_value_1, ...} -> {phase/loss_name_1: loss_value_1, ...}
        logs = {
            f"{phase}/{loss_name}": loss_value
            for loss_name, loss_value in losses.items()
        }

        # default x-axis is the current training step
        self.wandb_run.log(logs, step=step)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        TODO
        """
        epoch = None

        # {loss_name_1: loss_value_1, ...} -> {phase/loss_name_1: loss_value_1, ...}
        logs = {
            f"{phase}/{metric_name}": metric_value
            for metric_name, metric_value in metrics.items()
        }

        # default x-axis is the current training step
        self.wandb_run.log(logs, step=step)

    def log_images(
        self,
        images: dict[str, torch.Tensor],
        epoch: int,
        step: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        TODO

        Wandb expects NumPy array or PIL image.
        """
        step = None

        # process images
        for name_img, img in images.items():
            shape = img.shape
            if len(shape) == 2:
                wandb_images = wandb.Image(img.numpy())

                # log images
                self.wandb_run.log(
                    {f"{phase} samples: {name_img}": wandb_images}, step=epoch
                )
            elif len(shape) == 3:
                wandb_images = wandb.Image(img.permute(1, 2, 0).numpy())

                # log images
                self.wandb_run.log(
                    {f"{phase} samples: {name_img}": wandb_images}, step=epoch
                )
            elif len(shape) == 4:
                for j in range(len(img)):
                    wandb_images = wandb.Image(img[j].permute(1, 2, 0).numpy())

                    # log images
                    self.wandb_run.log(
                        {f"{phase} samples: {name_img}_{j}": wandb_images}, step=epoch
                    )

    def load_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        TODO
        """
        if "resume_id" in checkpoint:
            self.resume_id = checkpoint["resume_id"]

    def log_checkpoint(
        self, epoch: int, state: dict[str, Any], name: str | None = None
    ) -> None:
        """
        TODO
        """
        if state is None:
            state = {}

        if name is not None:
            checkpoint_file = self.local_checkpoint_dir / f"{name}.pth.tar"
        else:
            checkpoint_file = (
                self.local_checkpoint_dir / f"checkpoint_epoch_{epoch}.pth.tar"
            )

        # Add wandb run id to either an existing checkpoint or a new checkpoint
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(
                checkpoint_file, map_location="cpu", weights_only=False
            )  # this is a costly operation
            checkpoint["wandb_id"] = self.wandb_run.id
            torch.save(checkpoint, checkpoint_file)
        else:
            os.makedirs(self.local_checkpoint_dir, exist_ok=True)
            state["wandb_id"] = self.wandb_run.id
            torch.save(state)

    def finish_run(self) -> None:
        """
        TODO
        """
        self.wandb_run.finish()


class LocalLogger(RunLogger):
    """
    Concrete implementation of RunLogger that logs to local files.

    TODO
    """

    def __init__(
        self,
        log_dir: str = "logs",
        project_name: str | None = "default_project",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if run_name is None:
            run_name = get_timestamp()
        self.run_name = run_name
        self.stdout_logger = getLogger("stdout_logger")
        self.stdout_logger.setLevel("INFO")
        self.log_dir = Path(log_dir) / Path(project_name) / self.run_name
        self.loss_dir = self.log_dir / "losses"
        self.metrics_dir = self.log_dir / "metrics"
        self.images_dir = self.log_dir / "images"
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.loss_history = []

    def init_logger(self, hyperparams: dict[str, Any] | None = None) -> None:
        os.makedirs(self.log_dir, exist_ok=False)
        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Save hyperparameters
        if hyperparams:
            with open(self.log_dir / "hyperparams.json", "w") as f:
                json.dump(hyperparams, f, indent=4)

        # Setup logging to file
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

        self.loss_history = []

        self.stdout_logger.info(f"Log directory initialized: {self.log_dir}")

    def setLevel(self, level: str) -> None:
        """
        Set the logging level.
        """
        self.stdout_logger.setLevel(level)

    def log_losses(
        self,
        losses: dict[str, float],
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        if phase == "train":
            self.loss_history.append(["total_loss_avg"])

        # Human readable logging
        loss_str = "| ".join([f"{name}: {value:.6f}" for name, value in losses.items()])
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
            all_logs[phase] = []

        entry = {
            "epoch": epoch,
            "step": step,
            "losses": {name: float(value) for name, value in losses.items()},
        }

        # Add the new entry to the list
        all_logs[phase].append(entry)

        with open(log_file, "w") as f:
            json.dump(all_logs, f, indent=2)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        # Human readable logging
        metric_str = "| ".join(
            [f"{name}: {value:.6f}" for name, value in metrics.items()]
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
            all_logs[phase] = []

        entry = {
            "epoch": epoch,
            "step": step,
            "metrics": {name: float(value) for name, value in metrics.items()},
        }

        all_logs[phase].append(entry)

        with open(log_file, "w") as f:
            json.dump(all_logs, f, indent=2)

    def log_images(
        self,
        images: dict[str, torch.Tensor],
        epoch: int = 0,
        step: int | None = None,
        phase: str = "train",
    ) -> None:
        dir_path = self.images_dir / phase / f"epoch_{epoch}"
        if step is not None:
            dir_path = dir_path / f"step_{step}"
        os.makedirs(dir_path, exist_ok=True)

        for k, (name, img) in enumerate(images.items()):
            for i in range(img.size(0)):
                img_name = f"{dir_path}/{name}_{k}_{self.img_counter + i}.png"
                save_image(img[i], img_name)

    def load_from_checkpoint(self, checkpoint: dict[str, Any]):
        """
        TODO
        """
        pass

    def log_checkpoint(
        self,
        epoch: int,
        state: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        if state is None:
            state = {}

        if name is not None:
            ckpt_path = self.checkpoints_dir / f"{name}.pth.tar"
        else:
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
