from abc import ABC, abstractmethod
import json
import csv
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Union
import warnings
import os
from logging import getLogger
from deepinv.utils import AverageMeter
from deepinv.loss import Loss, Metric

import os

import torch
import wandb


class RunLogger(ABC):
    """
    Abstract base class for logging training runs.

    TODO
    """

    @abstractmethod
    def start_run(self, hyperparams: Optional[dict[str, Any]] = None):
        """
        Start a new training run.

        :param dict hyperparams: Dictionary of hyperparameters to log.
        """
        pass

    @abstractmethod
    def log_losses(
        self,
        losses: dict[str, float],
        step: int,
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
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
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
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
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        Log images for visualization.

        :param images: Dictionary of images to log.
        :param int step: Current training step.
        :param int epoch: Current training epoch.
        :param str phase: Training phase ('train', 'val', 'test').
        """
        pass

    @abstractmethod
    def log_checkpoint(self, epoch: int, state: dict[str, Any]):
        """
        Log training checkpoint.

        :param int epoch: Save checkpoint at the end of an epoch.
        :param dict state: Contains model weights, optimizer states, LR scheduler, etc.
        """
        pass

    @abstractmethod
    def finish_run(self):
        """
        Finalize and close the training run.
        """
        pass


class WandbRunLogger(RunLogger):
    """
    TODO
    """

    def __init__(
        self,
        local_checkpoint_dir: str,
        root_save_dir: str,
        proj_name: str,
        run_name: str,
        logging_mode: str = "online",
        resume_id: str = None,
    ):
        """
        TODO
        """
        self.local_checkpoint_dir = Path(local_checkpoint_dir)
        self.save_dir = Path(root_save_dir)
        self.proj_name = proj_name
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

    def start_run(self, hyperparams: Optional[dict[str, Any]] = None) -> None:
        """ """
        # Get a dict that contains wandb settings and experiment metadata, necessary to launch a Wandb run
        wandb_setup = self.get_wandb_setup(
            wandb_save_dir=self.save_dir,
            wandb_proj_name=self.proj_name,
            wandb_run_name=self.run_name,
            wandb_hp_config=hyperparams,
            wandb_logging_mode=self.logging_mode,
            wandb_resume_id=self.resume_id,
        )

        # Start Wandb run
        self.wandb_run = wandb.init(**wandb_setup)

    def log_losses(
        self,
        losses: dict[str, Any],
        step: int,
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
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
        epoch: Optional[int] = None,
        phase: str = "train",
    ):
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
        step: Optional[int] = None,
        phase: str = "train",
    ):
        """
        TODO
        """
        step = None

        # process images
        for name_img, img in images.items():
            wandb_images = wandb.Image(img, caption=name_img)

            # log images
            logs = {}
            logs[f"{phase} samples"] = wandb_images
            self.wandb_run.log(logs, step=epoch)

    def log_checkpoint(self, epoch, state=None):
        """
        TODO
        """
        state = None

        checkpoint_file = (
            self.local_checkpoint_dir / f"checkpoint_epoch_{epoch}.pth.tar"
        )

        # Add wandb run id to either an existing checkpoint or a new checkpoint
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(
                checkpoint_file, map_location="cpu", weights_only=False
            )
            checkpoint["wandb_id"] = self.wandb_run.id
            torch.save(checkpoint, checkpoint_file)
        else:
            os.makedirs(self.local_checkpoint_dir, exist_ok=True)
            torch.save({"wandb_id": self.wandb_run.id})

    def finish_run(self):
        """
        TODO
        """
        self.wandb_run.finish()
