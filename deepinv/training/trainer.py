from dataclasses import dataclass, field
from logging import getLogger
from typing import Union, Optional
import inspect
import os
import warnings

from tqdm import tqdm
import numpy as np
import torch

from deepinv.datasets.base import check_dataset
from deepinv.loss import Loss, SupLoss, BaseLossScheduler
from deepinv.loss.metric import PSNR, Metric
from deepinv.physics import Physics
from deepinv.physics.generator import PhysicsGenerator
from deepinv.training.run_logger import RunLogger, LocalLogger
from deepinv.utils import AverageMeter
from deepinv.utils.compat import zip_strict
from deepinv.utils.plotting import prepare_images


@dataclass
class Trainer:
    """
    TODO
    """

    ## Core Components
    model: torch.nn.Module
    iterative_model_returns_different_outputs: bool = False
    physics: Union[Physics, list[Physics]]
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"

    ## Data Loading
    train_dataloader: Union[
        torch.utils.data.DataLoader, list[torch.utils.data.DataLoader]
    ] = None
    val_dataloader: Union[
        torch.utils.data.DataLoader, list[torch.utils.data.DataLoader]
    ] = None
    test_dataloader: Union[
        torch.utils.data.DataLoader, list[torch.utils.data.DataLoader]
    ] = None

    ## Generate measurements for training purpose with `physics`
    online_measurements: bool = False
    physics_generator: PhysicsGenerator | list[PhysicsGenerator] = None
    loop_random_online_physics: bool = False

    ## Training Control
    optimizer: torch.optim.Optimizer = None
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    grad_clip: float = None
    optimizer_step_multi_dataset: bool = True

    ## Training Duration & Stopping
    epochs: int = 100
    max_batch_steps: int = 10**10
    early_stop: bool = False

    ## Loss & Metrics
    losses: Union[Loss, BaseLossScheduler, list[Loss], list[BaseLossScheduler]] = (
        SupLoss()
    )
    metrics: Union[Metric, list[Metric]] = field(default_factory=PSNR)
    compare_no_learning: bool = False
    no_learning_method: str = "A_adjoint"

    ## Checkpointing & Persistence (at the end of epoch)
    ckpt_pretrained: str = None
    ckpt_interval: int = 1

    ## Logging & Monitoring
    loggers: Optional[Union[RunLogger, list[RunLogger]]] = field(
        default_factory=lambda: [LocalLogger("./logs")]
    )
    log_images: bool = False
    rescale_mode: str = "clip"
    log_grad: bool = False
    show_progress_bar: bool = True
    verbose: bool = True

    def setup_run(self) -> None:
        r"""
        Set up the training/testing process.

        :param bool train: whether model is being trained.
        """
        # resume state from a training checkpoint
        self.epoch_start = 0
        self.load_ckpt(self.ckpt_pretrained)

        self._setup_data()
        self._setup_logging()

    def _setup_data(self) -> None:
        """
        Set up data and physics before running an experience.
        """
        # default value when dataloaders are not defined
        if self.train_dataloader is None:
            self.train_dataloader = []
        if self.val_dataloader is None:
            self.val_dataloader = []
        if self.test_dataloader is None:
            self.test_dataloader = []

        # ensure that train, val, and test are list for format consistency
        if not isinstance(self.train_dataloader, list):
            self.train_dataloader = [self.train_dataloader]
        if not isinstance(self.val_dataloader, list):
            self.val_dataloader = [self.val_dataloader]
        if not isinstance(self.test_dataloader, list):
            self.test_dataloader = [self.test_dataloader]

        # ensure that dataset in each dataloader respects the right format
        for loader in self.train_dataloader + self.val_dataloader:
            check_dataset(loader.dataset)

        # useful when training on multiple dataset
        self.G = len(self.train_dataloader)

        # ensure that physics is a list for format consistency
        if not isinstance(self.physics, list):
            self.physics = [self.physics]

        # online measurements setting
        if self.physics_generator is not None:
            if not self.online_measurements:
                warnings.warn(
                    "Since `online_measurement` is False, `physics` will not be used to generate degraded images."
                )
            elif self.loop_random_online_physics:
                warnings.warn(
                    "Generated measurements repeat each epoch."
                    "Ensure that dataloader is not shuffling."
                )

            # ensure that physics_generator is a list for format consistency
            if not isinstance(self.physics_generator, list):
                self.physics_generator = [self.physics_generator]

    def _setup_logging(self) -> None:
        r"""
        Set up the monitoring before running an experience..
        """
        # losses processing
        if not isinstance(self.losses, list):
            self.losses = [self.losses]
        for l in self.losses:
            self.model = l.adapt_model(self.model)

        # metrics processing
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        # losses computed during an epoch
        self.meter_total_loss_train = AverageMeter("Training loss", ":.2e")
        self.meters_losses_train = {
            l.__class__.__name__: AverageMeter(
                "Validation loss " + l.__class__.__name__, ":.2e"
            )
            for l in self.losses
        }

        self.meter_total_loss_val = AverageMeter("Validation loss", ":.2e")
        self.meters_losses_val = {
            l.__class__.__name__: AverageMeter(
                "Validation loss " + l.__class__.__name__, ":.2e"
            )
            for l in self.losses
        }
        if not hasattr(self, "train_loss_history"):
            self.train_loss_history = []

        # metrics computed during an epoch
        self.meters_metrics_train = {
            l.__class__.__name__: AverageMeter(
                "Validation metric " + l.__class__.__name__, ":.2e"
            )
            for l in self.metrics
        }

        self.meters_metrics_val = {
            l.__class__.__name__: AverageMeter(
                "Validation metric " + l.__class__.__name__, ":.2e"
            )
            for l in self.metrics
        }
        if not hasattr(self, "val_metrics_history_per_epoch"):
            self.val_metrics_history_per_epoch = {
                l.__class__.__name__: [] for l in self.metrics
            }

        if self.compare_no_learning:
            self.meters_metrics_no_learning = {
                l.__class__.__name__: AverageMeter(
                    "Validation metric " + l.__class__.__name__, ":.2e"
                )
                for l in self.metrics
            }

        # Init logger specific for the Trainer
        self.train_logger = getLogger("train_logger")

        # Init other loggers (File logging, Wandb logging, etc.)
        if self.loggers is None:
            self.loggers = []
        if not isinstance(self.loggers, list):
            self.loggers = [self.loggers]
        for logger in self.loggers:
            if not isinstance(logger, RunLogger):
                raise ValueError("loggers should be a list of RunLogger instances.")

        # Set verbosity level of loggers
        if self.verbose:
            self.train_logger.setLevel("DEBUG")
            for logger in self.loggers:
                logger.setLevel("DEBUG")
        else:
            self.train_logger.setLevel("WARNING")
            for logger in self.loggers:
                logger.setLevel("WARNING")

    def save_ckpt(self, epoch: int, name: str = None) -> None:
        r"""
        Save necessary information to resume training.

        :param int epoch: Current epoch.
        :param str name: Name of the checkpoint file.
        """
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "loss": self.train_loss_history,
            "val_metrics": self.val_metrics_history_per_epoch,
        }

        for logger in self.loggers:
            logger.log_checkpoint(epoch=epoch, state=state, name=name)

    def load_ckpt(
        self,
        ckpt_pretrained: Optional[str] = None,
    ) -> None:
        """Load model from checkpoint.

        :param str ckpt_pretrained: Path to the checkpoint file.
        """
        if ckpt_pretrained is not None:
            self.ckpt_pretrained = ckpt_pretrained

        # Load checkpoint from file
        checkpoint = torch.load(
            self.ckpt_pretrained, map_location=self.device, weights_only=False
        )

        self.epoch_start = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.train_loss_history = checkpoint["loss"]
        self.val_metrics_history_per_epoch = checkpoint["val_metrics"]

        # Optimizer and Scheduler may be None
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        for logger in self.loggers:
            logger.load_from_checkpoint(checkpoint)

    def apply_clip_grad(self) -> float:
        r"""
        Perform gradient clipping.
        """
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.log_grad:
            # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/7
            grads = [
                param.grad.detach().flatten()
                for param in self.model.parameters()
                if param.grad is not None
            ]
            return torch.cat(grads).norm().item()
        return None

    def get_samples_online(self, iterators, g):
        r"""
        Get the samples for the online measurements.

        In this setting, a new sample is generated at each iteration by calling the physics operator.
        This function returns a dictionary containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a tuple containing at least: the ground truth, the measurement, and the current physics operator.
        """
        data = next(iterators[g])

        params = {}
        if isinstance(data, (tuple, list)):
            x = data[0]

            if len(data) == 2 and isinstance(data[1], dict):
                params = data[1]
            else:
                warnings.warn(
                    "Generating online measurements from data x but dataloader returns tuples (x, ...). Discarding all data after x."
                )
        else:
            x = data

        if torch.isnan(x).all():
            raise ValueError("Online measurements can't be used if x is all NaN.")

        x = x.to(self.device)
        physics = self.physics[g]

        if self.physics_generator is not None:
            if params:  # not empty params
                warnings.warn(
                    "Physics generator is provided but dataloader also returns params. Ignoring params from dataloader."
                )
            params = self.physics_generator[g].step(batch_size=x.size(0))

        # Update parameters both via update and, if implemented in physics, via forward pass
        physics.update(**params)
        y = physics(x, **params)

        return x, y, physics

    def get_samples_offline(self, iterators, g):
        r"""
        Get the samples for the offline measurements.

        In this setting, samples have been generated offline and are loaded from the dataloader.
        This function returns a tuple containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data
        (you can override this function to add custom data).

        If the dataloader returns 3-tuples, this is assumed to be ``(x, y, params)`` where
        ``params`` is a dict of physics generator params. These params are then used to update
        the physics.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a dictionary containing at least: the ground truth, the measurement, and the current physics operator.
        """
        data = next(iterators[g])
        if not isinstance(data, (tuple, list)) or len(data) < 2:
            raise ValueError(
                "If online_measurements=False, the dataloader should output a tuple (x, y) or (x, y, params)"
            )

        if len(data) == 2:
            x, y, params = *data, None
            if isinstance(y, dict):  # x,params offline
                raise ValueError(
                    "If online_measurements=False, measurements y must be provided as a tensor."
                )
        elif len(data) == 3:
            x, y, params = data
        else:
            raise ValueError(
                "Dataloader returns too many items. For offline learning, dataloader should either return (x, y) or (x, y, params)."
            )

        batch_size_y = y[0].size(0) if isinstance(y, TensorList) else y.size(0)
        batch_size_x = x[0].size(0) if isinstance(x, TensorList) else x.size(0)

        if batch_size_x != batch_size_y:  # pragma: no cover
            raise ValueError(
                f"Data x, y must have same batch size, but got {batch_size_x}, {batch_size_y}"
            )

        if torch.isnan(x).all() and x.ndim <= 1:
            x = None  # Batch of NaNs -> no ground truth in deepinv convention
        else:
            x = x.to(self.device)

        y = y.to(self.device)
        physics = self.physics[g]

        if params is not None:
            params = {
                k: (p.to(self.device) if isinstance(p, torch.Tensor) else p)
                for k, p in params.items()
            }
            physics.update(**params)

        return x, y, physics

    def get_samples(self, iterators, g):
        r"""
        Get the samples.

        This function returns a dictionary containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: the tuple returned by the get_samples_online or get_samples_offline function.
        """
        # In case someone wants samples without actually launching a run
        self._setup_data()
        if self.online_measurements:  # the measurements y are created on-the-fly
            x, y, physics = self.get_samples_online(iterators, g)
        else:  # the measurements y were pre-computed
            x, y, physics = self.get_samples_offline(iterators, g)

        if x is not None:  # If x is None, we are in unsupervised case
            if torch.isinf(x).any() or torch.isnan(x).any():
                warnings.warn("x contains NaN or inf values.")

        return x, y, physics

    def model_inference(self, y, physics, x=None, train=True, **kwargs) -> torch.Tensor:
        r"""
        Perform the model inference.

        It returns the network reconstruction given the samples.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Optional ground truth, used for computing convergence metrics.
        :returns torch.Tensor: The network reconstruction.
        """
        y = y.to(self.device)

        kwargs = {}

        # check if the forward has 'update_parameters' method, and if so, update the parameters
        if "update_parameters" in inspect.signature(self.model.forward).parameters:
            kwargs["update_parameters"] = True

        if not train:
            with torch.no_grad():
                if self.iterative_model_returns_different_outputs:
                    x_net, _ = self.model(
                        y, physics, x_gt=x, compute_metrics=True, **kwargs
                    )
                else:
                    x_net = self.model(y, physics, **kwargs)
        else:
            x_net = self.model(y, physics, **kwargs)

        return x_net

    def compute_loss(
        self, x, x_net, y, physics, train=True, epoch: int = None
    ) -> torch.Tensor:
        r"""
        Compute the loss.

        Loss can be the sum of several individual losses and we keep track of these individual losses.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: Current epoch.
        :returns torch.Tensor: The loss evaluated on the current batch.
        """
        # Compute the losses
        loss_total = 0
        with torch.set_grad_enabled(
            train
        ):  # dynamically choose whether to track gradients
            for (
                l
            ) in self.losses:  # global loss is computed as the sum of individual losses
                loss = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                )
                loss_total += (
                    loss.mean()
                )  # average of the current loss function evaluted on our batch of data

                if len(self.losses) > 1:
                    meter = (
                        self.meters_losses_train[l.__class__.__name__]
                        if train
                        else self.meters_losses_val[l.__class__.__name__]
                    )
                    meter.update(
                        loss.detach().cpu().numpy()
                    )  # track the current loss per img (img are from several batches)

            meter = self.meter_total_loss_train if train else self.meter_total_loss_val
            meter.update(loss_total.item())

        return loss_total

    def compute_metrics(
        self, x, x_net, y, physics, train=True, epoch: int = None
    ) -> None:
        r"""
        Compute the metrics.

        It computes the metrics over the batch.
        During val/test, we can compare our model to a baseline linear reconstruction.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: Current epoch.
        :returns: The logs with the metrics.
        """
        with torch.no_grad():
            for m in self.metrics:
                metric = m(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                )
                meter = (
                    self.meters_metrics_train[m.__class__.__name__]
                    if train
                    else self.meters_metrics_val[m.__class__.__name__]
                )
                meter.update(
                    metric.detach().cpu().numpy()
                )  # track the current metric per img (img are from several batches)

                if not train and self.compare_no_learning:
                    x_lin = self.no_learning_inference(y, physics)
                    metric = m(x=x, x_net=x_lin, y=y, physics=physics, model=self.model)
                    self.meters_metrics_no_learning[m.__class__.__name__].update(
                        metric.detach().cpu().numpy()
                    )

    def no_learning_inference(self, y, physics):
        r"""
        Perform the no learning inference.

        By default it returns the (linear) pseudo-inverse reconstruction given the measurement.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :returns: Reconstructed image.
        """
        y = y.to(self.device)
        if self.no_learning_method == "A_adjoint" and hasattr(physics, "A_adjoint"):
            if isinstance(physics, torch.nn.DataParallel):
                x_nl = physics.module.A_adjoint(y)
            else:
                x_nl = physics.A_adjoint(y)
        elif self.no_learning_method == "A_dagger" and hasattr(physics, "A_dagger"):
            if isinstance(physics, torch.nn.DataParallel):
                x_nl = physics.module.A_dagger(y)
            else:
                x_nl = physics.A_dagger(y)
        elif self.no_learning_method == "prox_l2" and hasattr(physics, "prox_l2"):
            # this is a regularized version of the pseudo-inverse, with an l2 regularization
            # with parameter set to 5.0 for a mild regularization
            if isinstance(physics, torch.nn.DataParallel):
                x_nl = physics.module.prox_l2(0.0, y, 5.0)
            else:
                x_nl = physics.prox_l2(0.0, y, 5.0)
        elif self.no_learning_method == "y":
            x_nl = y
        else:
            raise ValueError(
                f"No learning reconstruction method {self.no_learning_method} not recognized or physics does not implement it"
            )

        return x_nl

    class _NoLearningModel(Reconstructor):
        def __init__(self, *, trainer: Trainer):
            super().__init__()
            self.trainer = trainer

        def forward(self, y, physics, **kwargs):
            if kwargs:
                warnings.warn(
                    f"The learning-free model in Trainer expects no keyword argument, but got {list(kwargs.keys())}. "
                    "You might be using metrics which pass extra arguments to the trained model but the learning-free model does not use them.",
                    UserWarning,
                    stacklevel=1,
                )
            return self.trainer.no_learning_inference(y, physics)

    def step(
        self,
        epoch: int,
        progress_bar: tqdm,
        train_ite: int = None,
        train: bool = True,
        last_batch: bool = False,
    ) -> None:
        r"""
        Train/Eval a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration.

        :param int epoch: Current epoch.
        :param progress_bar: `tqdm <https://tqdm.github.io/docs/tqdm/>`_ progress bar.
        :param int train_ite: train iteration, only needed for logging if ``Trainer.log_train_batch=True``
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        """
        # Zero grad
        if train and self.optimizer_step_multi_dataset:
            self.optimizer.zero_grad()  # Clear stored gradients

        loss_multi_dataset_step = 0.0
        for g in np.random.permutation(self.G):  # g is the dataloader index

            # Zero grad
            if train and not self.optimizer_step_multi_dataset:
                self.optimizer.zero_grad()

            # Get either online or offline samples
            x, y, physics_cur = self.get_samples(
                self.current_train_iterators if train else self.current_val_iterators,
                g,
            )

            # Evaluate reconstruction network
            x_net = self.model_inference(y=y, physics=physics_cur, x=x, train=train)

            # Compute the loss for the batch
            loss_cur = self.compute_loss(
                x,
                x_net,
                y,
                physics_cur,
                train=train,
                epoch=epoch,
            )
            loss_multi_dataset_step += loss_cur.item()

            # Backward + Optimizer
            if train:
                loss_cur.backward()

            if train and not self.optimizer_step_multi_dataset:
                loss_logs = {}
                loss_logs["Loss"] = loss_cur.item()

                # Gradient clipping
                grad_norm = self.apply_clip_grad()
                if self.log_grad:
                    loss_logs["gradient_norm"] = grad_norm

                # Optimizer step
                self.optimizer.step()

                # Update the progress bar
                progress_bar.set_postfix(loss_logs)

            # Compute the metrics for the batch
            x_net = x_net.detach()  # detach the network output for metrics and plotting
            self.compute_metrics(x, x_net, y, physics_cur, train=train, epoch=epoch)

            # Log images of last batch for each dataset
            if last_batch:
                self.save_images(
                    epoch,
                    physics_cur,
                    x,
                    y,
                    x_net,
                    train=train,
                )

        if train and self.optimizer_step_multi_dataset:
            loss_logs = {}
            loss_logs["Loss"] = loss_multi_dataset_step

            # Gradient clipping
            grad_norm = self.apply_clip_grad()
            if self.log_grad:
                loss_logs["gradient_norm"] = grad_norm

            # Optimizer step
            self.optimizer.step()

            # Update the progress bar
            progress_bar.set_postfix(loss_logs)

            # Track loss value used for an update and its gradient norm
            for logger in self.loggers:
                logger.log_losses(loss_logs, step=train_ite, phase="train")

        # LOG EPOCH LOSSES AND METRICS
        if last_batch:
            phase = "train" if train else "val"

            ## LOSSES
            epoch_loss_logs = {}

            # add individual losses over an epoch
            if len(self.losses) > 1:
                for l in self.losses:
                    meter = (
                        self.meters_losses_train[l.__class__.__name__]
                        if train
                        else self.meters_losses_val[l.__class__.__name__]
                    )
                    epoch_loss_logs[l.__class__.__name__] = meter.avg

            # add total loss over an epoch
            meter = self.meter_total_loss_train if train else self.meter_total_loss_val
            epoch_loss_logs["Total_Loss"] = meter.avg

            ## METRICS
            epoch_metrics_logs = {}
            for m in self.metrics:
                meter = (
                    self.meters_metrics_train[m.__class__.__name__]
                    if train
                    else self.meters_metrics_val[m.__class__.__name__]
                )
                epoch_metrics_logs[m.__class__.__name__] = meter.avg

            ## LOG
            for logger in self.loggers:
                logger.log_losses(epoch_loss_logs, step=epoch, phase=phase)
                logger.log_metrics(epoch_metrics_logs, step=epoch, phase=phase)

    def save_images(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Save the reconstructions.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        if self.log_images:
            if self.compare_no_learning:
                x_nl = self.no_learning_inference(y, physics)
            else:
                x_nl = None

            imgs, titles, grid_image, caption = prepare_images(
                x, y=y, x_net=x_net, x_nl=x_nl, rescale_mode=self.rescale_mode
            )
            dict_imgs = {t: im for t, im in zip_strict(titles, imgs)}

            for logger in self.loggers:
                logger.log_images(
                    dict_imgs, epoch=epoch, phase="train" if train else "val"
                )

    def reset_losses(self) -> None:
        r"""
        Reset the epoch losses.
        """
        self.meter_total_loss_train.reset()
        self.meter_total_loss_val.reset()

        for l in self.meters_losses_train.values():
            l.reset()

        for l in self.meters_losses_val.values():
            l.reset()

    def reset_metrics(self) -> None:
        r"""
        Reset the epoch metrics.
        """
        for l in self.meters_metrics_train.values():
            l.reset()

        for l in self.meters_metrics_val.values():
            l.reset()

    def save_best_model(self, epoch) -> None:
        r"""
        Save the best model using validation metrics.

        By default, uses validation based on first metric. Override this method to provide custom criterion.

        :param int epoch: Current epoch.
        """
        k = 0  # index of the first metric
        history = self.val_metrics_history_per_epoch[self.metrics[k].__class__.__name__]
        lower_better = getattr(self.metrics[k], "lower_better", True)

        best_metric = min(history) if lower_better else max(history)
        curr_metric = history[-1]
        if (lower_better and curr_metric <= best_metric) or (
            not lower_better and curr_metric >= best_metric
        ):
            self.save_ckpt(epoch=epoch, name="ckpt_best")
            self.train_logger.info(
                f"Best model saved at epoch {epoch + 1}, {self.metrics[k].__class__.__name__}: {curr_metric:.4f}"
            )

    def stop_criterion(self, epoch, train_ite, **kwargs) -> bool:
        r"""
        Stop criterion for early stopping.

        By default, stops optimization when first eval metric doesn't improve in the last 3 evaluations.

        Override this method to early stop on a custom condition.

        :param int epoch: Current epoch.
        :param int train_ite: Current training batch iteration, equal to (current epoch :math:`\times` number
            of batches) + current batch within epoch
        :param dict metric_history: Dictionary containing the metrics history, with the metric name as key.
        :param list metrics: List of metrics used for evaluation.
        """
        k = 0  # use first metric

        history = self.val_metrics_history_per_epoch[self.metrics[k].__class__.__name__]
        lower_better = getattr(self.metrics[k], "lower_better", True)

        best_metric = min(history) if lower_better else max(history)
        best_epoch = history.index(best_metric)

        early_stop = epoch > 2 + best_epoch
        if early_stop:
            self.train_logger.info(
                "Early stopping triggered as validation metrics have not improved in "
                "the last 2 epochs, disable it with early_stop=False"
            )

        return early_stop

    def train(
        self,
    ) -> nn.Module:
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        :returns: The trained model.
        """
        self.setup_run()

        # count the overall training parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.train_logger.info(f"The model has {params} trainable parameters.")

        stop_flag = False
        for epoch in range(self.epoch_start, self.epochs):
            self.reset_losses()  # epoch losses
            self.reset_metrics()  # epoch metrics

            ## Training
            self.current_train_iterators = [
                iter(loader) for loader in self.train_dataloader
            ]
            batches = min(
                [len(loader) - loader.drop_last for loader in self.train_dataloader]
            )

            if self.loop_random_online_physics and self.physics_generator is not None:
                for physics_generator in self.physics_generator:
                    physics_generator.reset_rng()

                for physics in self.physics:
                    if hasattr(physics.noise_model, "reset_rng"):
                        physics.noise_model.reset_rng()

            self.model.train()
            for i in (
                progress_bar := tqdm(
                    range(batches),
                    dynamic_ncols=True,
                    ncols=0,
                    disable=(not self.show_progress_bar),
                )
            ):
                progress_bar.set_description(f"Train epoch {epoch + 1}/{self.epochs}")
                train_ite = (epoch * batches) + i
                self.step(
                    epoch,
                    progress_bar,
                    train_ite=train_ite,
                    train=True,
                    last_batch=(i == batches - 1),
                )
                if train_ite + 1 > self.max_batch_steps:
                    stop_flag = True

            if self.scheduler:
                self.scheduler.step()

            ## Validation
            if self.val_dataloader is not None or self.val_dataloader is not []:
                self.current_val_iterators = [
                    iter(loader) for loader in self.val_dataloader
                ]
                val_batches = min(
                    [len(loader) - loader.drop_last for loader in self.val_dataloader]
                )

                self.model.eval()
                for j in (
                    val_progress_bar := tqdm(
                        range(val_batches),
                        dynamic_ncols=True,
                        disable=(not self.show_progress_bar),
                        colour="green",
                        ncols=0,
                    )
                ):
                    val_progress_bar.set_description(
                        f"Eval epoch {epoch + 1}/{self.epochs}"
                    )
                    self.step(
                        epoch,
                        val_progress_bar,
                        train_ite=train_ite,
                        train=False,
                        last_batch=(j == val_batches - 1),
                    )

            ## Early stopping
            self.train_loss_history.append(self.meter_total_loss_train.avg)
            for m in self.metrics:
                self.val_metrics_history_per_epoch[m.__class__.__name__].append(
                    self.meters_metrics_val[m.__class__.__name__].avg
                )  # store in metrics history

            self.save_best_model(epoch)
            if (epoch % self.ckpt_interval == 0) or epoch + 1 == self.epochs:
                self.save_ckpt(epoch=epoch)

            if self.early_stop:
                stop_flag = self.stop_criterion(epoch, train_ite)
            if stop_flag:
                break

        for logger in self.loggers:
            logger.finish_run()

        return self.model

    def test(
        self,
        test_dataloader,
        compare_no_learning: bool = True,
        log_raw_metrics: bool = False,
    ) -> dict:
        r"""
        Test the model, compute metrics and plot images.

        :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] test_dataloader: Test data loader(s), see :ref:`datasets user guide <datasets>`
            for how we expect data to be provided.
        :param bool compare_no_learning: If ``True``, the linear reconstruction is compared to the network reconstruction.
        :param bool log_raw_metrics: if `True`, also return non-aggregated metrics as a list.
        :returns: dict of metrics results with means and stds.
        """
        self.compare_no_learning = compare_no_learning
        self.setup_run()

        self.reset_metrics()

        if not isinstance(test_dataloader, list):
            self.test_dataloader = [test_dataloader]
        else:
            self.test_dataloader = test_dataloader
        self.G = len(self.test_dataloader)

        for loader in self.test_dataloader:
            check_dataset(loader.dataset)

        self.current_val_iterators = [iter(loader) for loader in self.test_dataloader]

        batches = min(
            [len(loader) - loader.drop_last for loader in self.test_dataloader]
        )

        self.model.eval()
        for i in (
            progress_bar := tqdm(
                range(batches),
                ncols=150,
                disable=(not self.show_progress_bar),
            )
        ):
            progress_bar.set_description(f"Test")
            self.step(
                0,
                progress_bar,
                train=False,
                last_batch=(i == batches - 1),
                update_progress_bar=(i % self.freq_update_progress_bar == 0),
            )

        self.train_logger.info("Test results:")

        out = {}
        for k, l in enumerate(self.meters_metrics_val):
            if compare_no_learning:
                name = self.metrics[k].__class__.__name__ + " no learning"
                out[name] = self.meters_metrics_no_learning[k].avg
                out[name + "_std"] = self.meters_metrics_no_learning[k].std
                if log_raw_metrics:
                    out[name + "_vals"] = self.meters_metrics_no_learning[k].vals
                self.train_logger.info(
                    f"{name}: {self.meters_metrics_no_learning[k].avg:.3f} +- {self.meters_metrics_no_learning[k].std:.3f}"
                )

            name = self.metrics[k].__class__.__name__
            out[name] = l.avg
            out[name + "_std"] = l.std
            if log_raw_metrics:
                out[name + "_vals"] = l.vals
            self.train_logger.info(f"{name}: {l.avg:.3f} +- {l.std:.3f}")

        return out


def train(
    model: torch.nn.Module,
    physics: Physics,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int = 100,
    losses: Union[Loss, list[Loss], None] = None,
    val_dataloader: torch.utils.data.DataLoader = None,
    *args,
    **kwargs,
):
    """
    Alias function for training a model using :class:`deepinv.Trainer` class.

    This function creates a Trainer instance and returns the trained model.

    .. warning::

        This function is deprecated and will be removed in future versions. Please use
        :class:`deepinv.Trainer` instead.

    :param deepinv.models.Reconstructor, torch.nn.Module model: Reconstruction network, which can be :ref:`any reconstruction network <reconstructors>`.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s) used by the reconstruction network.
    :param int epochs: Number of training epochs. Default is 100.
    :param torch.optim.Optimizer optimizer: Torch optimizer for training the network.
    :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s), see :ref:`datasets user guide <datasets>`
        for how we expect data to be provided.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        :ref:`See the libraries' training losses <loss>`.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] val_dataloader: Evaluation data loader(s), see :ref:`datasets user guide <datasets>`
        for how we expect data to be provided.
    :param args: Other positional arguments to pass to Trainer constructor. See :class:`deepinv.Trainer`.
    :param kwargs: Keyword arguments to pass to Trainer constructor. See :class:`deepinv.Trainer`.
    :return: Trained model.
    """
    if losses is None:
        losses = SupLoss()
    trainer = Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        epochs=epochs,
        losses=losses,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        *args,
        **kwargs,
    )
    trained_model = trainer.train()
    return trained_model
