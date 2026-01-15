from __future__ import annotations
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
import inspect
import warnings

from tqdm import tqdm
import numpy as np
import torch

from deepinv.datasets.base import check_dataset
from deepinv.loss import Loss, SupLoss, BaseLossScheduler
from deepinv.loss.metric import PSNR, Metric
from deepinv.models import Reconstructor
from deepinv.physics import Physics
from deepinv.physics.generator import PhysicsGenerator
from deepinv.training.run_logger import RunLogger, LocalLogger
from deepinv.utils import AverageMeter
from deepinv.utils.compat import zip_strict
from deepinv.utils.plotting import prepare_images
from deepinv.utils.tensorlist import TensorList


@dataclass
class Trainer:
    r"""Trainer(model, physics, optimizer, train_dataloader, ...)
    Trainer class for training and testing a reconstruction network on inverse image problems.

    .. seealso::

        See the :ref:`User Guide <trainer>` for more details and for how to adapt the trainer to your needs.

        See :ref:`sphx_glr_auto_examples_models_demo_training.py` for a simple example of how to use the trainer.

    Training can be done by calling the :func:`deepinv.Trainer.train` method, whereas
    testing can be done by calling the :func:`deepinv.Trainer.test` method.

    The trainer provides an interface for:
        - both **offline** (pre-computed measurements) and **online** (on-the-fly generated measurements) training setup
        - training simultaneously multiple pairs of (dataset, physics operator)
        - generating on-the-fly appropriate physics' parameters at each training step
        - performing gradient clipping
        - tracking losses and metrics
        - saving/loading checkpoints
        - a seamless integration of both local (writing logs in files) and remote logging tools (sending to remote server like `Weights & Biases <https://wandb.ai/site>`_) via :class:`deepinv.training.run_logger.RunLogger`.

    |sep|

    .. rubric:: Model

    :param deepinv.models.Reconstructor, torch.nn.Module model: Reconstruction network, which can be :ref:`any reconstruction network <reconstructors>`.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: :ref:`Forward operator(s) <physics_list>`.
    :param bool iterative_model_returns_different_outputs: If ``True``, indicate that the model returns an additional output.
    :param str, torch.device device: Device on which to run the training (e.g., 'cuda' or 'cpu'). Default is 'cuda' if available, otherwise 'cpu'.

    |sep|

    .. rubric:: Data and Measurements

    The **dataloaders** should return data in the correct format for DeepInverse: see :ref:`datasets user guide <datasets>` for
    how to use predefined datasets, create datasets, or generate datasets. These will be checked automatically with :func:`deepinv.datasets.check_dataset`.

    - If ``online_measurements=False`` (default), dataloaders must return tuples ``(x, y)`` or ``(x, y, params)``.
    - If ``online_measurements=True``, dataloaders may return only ``x`` or ``(x, params)``, and measurements will be generated online via ``y = physics(x, **params)``.

    For random physics or noise models that vary across iterations, a :class:`deepinv.physics.generator.PhysicsGenerator` can be used to produce different operators each step.

    .. tip::

        If your dataloaders do not return `y` but you do not want online measurements, use :func:`deepinv.datasets.generate_dataset` to generate a dataset
        of offline measurements from a dataset of `x` and a `physics`.

    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s), see :ref:`datasets user guide <datasets>` for how we expect data to be provided.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] val_dataloader: Validation data loader(s) used for early stopping and performance tracking.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] test_dataloader: Test data loader(s) for final evaluation after complete training.
    :param bool online_measurements: If ``True``, measurements ``y`` are generated online as ``physics(x)``, else they are provided by the dataset.
    :param None, deepinv.physics.generator.PhysicsGenerator physics_generator: Optional :ref:`physics generator <physics_generators>` for generating parameters for the physics operators.
    :param bool reproduce_same_online_measurements: If ``True``, resets the physics **and** noise models back to its initial state at the beginning of each epoch, so that the same measurements are generated each epoch. Requires ``shuffle=False`` in dataloaders.

        .. tip::

            Alternatively, a safer solution is to generate and save params offline using :func:`deepinv.datasets.generate_dataset`.
            The physics and noise models will then be automatically updated every time data is loaded.

    |sep|

    .. rubric:: Optimization

    :param None, torch.optim.Optimizer optimizer: Torch optimizer for training the network.
    :param None, torch.optim.lr_scheduler.LRScheduler scheduler: Torch scheduler for changing the learning rate across iterations.
    :param None, float grad_clip: Gradient clipping value for the optimizer. If None, no gradient clipping is performed.
    :param bool optimizer_step_multi_dataset: If ``True``, the optimizer step is performed after seeing one batch from each dataset. If ``False``, the optimizer step is performed on each dataset separately.
    :param int epochs: Max number of training epochs.
    :param int max_batch_steps: Max number of training batches the model sees. Default is `1e10`.
    :param bool early_stop: If not ``None``, the training stops when the first evaluation metric is not improving
        after `early_stop` passes over the eval dataset. Default is ``None`` (no early stopping).
        The user can modify the strategy for saving the best model by overriding the :func:`deepinv.Trainer.stop_criterion` method.

    |sep|

    .. rubric:: Losses and Metrics

    Any loss can be used if it takes as input ``(x, x_net, y, physics, model)`` and returns a tensor of shape ``(batch_size,)`` (i.e. no reduction),
    where:
        - ``x`` is the ground truth,
        - ``x_net`` is the network reconstruction :math:`\inversef{y}{A}`,
        - ``y`` is the measurement vector,
        - ``physics`` is the forward operator,
        - ``model`` is the reconstruction network.

    Note that not all inputs need to be used by the loss, e.g., self-supervised losses will not make use of ``x``.

    Likewise, metrics should have ``reduction=None``.

    Custom classes must inherit from :class:`deepinv.loss.Loss` or :class:`deepinv.loss.metric.Metric`.

    .. note::

        The losses can be chosen from :ref:`our training losses <loss>`.

        The evaluation metrics can be chosen from :ref:`our metrics <metric>`:

        - **Supervised evaluation**: If ground-truth data is available for validation, use any
          :ref:`full reference metric <full-reference-metrics>`, e.g. :class:`PSNR <deepinv.loss.metric.PSNR>`.

        - **Self-supervised evaluation**: If no ground-truth data is available for validation, it is
          still possible to validate using:

            #. :ref:`no reference metrics <no-reference-metrics>`, e.g. :class:`NIQE <deepinv.loss.metric.NIQE>`
            #. :ref:`self-supervised losses <self-supervised-losses>`.

    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        Optionally wrap losses using a loss scheduler for more advanced training.
        :ref:`See the libraries' training losses <loss>`.
        Where relevant, the underlying metric should have ``reduction=None`` as we perform the averaging
        using :class:`deepinv.utils.AverageMeter` to deal with uneven batch sizes. Default is :class:`supervised loss <deepinv.loss.SupLoss>`.
    :param list[str] compute_losses_on: Whether to compute losses on "train", "val", and "test".

        .. warning::

            Training will be aborted if "train" is not included in compute_losses_on.
    :param Metric, list[Metric], None metrics: Metric or list of metrics used for evaluating the model.
        They should have ``reduction=None`` as we perform the averaging using :class:`deepinv.utils.AverageMeter` to deal with uneven batch sizes.
        :ref:`See the libraries' evaluation metrics <metric>`. Default is :class:`PSNR <deepinv.loss.metric.PSNR>`.

        .. warning::

            The metrics from train_dataloader are computed using the model prediction in `model.train()` mode to avoid an additional
            forward pass. This can lead to metrics that are different from val_dataloader and test_dataloader when the model is in `model.eval()` mode,
            and/or produce errors if the network does not provide the same output shapes under train and eval modes (e.g., which is the case of :class:`some self-supervised losses <deepinv.loss.ReducedResolutionLoss>`).
    :param list[str] compute_metrics_on: Whether to compute metrics on "train", "val", and "test".

        .. tip::

            If `val_dataloader` is provided, the trainer will also **save the best model**
            according to the first metric in the list, using the following format: ``self.ckpt_dir/ckpt_best.pth.tar``.
            The user can modify the strategy for saving the best model by overriding the :func:`deepinv.Trainer.save_best_model` method.
            The best model can be also loaded using the :func:`deepinv.Trainer.load_best_model` method.
    :param str no_learning_method: Reconstruction method used for the no learning comparison. Options are ``'A_dagger'``, ``'A_adjoint'``, ``'prox_l2'``, or ``'y'``. Default is ``'A_dagger'``. The user can also provide a custom method by overriding the :func:`no_learning_inference <deepinv.Trainer.no_learning_inference>` method. Default is ``'A_adjoint'``.
    :param list[str] compute_comparison_on: Whether to compare the no_learning recoonstruction on "train", "val" and "test".

    |sep|

    .. rubric:: Checkpointing and Logging

    During training, :class:`RunLogger` instances are in charge of logging losses, metrics and images.
    Losses can be logged at every step or at every epoch.

    Training state is saved automatically every ``ckpt_interval`` epochs, and can be resumed with ``ckpt_pretrained``.

    Each checkpoint contains at least::

        {
            "epoch": current epoch,
            "state_dict": model weights,
            "optimizer": optimizer state dict (if defined),
            "scheduler": scheduler state dict (if defined),
            "train_loss": training loss history per epoch,
            "val_metrics": validation metrics history per epoch
        }

    In addition, :class:`RunLogger` instances (e.g. :class:`LocalLogger`, :class:`WandbLogger`, etc.) will add its own metadata.

    :param str ckpt_dir: folder path where to save checkpoints.
    :param None, str ckpt_pretrained: path of a pretrained checkpoint. If `None` (default), no pretrained checkpoint is loaded.
    :param int ckpt_interval: The model is saved every ``ckpt_interval`` epochs. Default is ``1``.
    :param None, RunLogger, list[RunLogger] loggers: Logging backends (e.g. LocalLogger, WandbLogger, MLflowLogger, etc.).
    :param bool log_images: Log the last batch reconstructions for each epoch. Default is ``False``.
    :param str rescale_mode: Rescale mode for plotting images. Default is ``'clip'``.
    :param bool log_grad: Whether to log the gradient norm at each optimization step. Default is ``False``.
    :param bool show_progress_bar: Display progress bar using tqdm.
    :param bool verbose: Verbosity flag for console and loggers.

    |sep|

    Launch a basic training ::

        from deepinv.training import Trainer, LocalLogger

        trainer = Trainer(
            model=my_model,
            physics=my_physics,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=torch.optim.Adam(my_model.parameters(), lr=1e-3),
            losses=[SupLoss()],
            metrics=[PSNR()],
            loggers=[LocalLogger(log_dir="./logs")],
        )
        trainer.train()

    """

    ## Model
    model: Reconstructor | torch.nn.Module
    physics: Physics | list[Physics]
    iterative_model_returns_different_outputs: bool = False
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Data Loading
    train_dataloader: (
        torch.utils.data.DataLoader | list[torch.utils.data.DataLoader] | None
    ) = None
    val_dataloader: (
        torch.utils.data.DataLoader | list[torch.utils.data.DataLoader] | None
    ) = None
    test_dataloader: (
        torch.utils.data.DataLoader | list[torch.utils.data.DataLoader] | None
    ) = None

    ## Generate measurements for training purpose with `physics`
    online_measurements: bool = False
    physics_generator: PhysicsGenerator | list[PhysicsGenerator] | None = None
    reproduce_same_online_measurements: bool = False

    ## Training Control
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    grad_clip: float | None = None
    optimizer_step_multi_dataset: bool = True

    ## Training Duration & Stopping
    epochs: int = 100
    max_batch_steps: int = 10**10
    early_stop: bool = False

    ## Loss & Metrics
    losses: Loss | BaseLossScheduler | list[Loss] | list[BaseLossScheduler] = SupLoss()
    compute_losses_on: list[str] = field(default_factory=lambda: ["train", "val"])
    metrics: Metric | list[Metric] = field(default_factory=PSNR)
    compute_metrics_on: list[str] = field(default_factory=lambda: ["val", "test"])
    no_learning_method: str = "A_adjoint"
    compute_comparison_on: list[str] = field(default_factory=lambda: ["val", "test"])

    ## Checkpointing & Persistence (at the end of epoch)
    ckpt_dir: str | Path = Path("./checkpoints")
    ckpt_pretrained: str | Path | None = None
    ckpt_interval: int = 1

    ## Logging & Monitoring
    loggers: RunLogger | list[RunLogger] | None = field(
        default_factory=lambda: [LocalLogger(Path("./logs"))]
    )
    log_images: bool = False
    rescale_mode: str = "clip"
    log_grad: bool = False
    show_progress_bar: bool = True
    verbose: bool = True

    def setup_run(self) -> None:
        r"""
        Set up the training/testing.
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
        for loader in (
            self.train_dataloader + self.val_dataloader + self.test_dataloader
        ):
            check_dataset(loader.dataset)

        # ensure that physics is a list for format consistency
        if not isinstance(self.physics, list):
            self.physics = [self.physics]

        # online measurements setting
        if self.physics_generator is not None:
            if not self.online_measurements:
                warnings.warn(
                    "Since `online_measurement` is False, `physics` will not be used to generate degraded images."
                )
            elif self.reproduce_same_online_measurements:
                warnings.warn(
                    "Generated measurements repeat each epoch."
                    "Ensure that dataloader is not shuffling."
                )

            # ensure that physics_generator is a list for format consistency
            if not isinstance(self.physics_generator, list):
                self.physics_generator = [self.physics_generator]

    def _setup_logging(self) -> None:
        r"""
        Set up the monitoring before running an experience.
        """
        # LOSSES processing
        if self.losses is None:
            self.losses = []
        if not isinstance(self.losses, list):
            self.losses = [self.losses]
        for l in self.losses:
            self.model = l.adapt_model(self.model)

        # METRICS processing
        if self.metrics is None:
            self.metrics = []
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        # LOSSES computed during an epoch
        self.meters_losses = {
            phase: {"Total Loss": AverageMeter(f"{phase} total loss", ":.2e")}
            | {
                l.__class__.__name__: AverageMeter(
                    f"{phase} {l.__class__.__name__} loss", ":.2e"
                )
                for l in self.losses
            }
            for phase in self.compute_losses_on
        }  # 1 AverageMeter per loss term and 1 for the total loss

        if not hasattr(self, "train_loss_history_per_epoch"):
            self.train_loss_history_per_epoch = []

        # METRICS computed during an epoch
        self.meters_metrics = {
            phase: {
                m.__class__.__name__: AverageMeter(
                    f"{phase} {m.__class__.__name__} metric ", ":.2e"
                )
                for m in self.metrics
            }
            for phase in self.compute_metrics_on
        }

        if not hasattr(self, "val_metrics_history_per_epoch"):
            self.val_metrics_history_per_epoch = {
                m.__class__.__name__: [] for m in self.metrics
            }

        self.meters_metrics_no_learning = {
            phase: {
                m.__class__.__name__: AverageMeter(
                    f"{phase} {m.__class__.__name__} metric ", ":.2e"
                )
                for m in self.metrics
            }
            for phase in self.compute_comparison_on
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
            logger.init_logger()

        # Set verbosity level of the logger
        if self.verbose:
            self.train_logger.setLevel("DEBUG")
            for logger in self.loggers:
                logger.setLevel("DEBUG")
        else:
            self.train_logger.setLevel("WARNING")
            for logger in self.loggers:
                logger.setLevel("WARNING")

    def save_ckpt(self, epoch: int, name: str | None = None) -> None:
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
            "train_loss": self.train_loss_history_per_epoch,
            "val_metrics": self.val_metrics_history_per_epoch,
        }

        # Enrich checkpoint with logger-specific metadata
        for logger in self.loggers:
            state = logger.prepare_checkpoint(checkpoint_dict=state)

        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        if name is not None:
            ckpt_path = Path(self.ckpt_dir) / f"{name}.pth.tar"
        else:
            ckpt_path = Path(self.ckpt_dir) / f"checkpoint_epoch_{epoch}.pth.tar"

        torch.save(
            state,
            ckpt_path,
        )

        self.train_logger.info(
            f"Checkpoint of epoch {epoch + 1} saved at: {ckpt_path}."
        )

    def load_ckpt(
        self,
        ckpt_pretrained: str | None = None,
    ) -> None:
        """Load a checkpoint for resuming training or for testing.

        This method loads model weights and relevant state information from a checkpoint file.
        If called during training, it resumes from the saved epoch, optimizer, and scheduler states.
        If called for evaluation/testing, it loads the model weights while optimizer and scheduler may be ignored.

        :param str ckpt_pretrained: Path to the checkpoint file.
        """
        if ckpt_pretrained is not None:
            self.ckpt_pretrained = ckpt_pretrained

        # Early return if no checkpoint to load
        if self.ckpt_pretrained is None:
            return

        # Load checkpoint from file
        checkpoint = torch.load(
            self.ckpt_pretrained, map_location=self.device, weights_only=False
        )

        self.epoch_start = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.train_loss_history_per_epoch = checkpoint["train_loss"]
        self.val_metrics_history_per_epoch = checkpoint["val_metrics"]

        for logger in self.loggers:
            logger.load_from_checkpoint(checkpoint_dict=checkpoint)

        self.train_logger.info(
            f"Checkpoint of epoch {self.epoch_start} loaded from: {ckpt_pretrained}."
        )

    def get_samples_online(
        self, iterators: list, g: int
    ) -> tuple[torch.Tensor, torch.Tensor, Physics]:
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

        x = x.to(self.device, non_blocking=self.non_blocking_transfers)
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

    def get_samples_offline(
        self, iterators: list, g: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, Physics]:
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
            x = x.to(self.device, non_blocking=self.non_blocking_transfers)

        y = y.to(self.device, non_blocking=self.non_blocking_transfers)
        physics = self.physics[g]

        if params is not None:
            params = {
                k: (
                    p.to(self.device, non_blocking=self.non_blocking_transfers)
                    if isinstance(p, torch.Tensor)
                    else p
                )
                for k, p in params.items()
            }
            physics.update(**params)

        return x, y, physics

    def get_samples(
        self, iterators: list, g: int
    ) -> tuple[torch.Tensor | None, torch.Tensor, Physics]:
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

    def no_learning_inference(self, y: torch.Tensor, physics: Physics) -> torch.Tensor:
        r"""
        Perform the no learning inference.

        By default it returns the (linear) pseudo-inverse reconstruction given the measurement.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :returns: Reconstructed image.
        """
        y = y.to(self.device)

        with torch.no_grad():
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

    def model_inference(
        self,
        y: torch.Tensor,
        physics: Physics,
        x: torch.Tensor | None = None,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Perform the model inference.

        It returns the network reconstruction given the samples.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param None, torch.Tensor x: Optional ground truth, used for computing convergence metrics.
        :param bool train : If "True", activations are stored in memory to be used later during gradient computation.
        :returns torch.Tensor: The network reconstruction.
        """
        y = y.to(self.device)

        # check if the forward has 'update_parameters' method, and if so, update the parameters
        kwargs = {}
        if "update_parameters" in inspect.signature(self.model.forward).parameters:
            kwargs["update_parameters"] = True

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            if self.iterative_model_returns_different_outputs:
                x_net, _ = self.model(
                    y, physics, x_gt=x, compute_metrics=True, **kwargs
                )
            else:
                x_net = self.model(y, physics, **kwargs)

        return x_net

    def compute_loss(
        self,
        x: torch.Tensor,
        x_net: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        meters: dict[str, AverageMeter],
        train: bool = True,
    ) -> torch.Tensor:
        r"""
        Compute the loss on a batch of data.

        The total loss may be defined as the sum of multiple independent loss functions,
        and we keep track of these components individually.

        The total loss on a batch is computed as:

        .. math::

            \mathcal{L}_{\text{total}}
            = \sum_{k=1}^{K} \mathcal{L}_k(x, x_{\text{net}}, y, \text{physics}, \text{model})

        where :math:`K` is the number of loss functions, :math:`x` is the
        ground truth, :math:`x_{\text{net}}` is the network reconstruction,
        :math:`y` is the measurement, :math:`\text{physics}` is the current
        physics operator, and :math:`\text{model}` is the network used to compute
        the reconstruction.

        Each individual loss term :math:`\mathcal{L}_k` is computed on the batch as:

        .. math::

            \mathcal{L}_k
            = \frac{1}{N} \sum_{i=1}^{N} \ell_k(x^{(i)}, x_{\text{net}}^{(i)}, y^{(i)}, \text{physics}, \text{model})

        where :math:`\ell_k` denotes the loss function applied to each sample of
        the batch of size :math:`N`.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param dict[str, deepinv.utils.AverageMeter meters]: Store the loss computed on a batch to later compute the epoch loss.
        :param bool train : If "True", activations are stored in memory to be used later during gradient computation.
        :returns torch.Tensor: The total loss evaluated on the current batch.
        """
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
                )
                loss_total += (
                    loss.mean()
                )  # accumulate the average of the current loss function evaluated on our batch of data

                meter = meters[l.__class__.__name__]
                meter.update(
                    loss.detach().cpu().numpy()
                )  # track the current loss per img

            meter = meters["Total Loss"]
            meter.update(loss_total.item())

        return loss_total

    def compute_metrics(
        self,
        x: torch.Tensor,
        x_net: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        meters: dict[str, AverageMeter],
    ) -> None:
        r"""
        Compute the evaluation metrics on a batch of data.

        The trainer may use several independent metric functions, and **these
        metrics are not summed or combined**.
        Each metric is computed, tracked, and reported individually.

        Each metric :math:`\mathcal{M}_k` is computed as the average over the batch:

        .. math::

            \mathcal{M}_k
            = \frac{1}{N} \sum_{i=1}^{N} m_k(x^{(i)}, x_{\text{net}}^{(i)}, y^{(i)}, \text{physics}, \text{model})

        where :math:`m_k` denotes the metric function applied to each sample of the batch of size :math:`N`.

        We can optionally compare our model reconstruction to a baseline linear reconstruction.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param dict[str, deepinv.utils.AverageMeter meters]: Store the loss computed on a batch to later compute the epoch loss.
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
                meter = meters[m.__class__.__name__]
                meter.update(
                    metric.detach().cpu().numpy()
                )  # track the current metric per img (img are from several batches)

    def _process_gradients(self) -> None:
        """Clip gradients and optionally compute the gradient norm."""
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.log_grad:
            grads = [
                param.grad.detach().flatten()
                for param in self.model.parameters()
                if param.grad is not None
            ]
            return torch.cat(grads).norm().item()
        return None

    def step(
        self,
        epoch: int,
        progress_bar: tqdm,
        train_ite: int = None,
        phase: str = "train",
        last_batch: bool = False,
    ) -> None:
        r"""
        Run one iteration of **training**, **validation**, or **testing** depending on the selected ``phase``.

        This method:
            - performs a forward pass,
            - optionally computes losses,
            - optionally performs backpropagation and an optimizer update,
            - optionally computes gradient norm and perform gradient clipping
            - optionally computes metrics,
            - optionally computes metrics on a linear baseline for comparison
            - optionally logs images.

        ---

        **Phase behavior**

        phase="train"
            - Training mode enabled and gradients tracked.
            - Loss is computed only if `"train"` is in `compute_losses_on`.
            - Backpropagation is performed for every dataset batch.
            - Optimizer updates:
                - `optimizer_step_multi_dataset=False` → one optimizer step after seeing one batch from any dataset.
                - `optimizer_step_multi_dataset=True`  → one optimizer step after seeing one batch per dataset.
            - Metrics and comparison metrics are computed only if enabled in `compute_metrics_on` / `compute_comparison_on`.
            - Logging of losses, gradient norms, and images (last batch only).

        phase="val"
            - Evaluation mode and no gradient tracking.
            - Loss is computed only if `"val"` is in `compute_losses_on`.
            - **No backward** and **no optimizer step**.
            - Metrics and comparison metrics are computed only if enabled in `compute_metrics_on` / `compute_comparison_on`.
            - Logging of images (last batch only).

        phase="test"
            - Evaluation mode (no gradients).
            - Loss is computed only if `"test"` is in `compute_losses_on`.
            - **No backward** and **no optimizer step**.
            - Metrics and comparison metrics are computed only if enabled in `compute_metrics_on` / `compute_comparison_on`.
            - Logging of images (last batch only).

        :param int epoch: Current epoch.
        :param progress_bar: `tqdm <https://tqdm.github.io/docs/tqdm/>`_ progress bar.
        :param int train_ite: train iteration, only needed when logging training losses each step.
        :param str phase: Training phase ('train', 'val', 'test').
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        """
        # Zero grad
        if phase == "train" and self.optimizer_step_multi_dataset:
            self.optimizer.zero_grad()  # Clear stored gradients

        loss_multi_dataset_step = 0.0
        for g in np.random.permutation(self.G):  # g is the dataloader index

            # Zero grad
            if phase == "train" and not self.optimizer_step_multi_dataset:
                self.optimizer.zero_grad()

            # Get either online or offline samples
            x, y, physics_cur = self.get_samples(self.current_iterators, g)

            # Evaluate reconstruction network
            x_net = self.model_inference(
                y=y, physics=physics_cur, x=x, train=(phase == "train")
            )

            # Compute the loss for the batch
            if phase in self.compute_losses_on:
                loss_cur = self.compute_loss(
                    x,
                    x_net,
                    y,
                    physics_cur,
                    meters=self.meters_losses[phase],
                    train=(phase == "train"),
                )
                loss_multi_dataset_step += loss_cur.item()

                # Backward
                if phase == "train":
                    loss_cur.backward()

                # Process gradients + Optimizer
                if phase == "train" and not self.optimizer_step_multi_dataset:
                    loss_logs = {}
                    loss_logs["Loss"] = loss_cur.item()

                    grad_norm_on_cpu = self._process_gradients()
                    if grad_norm_on_cpu is not None:
                        loss_logs["gradient_norm"] = grad_norm_on_cpu

                    # Optimizer step
                    self.optimizer.step()

                    # Update the progress bar
                    progress_bar.set_postfix(loss_logs)

            # Compute the metrics for the batch
            if phase in self.compute_metrics_on:
                x_net = (
                    x_net.detach()
                )  # detach the network output for metrics and plotting
                self.compute_metrics(
                    x, x_net, y, physics_cur, meters=self.meters_metrics[phase]
                )

            # Compute the metrics for the linear reconstruction of the batch
            if phase in self.compute_comparison_on:
                x_lin = self.no_learning_inference(y, physics_cur)
                self.compute_metrics(
                    x,
                    x_lin,
                    y,
                    physics_cur,
                    meters=self.meters_metrics_no_learning[phase],
                )
            else:
                x_lin = None

            # Log images of last batch for each dataset
            if last_batch and self.log_images:
                self.save_images(
                    epoch,
                    x,
                    y,
                    x_net,
                    x_nl=x_lin,
                    phase=phase,
                )

        # Process gradients + Optimizer
        if phase in self.compute_losses_on:
            if phase == "train" and self.optimizer_step_multi_dataset:
                loss_logs = {}
                loss_logs["Step " + "Total Loss"] = loss_multi_dataset_step

                # Process gradients
                grad_norm_on_cpu = self._process_gradients()
                if grad_norm_on_cpu is not None:
                    loss_logs["Step " + "Gradient Norm"] = grad_norm_on_cpu

                # Optimizer step
                self.optimizer.step()

                # Update the progress bar
                progress_bar.set_postfix(loss_logs)

                # Track loss value used for an update and its gradient norm
                for logger in self.loggers:
                    logger.log_scalars(
                        loss_logs, step=train_ite, phase=phase, kind="loss"
                    )

    def save_images(
        self,
        epoch: int,
        x: torch.Tensor | None,
        y: torch.Tensor,
        x_net: torch.Tensor,
        x_nl: torch.Tensor | None,
        phase: str,
    ) -> None:
        r"""
        Save the reconstructions.

        :param int epoch: Current epoch.
        :param None, torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param None, torch.Tensor x_nl: Optional linear reconstruction for
        :param bool phase: Between "train", "val", and "test".
        """
        imgs, titles, grid_image, caption = prepare_images(
            x, y=y, x_net=x_net, x_nl=x_nl, rescale_mode=self.rescale_mode
        )
        dict_imgs = {t: im for t, im in zip_strict(titles, imgs)}

        for logger in self.loggers:
            logger.log_images(dict_imgs, epoch=epoch, phase=phase)

    def reset_meters(self) -> None:
        r"""
        Reset all the deepinv.utils.AverageMeter that are used to compute epoch losses and metrics.
        """
        for phase, meters in self.meters_losses.items():
            for loss_name, meter in meters.items():
                meter.reset()

        for phase, meters in self.meters_metrics.items():
            for metric_name, meter in meters.items():
                meter.reset()

        for phase, meters in self.meters_metrics_no_learning.items():
            for metric_name, meter in meters.items():
                meter.reset()

    def save_best_model(self, epoch: int) -> None:
        r"""
        Save the best model using validation metrics.

        By default, uses validation based on first metric.

        Override this method to provide custom criterion.

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

    def load_best_model(self) -> Reconstructor | torch.nn.Module:
        r"""
        Load the best model saved in the checkpoints folder.
        """
        if (Path(self.ckpt_dir) / f"ckpt_best.pth.tar").exists():
            self.load_ckpt(Path(self.ckpt_dir) / f"ckpt_best.pth.tar")
        return self.model

    def stop_criterion(self, epoch: int) -> bool:
        r"""
        Stop criterion for early stopping.

        By default, stops optimization when first eval metric doesn't improve in the last 3 evaluations.

        Override this method to early stop on a custom condition.

        :param int epoch: Current epoch.
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

    def log_epoch_losses_and_metrics(self, epoch, phase="train"):
        """
        Log average losses and metrics for the current epoch.

        Computes and logs the epoch-averaged values for all losses and metrics to all configured loggers.
        For multi-loss setups, logs both individual losses and the total loss.

        :param int epoch: Current epoch number.
        :param str phase: Training phase - either "train" or "val". Default is "train".
        """

        ## Losses
        if phase in self.compute_losses_on:
            epoch_loss_logs = {}

            # Add individual losses over an epoch
            if len(self.losses) > 1:
                for l in self.losses:
                    l_name = l.__class__.__name__
                    epoch_loss_logs["Epoch " + l_name] = self.meters_losses[phase][
                        l_name
                    ].avg

            # Add total loss over an epoch
            epoch_loss_logs["Epoch " + "Total Loss"] = self.meters_losses[phase][
                "Total Loss"
            ].avg

        ## Metrics
        if phase in self.compute_metrics_on:
            epoch_metrics_logs = {}
            for m in self.metrics:
                m_name = m.__class__.__name__
                epoch_metrics_logs["Epoch " + m_name] = self.meters_metrics[phase][
                    m_name
                ].avg

        ## Metrics on Linear Reconstruction
        if phase in self.compute_comparison_on:
            epoch_comparison_logs = {}
            for m in self.metrics:
                m_name = m.__class__.__name__
                epoch_comparison_logs["Epoch " + m_name + " no learning"] = (
                    self.meters_metrics_no_learning[phase][m_name].avg
                )

        ## Logging
        for logger in self.loggers:
            if phase in self.compute_losses_on:
                logger.log_scalars(
                    epoch_loss_logs, step=epoch, phase=phase, kind="loss"
                )
            if phase in self.compute_metrics_on:
                logger.log_scalars(
                    epoch_metrics_logs, step=epoch, phase=phase, kind="metric"
                )
            if phase in self.compute_comparison_on:
                logger.log_scalars(
                    epoch_comparison_logs, step=epoch, phase=phase, kind="metric"
                )

        self.train_logger.info(f"Epoch Losses and Metrics are logged.")

    def train(
        self,
    ) -> Reconstructor | torch.nn.Module:
        r"""
        Train the reconstruction model.

        Performs the complete training loop including forward passes, loss computation,
        backpropagation, validation, checkpointing, and early stopping.

        For each epoch:

            1. **Training Phase**

            - Iterate through all training batches
            - Compute losses (if "train" in ``compute_losses_on``)
            - Perform backpropagation and optimizer step
            - Compute metrics (if "train" in ``compute_metrics_on``).
            - Log step training losses
            - Log epoch training losses and metrics

            2. **Validation Phase** (if ``val_dataloader`` is provided)

            - Iterate through all validation batches
            - Compute losses (if "val" in ``compute_losses_on``)
            - Compute metrics (if "val" in ``compute_metrics_on``)
            - Log validation losses and metrics

            3. **Checkpointing and Early Stopping**

            - Update training loss history
            - Update validation metrics history
            - Save best model based on validation metrics
            - Save regular checkpoints every ``ckpt_interval`` epochs
            - Break if stopping conditions

        **Stopping Conditions:**

        Training stops when any of the following conditions are met:
            - The maximum number of epochs (`self.epochs`) is reached
            - The maximum number of batch steps (`self.max_batch_steps`) is exceeded
            - Early stopping is triggered (if `self.early_stop=True`)

        :returns: The trained model.
        """
        self.setup_run()

        # useful when training on multiple dataset
        self.G = len(self.train_dataloader)

        # count the overall training parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.train_logger.info(f"The model has {params} trainable parameters.")

        stop_flag = False
        for epoch in range(self.epoch_start, self.epochs):
            self.reset_meters()  # reset epoch losses and metrics

            ## Training
            self.current_iterators = [iter(loader) for loader in self.train_dataloader]
            batches = min(
                [len(loader) - loader.drop_last for loader in self.train_dataloader]
            )

            if self.reproduce_same_online_measurements:
                if self.physics_generator is not None:
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
                    phase="train",
                    last_batch=(i == batches - 1),
                )

                if train_ite + 1 > self.max_batch_steps:
                    stop_flag = True

            if self.scheduler:  # update the scheduler after an epoch
                self.scheduler.step()

            self.log_epoch_losses_and_metrics(epoch=epoch, phase="train")

            ## Validation
            if self.val_dataloader:
                self.current_iterators = [
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
                        f"Val epoch {epoch + 1}/{self.epochs}"
                    )
                    self.step(
                        epoch,
                        val_progress_bar,
                        train_ite=train_ite,
                        phase="val",
                        last_batch=(j == val_batches - 1),
                    )

                self.log_epoch_losses_and_metrics(epoch=epoch, phase="val")

            ## Early stopping & checkpointing
            if "train" in self.compute_losses_on:
                self.train_loss_history_per_epoch.append(
                    self.meters_losses["train"]["Total Loss"].avg
                )
            if self.val_dataloader and "val" in self.compute_metrics_on:
                for m in self.metrics:
                    self.val_metrics_history_per_epoch[m.__class__.__name__].append(
                        self.meters_metrics["val"][m.__class__.__name__].avg
                    )  # store in metrics history

                self.save_best_model(epoch)

            if (epoch % self.ckpt_interval == 0) or epoch + 1 == self.epochs:
                self.save_ckpt(epoch=epoch)

            if self.early_stop:
                stop_flag = self.stop_criterion(epoch)
            if stop_flag:
                break

        for logger in self.loggers:
            logger.finish_run()

        return self.model

    def test(
        self,
        test_dataloader: (
            torch.utils.data.DataLoader | list[torch.utils.data.DataLoader] | None
        ) = None,
        metrics: Metric | list[Metric] | None = None,
        compare_no_learning: bool | None = None,
        no_learning_method: str | None = None,
        loggers: RunLogger | list[RunLogger] | None = None,
        log_raw_metrics: bool = False,
    ) -> dict:
        r"""
        Test the model by computing metrics and logging images.

        :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] test_dataloader: Test data loader(s), see :ref:`datasets user guide <datasets>` for how we expect data to be provided.
        :param Metric, list[Metric], None metrics: Metric or list of metrics used for evaluation. If `None`, uses the metrics provided during Trainer initialization.
        :param bool compare_no_learning: If ``True``, the linear reconstruction is compared to the network reconstruction.
        :param None, str no_learning_method: Reconstruction method used for the no learning comparison. Options are ``'A_dagger'``, ``'A_adjoint'``, ``'prox_l2'``, or ``'y'``.
        :param None, RunLogger, list[RunLogger] loggers: Logging backends (e.g. LocalLogger, WandbLogger, MLflowLogger, etc.).
        :param bool log_raw_metrics: if `True`, also return non-aggregated metrics as a list.
        :returns: dict of metrics results with means and stds.
        """
        # Setup
        if test_dataloader is not None:
            self.test_dataloader = test_dataloader
        if metrics is not None:
            self.metrics = metrics
        if compare_no_learning is not None:
            self.compute_comparison_on += ["test"]
        if no_learning_method is not None:
            self.no_learning_method = no_learning_method
        if loggers is not None:
            self.loggers = loggers

        self.setup_run()

        # useful when testing on multiple dataset
        self.G = len(self.test_dataloader)

        self.current_iterators = [iter(loader) for loader in self.test_dataloader]
        batches = min(
            [len(loader) - loader.drop_last for loader in self.test_dataloader]
        )

        for i in (
            test_progress_bar := tqdm(
                range(batches),
                ncols=150,
                disable=(not self.show_progress_bar),
            )
        ):
            test_progress_bar.set_description(f"Test")
            self.step(0, test_progress_bar, phase="test", last_batch=(i == batches - 1))

        out = {}
        for m in self.metrics:
            if compare_no_learning:
                name = m.__class__.__name__ + " no learning"
                meter = self.meters_metrics_no_learning["test"][m.__class__.__name__]
                out[name] = meter.avg
                out[name + "_std"] = meter.std
                if log_raw_metrics:
                    out[name + "_vals"] = meter.vals

                self.train_logger.info(f"{name}: {meter.avg:.3f} +- {meter.std:.3f}")

            name = m.__class__.__name__
            meter = self.meters_metrics["test"][m.__class__.__name__]
            out[name] = meter.avg
            out[name + "_std"] = meter.std
            if log_raw_metrics:
                out[name + "_vals"] = meter.vals
            self.train_logger.info(f"{name}: {meter.avg:.3f} +- {meter.std:.3f}")

        return out
