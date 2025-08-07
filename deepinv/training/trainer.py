import warnings
from deepinv.utils import AverageMeter, get_timestamp, plot, plot_curves
import os
import numpy as np
from tqdm import tqdm
import torch

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = ImportError(
        "The wandb package is not installed. Please install it with `pip install wandb`."
    )  # pragma: no cover

from pathlib import Path
from typing import Union
from dataclasses import dataclass, field
from deepinv.loss import Loss, SupLoss, BaseLossScheduler
from deepinv.loss.metric import PSNR, Metric
from deepinv.physics import Physics
from deepinv.physics.generator import PhysicsGenerator
from deepinv.utils.plotting import prepare_images
from torchvision.utils import save_image
import inspect


@dataclass
class Trainer:
    r"""Trainer(model, physics, optimizer, train_dataloader, ...)
    Trainer class for training a reconstruction network.

    .. seealso::

        See the :ref:`User Guide <trainer>` for more details and for how to adapt the trainer to your needs.

    Training can be done by calling the :func:`deepinv.Trainer.train` method, whereas
    testing can be done by calling the :func:`deepinv.Trainer.test` method.

    Training details are saved every ``ckp_interval`` epochs in the following format

    ::

        save_path/yyyy-mm-dd_hh-mm-ss/ckp_{epoch}.pth.tar

    where ``.pth.tar`` file contains a dictionary with the keys: ``epoch`` current epoch, ``state_dict`` the state
    dictionary of the model, ``loss`` the loss history, ``optimizer`` the state dictionary of the optimizer,
    and ``eval_metrics`` the evaluation metrics history.

    Assuming that `x` is the ground-truth reference and `y` is the measurement and `params` is a dict of :ref:`physics parameters <physics_generators>`,
    the **dataloaders** should return one of the following options:

    1. `(x, y)` or `(x, y, params)`, which requires `online_measurements=False` (default) otherwise `y` will be ignored and new measurements will be generated online.
    2. `(x)` or `(x, params)`, which requires `online_measurements=True` for generating measurements in an online manner (optionally with parameters) as `y=physics(x)` or `y=physics(x, **params)`. Otherwise, first generate a dataset of `(x,y)` with :class:`deepinv.datasets.generate_dataset` and then use option 1 above.
    3. If you have a dataset of measurements only `(y)` or `(y, params)` you should modify it such that it returns `(torch.nan, y)` or `(torch.nan, y, params)`. Set `online_measurements=False`.

    .. note::

        The losses and evaluation metrics can be chosen from :ref:`our training losses <loss>` or :ref:`our metrics <metric>`

        Custom losses can be used, as long as it takes as input ``(x, x_net, y, physics, model)``
        and returns a tensor of length `batch_size` (i.e. `reduction=None` in the underlying metric, as we perform averaging to deal with uneven batch sizes),
        where ``x`` is the ground truth, ``x_net`` is the network reconstruction :math:`\inversef{y}{A}`,
        ``y`` is the measurement vector, ``physics`` is the forward operator
        and ``model`` is the reconstruction network. Note that not all inputs need to be used by the loss,
        e.g., self-supervised losses will not make use of ``x``.

        Custom metrics can also be used in the exact same way as custom losses.

    .. note::

        The training code can synchronize with `Weights & Biases <https://wandb.ai/site>`_ for logging and visualization
        by setting ``wandb_vis=True``. The user can also customize the wandb setup by providing
        a dictionary with the setup for wandb.

    Parameters are described below, grouped into **Basics**, **Optimization**, **Evaluation**, **Physics Generators**,
    **Model Saving**, **Comparing with Pseudoinverse Baseline**, **Plotting**, **Verbose** and **Weights & Biases**.

    :Basics:

    :param deepinv.models.Reconstructor, torch.nn.Module model: Reconstruction network, which can be :ref:`any reconstruction network <reconstructors>`.
        or any other custom reconstruction network.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: :ref:`Forward operator(s) <physics_list>`.
    :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s), see options 1 to 3
        above for how we expect data to be provided.
    :param bool online_measurements: Generate new measurements `y` in an online manner at each iteration by calling
        `y=physics(x)`. If `False` (default), the measurements are loaded from the training dataset.
    :param str, torch.device device: Device on which to run the training (e.g., 'cuda' or 'cpu'). Default is 'cuda' if available, otherwise 'cpu'.

    |sep|

    :Optimization:

    :param None, torch.optim.Optimizer optimizer: Torch optimizer for training the network. Default is ``None``.
    :param int epochs: Number of training epochs.
        Default is 100. The trainer will perform gradient steps equal to the `min(epochs*n_batches, max_batch_steps)`.
    :param int max_batch_steps: Number of gradient steps per iteration.
        Default is `1e10`. The trainer will perform batch steps equal to the `min(epochs*n_batches, max_batch_steps)`.
    :param None, torch.optim.lr_scheduler.LRScheduler scheduler: Torch scheduler for changing the learning rate across iterations. Default is ``None``.
    :param bool early_stop: If ``True``, the training stops when the evaluation metric is not improving. Default is ``False``.
        The user can modify the strategy for saving the best model by overriding the :func:`deepinv.Trainer.stop_criterion` method.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        Optionally wrap losses using a loss scheduler for more advanced training.
        :ref:`See the libraries' training losses <loss>`.
        Where relevant, the underlying metric should have ``reduction=None`` as we perform the averaging
        using :class:`deepinv.utils.AverageMeter` to deal with uneven batch sizes. Default is :class:`supervised loss <deepinv.loss.SupLoss>`.
    :param float grad_clip: Gradient clipping value for the optimizer. If None, no gradient clipping is performed. Default is None.
    :param bool optimizer_step_multi_dataset: If ``True``, the optimizer step is performed once on all datasets. If ``False``, the optimizer step is performed on each dataset separately.

    |sep|

    :Evaluation:

    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] eval_dataloader: Evaluation data loader(s),
        see options 1 to 3 above for how we expect data to be provided.
    :param Metric, list[Metric] metrics: Metric or list of metrics used for evaluating the model.
        They should have ``reduction=None`` as we perform the averaging using :class:`deepinv.utils.AverageMeter` to deal with uneven batch sizes.
        :ref:`See the libraries' evaluation metrics <metric>`. Default is :class:`PSNR <deepinv.loss.metric.PSNR>`.
    :param int eval_interval: Number of epochs (or train iters, if ``log_train_batch=True``) between each evaluation of
        the model on the evaluation set. Default is ``1``.
    :param bool log_train_batch: if ``True``, log train batch and eval-set metrics and losses for each train batch during training.
        This is useful for visualising train progress inside an epoch, not just over epochs.
        If ``False`` (default), log average over dataset per epoch (standard training).

    .. tip::
        If a validation dataloader `eval_dataloader` is provided, the trainer will also **save the best model** according to the
        first metric in the list, using the following format:
        ``save_path/yyyy-mm-dd_hh-mm-ss/ckp_best.pth.tar``. The user can modify the strategy for saving the best model
        by overriding the :func:`deepinv.Trainer.save_best_model` method.
        The best model can be also loaded using the :func:`deepinv.Trainer.load_best_model` method.

    |sep|

    :Physics Generators:

    :param None, deepinv.physics.generator.PhysicsGenerator physics_generator: Optional :ref:`physics generator <physics_generators>` for generating
        the physics operators. If not `None`, the physics operators are randomly sampled at each iteration using the generator.
        Should be used in conjunction with ``online_measurements=True``, no effect when ``online_measurements=False``. Also see ``loop_random_online_physics``. Default is ``None``.
    :param bool loop_random_online_physics: if `True`, resets the physics generator **and** noise model back to its initial state at the beginning of each epoch,
        so that the same measurements are generated each epoch. Requires `shuffle=False` in dataloaders. If `False`, generates new physics every epoch.
        Used in conjunction with ``online_measurements=True`` and `physics_generator` or noise model in `physics`, no effect when ``online_measurements=False``. Default is ``False``.

    .. warning::

        If the physics changes at each iteration for online measurements (e.g. if `physics_generator` is used to generate random physics operators or noise model is used),
        the generated measurements will randomly vary each epoch.
        If this is not desired (i.e. you want the same online measurements each epoch), set ``loop_random_online_physics=True``.
        This resets the physics generator and noise model's random generators every epoch.

        **Caveat**: this requires ``shuffle=False`` in your dataloaders.

        An alternative, safer solution is to generate and save params offline using :func:`deepinv.datasets.generate_dataset`.
        The params dict will then be automatically updated every time data is loaded.

    |sep|

    :Model Saving:

    :param str save_path: Directory in which to save the trained model. Default is ``"."`` (current folder).
    :param int ckp_interval: The model is saved every ``ckp_interval`` epochs. Default is ``1``.
    :param str ckpt_pretrained: path of the pretrained checkpoint. If `None` (default), no pretrained checkpoint is loaded.

    |sep|

    :Comparison with Pseudoinverse Baseline:

    :param bool compare_no_learning: If ``True``, the no learning method is compared to the network reconstruction. Default is ``False``.
    :param str no_learning_method: Reconstruction method used for the no learning comparison. Options are ``'A_dagger'``, ``'A_adjoint'``,
        ``'prox_l2'``, or ``'y'``. Default is ``'A_dagger'``. The user can also provide a custom method by overriding the
        :func:`no_learning_inference <deepinv.Trainer.no_learning_inference>` method. Default is ``'A_dagger'``.

    |sep|

    :Plotting:

    :param bool plot_images: Plots reconstructions every ``ckp_interval`` epochs. Default is ``False``.
    :param bool plot_measurements: Plot the measurements y, default=`True`.
    :param bool plot_convergence_metrics: Plot convergence metrics for model, default=`False`.
    :param str rescale_mode: Rescale mode for plotting images. Default is ``'clip'``.

    |sep|

    :Verbose:

    :param bool verbose: Output training progress information in the console. Default is ``True``.
    :param bool verbose_individual_losses: If ``True``, the value of individual losses are printed during training.
        Otherwise, only the total loss is printed. Default is ``True``.
    :param bool show_progress_bar: Show a progress bar during training. Default is ``True``.
    :param bool check_grad: Compute and print the gradient norm at each iteration. Default is ``False``.
    :param bool display_losses_eval: If ``True``, the losses are displayed during evaluation. Default is ``False``.

    |sep|

    :Weights & Biases:

    :param bool wandb_vis: Logs data onto Weights & Biases, see https://wandb.ai/ for more details. Default is ``False``.
    :param dict wandb_setup: Dictionary with the setup for wandb, see https://docs.wandb.ai/quickstart for more details. Default is ``{}``.
    :param int plot_interval: Frequency of plotting images to wandb during train evaluation (at the end of each epoch).
        If ``1``, plots at each epoch. Default is ``1``.
    :param int freq_plot: deprecated. Use ``plot_interval``
    """

    model: torch.nn.Module
    physics: Union[Physics, list[Physics]]
    optimizer: Union[torch.optim.Optimizer, None]
    train_dataloader: torch.utils.data.DataLoader
    epochs: int = 100
    max_batch_steps: int = 10**10
    losses: Union[Loss, BaseLossScheduler, list[Loss], list[BaseLossScheduler]] = (
        SupLoss()
    )
    eval_dataloader: torch.utils.data.DataLoader = None
    early_stop: bool = False
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    online_measurements: bool = False
    physics_generator: Union[PhysicsGenerator, list[PhysicsGenerator]] = None
    loop_random_online_physics: bool = False
    optimizer_step_multi_dataset: bool = True
    metrics: Union[Metric, list[Metric]] = PSNR()
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_pretrained: Union[str, None] = None
    save_path: Union[str, Path] = "."
    compare_no_learning: bool = False
    no_learning_method: str = "A_adjoint"
    grad_clip: float = None
    check_grad: bool = False
    wandb_vis: bool = False
    wandb_setup: dict = field(default_factory=dict)
    ckp_interval: int = 1
    eval_interval: int = 1
    plot_interval: int = 1
    freq_plot: int = None
    plot_images: bool = False
    plot_measurements: bool = True
    plot_convergence_metrics: bool = False
    rescale_mode: str = "clip"
    display_losses_eval: bool = False
    log_train_batch: bool = False
    verbose: bool = True
    verbose_individual_losses: bool = True
    show_progress_bar: bool = True

    def setup_train(self, train=True, **kwargs):
        r"""
        Set up the training process.

        It initializes the wandb logging, the different metrics, the save path, the physics and dataloaders,
        and the pretrained checkpoint if given.

        :param bool train: whether model is being trained.
        """

        if type(self.train_dataloader) is not list:
            self.train_dataloader = [self.train_dataloader]

        if self.eval_dataloader is not None and type(self.eval_dataloader) is not list:
            self.eval_dataloader = [self.eval_dataloader]

        self.save_path = Path(self.save_path) if self.save_path else None

        self.G = len(self.train_dataloader)

        if self.freq_plot is not None:
            warnings.warn(
                "freq_plot parameter of Trainer is deprecated. Use plot_interval instead."
            )
            self.plot_interval = self.freq_plot

        if (
            self.wandb_setup != {}
            and self.wandb_setup is not None
            and not self.wandb_vis
        ):
            warnings.warn(
                "wandb_vis is False but wandb_setup is provided. Wandb deactivated (wandb_vis=False)."
            )

        if self.physics_generator is not None and not self.online_measurements:
            warnings.warn(
                "Physics generator is provided but online_measurements is False. Physics generator will not be used."
            )
        elif (
            self.physics_generator is not None
            and self.online_measurements
            and self.loop_random_online_physics
        ):
            warnings.warn(
                "Generated measurements repeat each epoch. Ensure that dataloader is not shuffling."
            )

        self.epoch_start = 0

        self.conv_metrics = None
        # wandb initialization
        if self.wandb_vis:
            if wandb.run is None:
                wandb.init(**self.wandb_setup)

        if not isinstance(self.losses, (list, tuple)):
            self.losses = [self.losses]

        for l in self.losses:
            self.model = l.adapt_model(self.model)

        if not isinstance(self.metrics, (list, tuple)):
            self.metrics = [self.metrics]

        # losses
        self.logs_total_loss_train = AverageMeter("Training loss", ":.2e")
        self.logs_losses_train = [
            AverageMeter("Training loss " + l.__class__.__name__, ":.2e")
            for l in self.losses
        ]

        self.logs_total_loss_eval = AverageMeter("Validation loss", ":.2e")
        self.logs_losses_eval = [
            AverageMeter("Validation loss " + l.__class__.__name__, ":.2e")
            for l in self.losses
        ]

        # metrics
        self.logs_metrics_train = [
            AverageMeter("Training metric " + l.__class__.__name__, ":.2e")
            for l in self.metrics
        ]

        self.logs_metrics_eval = [
            AverageMeter("Validation metric " + l.__class__.__name__, ":.2e")
            for l in self.metrics
        ]
        if self.compare_no_learning:
            self.logs_metrics_no_learning = [
                AverageMeter("Validation metric " + l.__class__.__name__, ":.2e")
                for l in self.metrics
            ]

        self.eval_metrics_history = {}
        for l in self.metrics:
            self.eval_metrics_history[l.__class__.__name__] = []

        # gradient clipping
        if train and self.check_grad:
            self.check_grad_val = AverageMeter("Gradient norm", ":.2e")

        if self.save_path:
            # NOTE: Two separate training should not write to the same
            # directory. For this reason, we make sure the directory does not
            # already exist.
            dir_path = f"{self.save_path}/{get_timestamp()}"
            # Acquire the output directory (might fail with an exception)
            os.makedirs(dir_path, exist_ok=False)
            self.save_path = dir_path
        else:
            self.save_path = None

        # count the overall training parameters
        if self.verbose and train:
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"The model has {params} trainable parameters")

        # make physics and data_loaders of list type
        if type(self.physics) is not list:
            self.physics = [self.physics]

        if (
            self.physics_generator is not None
            and type(self.physics_generator) is not list
        ):
            self.physics_generator = [self.physics_generator]

        if train:
            self.loss_history = []
        self.save_folder_im = None

        _ = self.load_model()

    def load_model(
        self, ckpt_pretrained: Union[str, Path] = None, strict: bool = True
    ) -> dict:
        """Load model from checkpoint.

        :param str ckpt_pretrained: checkpoint filename. If `None`, use checkpoint passed to class init.
            If not `None`, override checkpoint passed to class.
        :param bool strict: strict load weights to model.
        :return: if checkpoint loaded, return checkpoint dict, else return ``None``
        """
        if ckpt_pretrained is None and self.ckpt_pretrained is not None:
            ckpt_pretrained = self.ckpt_pretrained
            # Set to None to prevent it being loaded again
            self.ckpt_pretrained = None

        if ckpt_pretrained is not None:
            checkpoint = torch.load(
                ckpt_pretrained, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["state_dict"], strict=strict)
            if "optimizer" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if "wandb_id" in checkpoint and self.wandb_vis:
                self.wandb_setup["id"] = checkpoint["wandb_id"]
                self.wandb_setup["resume"] = "allow"
            if "epoch" in checkpoint:
                self.epoch_start = checkpoint["epoch"] + 1
            return checkpoint

    def log_metrics_wandb(self, logs: dict, step: int, train: bool = True):
        r"""
        Log the metrics to wandb.

        It logs the metrics to wandb.

        :param dict logs: Dictionary containing the metrics to log.
        :param int step: Current step to log. If ``Trainer.log_train_batch=True``, this is the batch iteration, if ``False`` (default), this is the epoch.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        if step is None:
            raise ValueError("wandb logging step must be specified.")

        if not train:
            logs = {"Eval " + str(key): val for key, val in logs.items()}

        if self.wandb_vis:
            wandb.log(logs, step=step)

    def check_clip_grad(self):
        r"""
        Check the gradient norm and perform gradient clipping if necessary.

        """
        out = None

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.check_grad:
            # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/7
            grads = [
                param.grad.detach().flatten()
                for param in self.model.parameters()
                if param.grad is not None
            ]
            norm_grads = torch.cat(grads).norm()
            out = norm_grads.item()
            self.check_grad_val.update(norm_grads.item())

        return out

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
        if (type(data) is not tuple and type(data) is not list) or len(data) < 2:
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

        if type(x) is list or type(x) is tuple:
            x = [s.to(self.device) for s in x]
        else:
            x = x.to(self.device)

        if x.numel() == 1 and torch.isnan(x):
            x = None  # unsupervised case

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
        if self.online_measurements:  # the measurements y are created on-the-fly
            x, y, physics = self.get_samples_online(iterators, g)
        else:  # the measurements y were pre-computed
            x, y, physics = self.get_samples_offline(iterators, g)

        if x is not None:  # If x is None, we are in unsupervised case
            if torch.isinf(x).any() or torch.isnan(x).any():
                warnings.warn("x contains NaN or inf values.")

        return x, y, physics

    def model_inference(self, y, physics, x=None, train=True, **kwargs):
        r"""
        Perform the model inference.

        It returns the network reconstruction given the samples.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Optional ground truth, used for computing convergence metrics.
        :returns: The network reconstruction.
        """
        y = y.to(self.device)

        kwargs = {}

        # check if the forward has 'update_parameters' method, and if so, update the parameters
        if "update_parameters" in inspect.signature(self.model.forward).parameters:
            kwargs["update_parameters"] = True

        if self.plot_convergence_metrics and not train:
            with torch.no_grad():
                x_net, self.conv_metrics = self.model(
                    y, physics, x_gt=x, compute_metrics=True, **kwargs
                )
            x_net, self.conv_metrics = self.model(
                y, physics, x_gt=x, compute_metrics=True, **kwargs
            )
        else:
            x_net = self.model(y, physics, **kwargs)

        return x_net

    def compute_loss(self, physics, x, y, train=True, epoch: int = None, step=False):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :param bool step: Whether to perform an optimization step when computing the loss.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        if train and step:
            self.optimizer.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics, x=x, train=train)

        if train or self.display_losses_eval:
            # Compute the losses
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                )
                loss_total += loss.mean()
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    meters = (
                        self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                    )
                    meters.update(loss.detach().cpu().numpy())
                    logs[l.__class__.__name__] = meters.avg

            meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
            meters.update(loss_total.item())
            logs[f"TotalLoss"] = meters.avg
        else:  # TODO question: what do we want to do at test time?
            loss_total = 0

        if train:
            loss_total.backward()  # Backward the total loss

            norm = self.check_clip_grad()
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            if step:
                self.optimizer.step()  # Optimizer step

        return loss_total, x_net, logs

    def compute_metrics(
        self, x, x_net, y, physics, logs, train=True, epoch: int = None
    ):
        r"""
        Compute the metrics.

        It computes the metrics over the batch.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param dict logs: Dictionary containing the logs for printing the training progress.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: The logs with the metrics.
        """
        # Compute the metrics over the batch
        with torch.no_grad():
            for k, l in enumerate(self.metrics):
                metric = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                )

                current_log = (
                    self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                )
                current_log.update(metric.detach().cpu().numpy())
                logs[l.__class__.__name__] = current_log.avg

                if not train and self.compare_no_learning:
                    x_lin = self.no_learning_inference(y, physics)
                    metric = l(x=x, x_net=x_lin, y=y, physics=physics, model=self.model)
                    self.logs_metrics_no_learning[k].update(
                        metric.detach().cpu().numpy()
                    )
                    logs[f"{l.__class__.__name__} no learning"] = (
                        self.logs_metrics_no_learning[k].avg
                    )
        return logs

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

    def step(
        self,
        epoch,
        progress_bar,
        train_ite=None,
        train=True,
        last_batch=False,
    ):
        r"""
        Train/Eval a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration.

        :param int epoch: Current epoch.
        :param progress_bar: `tqdm <https://tqdm.github.io/docs/tqdm/>`_ progress bar.
        :param int train_ite: train iteration, only needed for logging if ``Trainer.log_train_batch=True``
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        :returns: The current physics operator, the ground truth, the measurement, and the network reconstruction.
        """
        if train and self.optimizer_step_multi_dataset:
            self.optimizer.zero_grad()  # Clear stored gradients

        # random permulation of the dataloaders
        G_perm = np.random.permutation(self.G)
        loss = 0

        if self.log_train_batch and train:
            self.reset_metrics()

        for g in G_perm:  # for each dataloader
            x, y, physics_cur = self.get_samples(
                self.current_train_iterators if train else self.current_eval_iterators,
                g,
            )

            # Compute loss and perform backprop
            loss_cur, x_net, logs = self.compute_loss(
                physics_cur,
                x,
                y,
                train=train,
                epoch=epoch,
                step=(not self.optimizer_step_multi_dataset),
            )
            loss += loss_cur

            # detach the network output for metrics and plotting
            x_net = x_net.detach()

            # Log metrics
            logs = self.compute_metrics(
                x, x_net, y, physics_cur, logs, train=train, epoch=epoch
            )

            # Update the progress bar
            progress_bar.set_postfix(logs)

        if self.log_train_batch and train:
            self.log_metrics_wandb(logs, step=train_ite, train=train)

        if train and self.optimizer_step_multi_dataset:
            self.optimizer.step()  # Optimizer step

        if last_batch:
            if self.verbose and not self.show_progress_bar:
                if self.verbose_individual_losses:
                    print(
                        f"{'Train' if train else 'Eval'} epoch {epoch}:"
                        f" {', '.join([f'{k}={round(v, 3)}' for (k, v) in logs.items()])}"
                    )
                else:
                    print(
                        f"{'Train' if train else 'Eval'} epoch {epoch}: Total loss: {logs['TotalLoss']}"
                    )

            if self.log_train_batch and train:
                logs["step"] = train_ite
            elif train:
                logs["step"] = epoch
                self.log_metrics_wandb(logs, step=epoch, train=train)
            elif self.log_train_batch:  # train=False
                logs["step"] = train_ite
                self.log_metrics_wandb(logs, step=train_ite, train=train)
            else:
                self.log_metrics_wandb(logs, step=epoch, train=train)

            self.plot(
                epoch,
                physics_cur,
                x,
                y,
                x_net,
                train=train,
            )

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot and optinally save the reconstructions.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        post_str = "Training" if train else "Eval"

        plot_images = self.plot_images and ((epoch + 1) % self.plot_interval == 0)
        save_images = self.save_folder_im is not None

        if plot_images or save_images:
            if self.compare_no_learning:
                x_nl = self.no_learning_inference(y, physics)
            else:
                x_nl = None

            imgs, titles, grid_image, caption = prepare_images(
                x, y=y, x_net=x_net, x_nl=x_nl, rescale_mode=self.rescale_mode
            )

        if plot_images:
            plot(
                imgs,
                titles=titles,
                show=self.plot_images,
                return_fig=True,
                rescale_mode=self.rescale_mode,
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch, step=epoch)

        if save_images:
            # save images
            for k, img in enumerate(imgs):
                for i in range(img.size(0)):
                    img_name = f"{self.save_folder_im}/{titles[k]}/"
                    # make dir
                    Path(img_name).mkdir(parents=True, exist_ok=True)
                    save_image(img, img_name + f"{self.img_counter + i}.png")

                self.img_counter += len(imgs[0])

        if self.conv_metrics is not None:
            plot_curves(
                self.conv_metrics,
                save_dir=f"{self.save_folder_im}/convergence_metrics/",
                show=True,
            )
            self.conv_metrics = None

    def save_model(self, filename, epoch, state={}):
        r"""
        Save the model.

        It saves the model every ``ckp_interval`` epochs.

        :param int epoch: Current epoch.
        :param None, float eval_metrics: Evaluation metrics across epochs.
        :param dict state: custom objects to save with model
        """

        if not self.save_path:
            return

        os.makedirs(str(self.save_path), exist_ok=True)
        state = state | {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "loss": self.loss_history,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        state["eval_metrics"] = self.eval_metrics_history
        if self.wandb_vis:
            state["wandb_id"] = wandb.run.id

        torch.save(
            state,
            Path(self.save_path) / Path(filename),
        )

    def reset_metrics(self):
        r"""
        Reset the metrics.
        """
        self.img_counter = 0

        self.logs_total_loss_train.reset()
        self.logs_total_loss_eval.reset()

        for l in self.logs_losses_train:
            l.reset()

        for l in self.logs_losses_eval:
            l.reset()

        for l in self.logs_metrics_train:
            l.reset()

        for l in self.logs_metrics_eval:
            l.reset()

        if hasattr(self, "check_grad_val"):
            self.check_grad_val.reset()

    def save_best_model(self, epoch, train_ite, **kwargs):
        r"""
        Save the best model using validation metrics.

        By default, uses validation based on first metric. Override this method to provide custom criterion.

        :param int epoch: Current epoch.
        :param int train_ite: Current training batch iteration, equal to (current epoch :math:`\times`
            number of batches) + current batch within epoch
        """
        k = 0  # index of the first metric
        history = self.eval_metrics_history[self.metrics[k].__class__.__name__]
        lower_better = getattr(self.metrics[k], "lower_better", True)

        best_metric = min(history) if lower_better else max(history)
        curr_metric = history[-1]
        if (lower_better and curr_metric <= best_metric) or (
            not lower_better and curr_metric >= best_metric
        ):
            # Saving the model
            self.save_model("ckp_best.pth.tar", epoch)
            if self.verbose:
                print(f"Best model saved at epoch {epoch + 1}")

    def load_best_model(self):
        r"""
        Load the best model.

        It loads the model from the checkpoint saved during training.

        :returns: The model.
        """
        if not self.save_path:
            raise ValueError(
                "No save path provided. Please provide a save path to load the best model."
            )
        else:
            self.load_model(
                ckpt_pretrained=Path(self.save_path) / Path("ckp_best.pth.tar")
            )
        return self.model

    def stop_criterion(self, epoch, train_ite, **kwargs):
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

        history = self.eval_metrics_history[self.metrics[k].__class__.__name__]
        lower_better = getattr(self.metrics[k], "lower_better", True)

        best_metric = min(history) if lower_better else max(history)
        best_epoch = history.index(best_metric) * self.eval_interval

        early_stop = epoch > 2 * self.eval_interval + best_epoch
        if early_stop and self.verbose:
            print(
                "Early stopping triggered as validation metrics have not improved in "
                "the last 3 validation steps, disable it with early_stop=False"
            )

        return early_stop

    def train(
        self,
    ):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        :returns: The trained model.
        """

        self.setup_train()
        stop_flag = False
        for epoch in range(self.epoch_start, self.epochs):
            self.reset_metrics()

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
                    ncols=150,
                    disable=(not self.verbose or not self.show_progress_bar),
                )
            ):
                progress_bar.set_description(f"Train epoch {epoch + 1}/{self.epochs}")
                last_batch = i == batches - 1
                train_ite = (epoch * batches) + i
                self.step(
                    epoch,
                    progress_bar,
                    train_ite=train_ite,
                    train=True,
                    last_batch=last_batch,
                )

                perform_eval = self.eval_dataloader and (
                    (
                        (epoch % self.eval_interval == 0 or epoch + 1 == self.epochs)
                        and not self.log_train_batch
                    )
                    or (
                        (i % self.eval_interval == 0 or i + 1 == batches)
                        and self.log_train_batch
                    )
                )
                if perform_eval and (last_batch or self.log_train_batch):
                    ## Evaluation
                    self.current_eval_iterators = [
                        iter(loader) for loader in self.eval_dataloader
                    ]

                    eval_batches = min(
                        [
                            len(loader) - loader.drop_last
                            for loader in self.eval_dataloader
                        ]
                    )

                    self.model.eval()
                    # close train progress bar
                    progress_bar.update(1)
                    progress_bar.close()
                    for j in (
                        eval_progress_bar := tqdm(
                            range(eval_batches),
                            ncols=150,
                            disable=(not self.verbose or not self.show_progress_bar),
                            colour="green",
                        )
                    ):
                        eval_progress_bar.set_description(
                            f"Eval epoch {epoch + 1}/{self.epochs}"
                        )
                        self.step(
                            epoch,
                            eval_progress_bar,
                            train_ite=train_ite,
                            train=False,
                            last_batch=(j == eval_batches - 1),
                        )

                    for k in range(len(self.metrics)):
                        metric = self.logs_metrics_eval[k].avg
                        self.eval_metrics_history[
                            self.metrics[k].__class__.__name__
                        ].append(
                            metric
                        )  # store metrics history

                    self.save_best_model(epoch, train_ite)

                    if self.early_stop:
                        stop_flag = self.stop_criterion(epoch, train_ite)

                if train_ite + 1 > self.max_batch_steps:
                    stop_flag = True
                    progress_bar.update(1)
                    progress_bar.close()
                    break

            self.loss_history.append(self.logs_total_loss_train.avg)

            if self.scheduler:
                self.scheduler.step()

            if (
                epoch > 0 and epoch % self.ckp_interval == 0
            ) or epoch + 1 == self.epochs:
                self.save_model(f"ckp_{epoch}.pth.tar", epoch)

            if stop_flag:
                break

        if self.wandb_vis:
            wandb.save("model.h5")
            wandb.finish()

        return self.model

    def test(
        self,
        test_dataloader,
        save_path: Union[str, Path] = None,
        compare_no_learning: bool = True,
        log_raw_metrics: bool = False,
    ) -> dict:
        r"""
        Test the model, compute metrics and plot images.

        :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] test_dataloader: Test data loader(s), see options 1 to 3
            above for how we expect data to be provided.
        :param str save_path: Directory in which to save the plotted images.
        :param bool compare_no_learning: If ``True``, the linear reconstruction is compared to the network reconstruction.
        :param bool log_raw_metrics: if `True`, also return non-aggregated metrics as a list.
        :returns: dict of metrics results with means and stds.
        """
        self.compare_no_learning = compare_no_learning
        self.setup_train(train=False)

        self.save_folder_im = save_path
        aux = (self.wandb_vis, self.log_train_batch)
        self.wandb_vis = False
        self.log_train_batch = False

        self.reset_metrics()

        if not isinstance(test_dataloader, list):
            test_dataloader = [test_dataloader]

        self.current_eval_iterators = [iter(loader) for loader in test_dataloader]

        batches = min([len(loader) - loader.drop_last for loader in test_dataloader])

        self.model.eval()
        for i in (
            progress_bar := tqdm(
                range(batches),
                ncols=150,
                disable=(not self.verbose or not self.show_progress_bar),
            )
        ):
            progress_bar.set_description(f"Test")
            self.step(0, progress_bar, train=False, last_batch=(i == batches - 1))

        self.wandb_vis, self.log_train_batch = aux

        if self.verbose:
            print("Test results:")

        out = {}
        for k, l in enumerate(self.logs_metrics_eval):
            if compare_no_learning:
                name = self.metrics[k].__class__.__name__ + " no learning"
                out[name] = self.logs_metrics_no_learning[k].avg
                out[name + "_std"] = self.logs_metrics_no_learning[k].std
                if log_raw_metrics:
                    out[name + "_vals"] = self.logs_metrics_no_learning[k].vals
                if self.verbose:
                    print(
                        f"{name}: {self.logs_metrics_no_learning[k].avg:.3f} +- {self.logs_metrics_no_learning[k].std:.3f}"
                    )

            name = self.metrics[k].__class__.__name__
            out[name] = l.avg
            out[name + "_std"] = l.std
            if log_raw_metrics:
                out[name + "_vals"] = l.vals
            if self.verbose:
                print(f"{name}: {l.avg:.3f} +- {l.std:.3f}")

        return out


def train(
    model: torch.nn.Module,
    physics: Physics,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int = 100,
    losses: Union[Loss, list[Loss]] = SupLoss(),
    eval_dataloader: torch.utils.data.DataLoader = None,
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
    :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s), see options 1 to 3
        above for how we expect data to be provided.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        :ref:`See the libraries' training losses <loss>`.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] eval_dataloader: Evaluation data loader(s), see options 1 to 3
        above for how we expect data to be provided.
    :param args: Other positional arguments to pass to Trainer constructor. See :class:`deepinv.Trainer`.
    :param kwargs: Keyword arguments to pass to Trainer constructor. See :class:`deepinv.Trainer`.
    :return: Trained model.
    """
    trainer = Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        epochs=epochs,
        losses=losses,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        *args,
        **kwargs,
    )
    trained_model = trainer.train()
    return trained_model
