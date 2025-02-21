import warnings
from deepinv.utils import AverageMeter, get_timestamp, plot, plot_curves
import os
import numpy as np
from tqdm import tqdm
import torch
import wandb
from pathlib import Path
from typing import Union, List
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

    See the :ref:`User Guide <trainer>` for more details on how to adapt the trainer to your needs.

    Training can be done by calling the :func:`deepinv.Trainer.train` method, whereas
    testing can be done by calling the :func:`deepinv.Trainer.test` method.

    Training details are saved every ``ckp_interval`` epochs in the following format

    ::

        save_path/yyyy-mm-dd_hh-mm-ss/ckp_{epoch}.pth.tar

    where ``.pth.tar`` file contains a dictionary with the keys: ``epoch`` current epoch, ``state_dict`` the state
    dictionary of the model, ``loss`` the loss history, ``optimizer`` the state dictionary of the optimizer,
    and ``eval_metrics`` the evaluation metrics history.

    - Use :func:`deepinv.Trainer.get_samples_online` when measurements are simulated from a ground truth returned by the dataloader.
    - Use :func:`deepinv.Trainer.get_samples_offline` when both the ground truth and measurements are returned by the dataloader (and also optionally physics generator params).

    .. note::

        The training code can synchronize with `Weights & Biases <https://wandb.ai/site>`_ for logging and visualization
        by setting ``wandb_vis=True``. The user can also customize the wandb setup by providing
        a dictionary with the setup for wandb.

    .. note::

        The losses and evaluation metrics
        can be chosen from :ref:`the libraries' training losses <loss>`, or can be a custom loss function,
        as long as it takes as input ``(x, x_net, y, physics, model)`` and returns a scalar, where ``x`` is the ground
        reconstruction, ``x_net`` is the network reconstruction :math:`\inversef{y}{A}`,
        ``y`` is the measurement vector, ``physics`` is the forward operator
        and ``model`` is the reconstruction network. Note that not all inpus need to be used by the loss,
        e.g., self-supervised losses will not make use of ``x``.

    .. warning::

        If a physics generator is used to generate params for online measurements, the generated params will vary each epoch.
        If this is not desired (you want the same online measurements each epoch), set ``loop_physics_generator=True``.
        Caveat: this requires ``shuffle=False`` in your dataloaders.
        An alternative, safer solution is to generate and save params offline using :func:`deepinv.datasets.generate_dataset`.
        The params dict will then be automatically updated every time data is loaded.

    :param torch.nn.Module model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s) used by the reconstruction network.
    :param int epochs: Number of training epochs. Default is 100.
    :param torch.optim.Optimizer optimizer: Torch optimizer for training the network.
    :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s) should provide a
        a signal x or a tuple of (x, y) signal/measurement pairs.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        Optionally wrap losses using a loss scheduler for more advanced training.
        :ref:`See the libraries' training losses <loss>`. By default, it uses the supervised mean squared error.
        Where relevant, the underlying metric should have ``reduction=None`` as we perform the averaging using :class:`deepinv.utils.AverageMeter` to deal with uneven batch sizes.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] eval_dataloader: Evaluation data loader(s)
        should provide a signal x or a tuple of (x, y) signal/measurement pairs.
    :param None, torch.optim.lr_scheduler.LRScheduler scheduler: Torch scheduler for changing the learning rate across iterations.
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``. This results in a wider range of measurements if the physics' parameters, such as
        parameters of the forward operator or noise realizations, can change between each sample;
        the measurements are loaded from the training dataset.
    :param None, deepinv.physics.generator.PhysicsGenerator physics_generator: Optional physics generator for generating
        the physics operators. If not None, the physics operators are randomly sampled at each iteration using the generator.
        Should be used in conjunction with ``online_measurements=True``. Also see ``loop_physics_generator``.
    :param bool loop_physics_generator: if True, resets the physics generator back to its initial state at the beginning of each epoch,
        so that the same measurements are generated each epoch. Requires `shuffle=False` in dataloaders. If False, generates new physics every epoch.
        Used in conjunction with ``physics_generator``.
    :param Metric, list[Metric] metrics: Metric or list of metrics used for evaluating the model.
        They should have ``reduction=None`` as we perform the averaging using :class:`deepinv.utils.AverageMeter` to deal with uneven batch sizes.
        :ref:`See the libraries' evaluation metrics <metric>`.
    :param str device: Device on which to run the training (e.g., 'cuda' or 'cpu').
    :param str ckpt_pretrained: path of the pretrained checkpoint. If None, no pretrained checkpoint is loaded.
    :param str save_path: Directory in which to save the trained model.
    :param bool compare_no_learning: If ``True``, the no learning method is compared to the network reconstruction.
    :param str no_learning_method: Reconstruction method used for the no learning comparison. Options are ``'A_dagger'``, ``'A_adjoint'``,
        ``'prox_l2'``, or ``'y'``. Default is ``'A_dagger'``. The user can also provide a custom method by overriding the
        :func:`no_learning_inference <deepinv.Trainer.no_learning_inference>` method.
    :param float grad_clip: Gradient clipping value for the optimizer. If None, no gradient clipping is performed.
    :param bool check_grad: Compute and print the gradient norm at each iteration.
    :param bool wandb_vis: Logs data onto Weights & Biases, see https://wandb.ai/ for more details.
    :param dict wandb_setup: Dictionary with the setup for wandb, see https://docs.wandb.ai/quickstart for more details.
    :param int ckp_interval: The model is saved every ``ckp_interval`` epochs.
    :param int eval_interval: Number of epochs (or train iters, if ``log_train_batch=True``) between each evaluation of the model on the evaluation set.
    :param int plot_interval: Frequency of plotting images to wandb during train evaluation (at the end of each epoch).
        If ``1``, plots at each epoch.
    :param int freq_plot: deprecated. Use ``plot_interval``
    :param bool plot_images: Plots reconstructions every ``ckp_interval`` epochs.
    :param bool plot_measurements: Plot the measurements y, default=`True`.
    :param bool plot_convergence_metrics: Plot convergence metrics for model, default=`False`.
    :param str rescale_mode: Rescale mode for plotting images. Default is ``'clip'``.
    :param bool display_losses_eval: If ``True``, the losses are displayed during evaluation.
    :param bool log_train_batch: if ``True``, log train batch and eval-set metrics and losses for each train batch during training.
        This is useful for visualising train progress inside an epoch, not just over epochs.
        If ``False`` (default), log average over dataset per epoch (standard training).
    :param bool verbose: Output training progress information in the console.
    :param bool verbose_individual_losses: If ``True``, the value of individual losses are printed during training.
        Otherwise, only the total loss is printed.
    :param bool show_progress_bar: Show a progress bar during training.
    """

    model: torch.nn.Module
    physics: Union[Physics, List[Physics]]
    optimizer: Union[torch.optim.Optimizer, None]
    train_dataloader: torch.utils.data.DataLoader
    epochs: int = 100
    losses: Union[Loss, BaseLossScheduler, List[Loss], List[BaseLossScheduler]] = (
        SupLoss()
    )
    eval_dataloader: torch.utils.data.DataLoader = None
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    online_measurements: bool = False
    physics_generator: Union[PhysicsGenerator, List[PhysicsGenerator]] = None
    loop_physics_generator: bool = False
    metrics: Union[Metric, List[Metric]] = PSNR()
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

        self.eval_metrics_history = {}
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
            and self.loop_physics_generator
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

        if not isinstance(self.losses, list) or isinstance(self.losses, tuple):
            self.losses = [self.losses]

        for l in self.losses:
            self.model = l.adapt_model(self.model)

        if not isinstance(self.metrics, list) or isinstance(self.metrics, tuple):
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
            self.logs_metrics_linear = [
                AverageMeter("Validation metric " + l.__class__.__name__, ":.2e")
                for l in self.metrics
            ]

        # gradient clipping
        if train and self.check_grad:
            self.check_grad_val = AverageMeter("Gradient norm", ":.2e")

        self.save_path = (
            f"{self.save_path}/{get_timestamp()}" if self.save_path else None
        )

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

        self.load_model()

    def load_model(self, ckpt_pretrained: str = None):
        """Load model from checkpoint.

        :param str ckpt_pretrained: checkpoint filename. If `None`, use checkpoint passed to class. If not `None`, override checkpoint passed to class.
        """
        if ckpt_pretrained is None and self.ckpt_pretrained is not None:
            ckpt_pretrained = self.ckpt_pretrained

        if ckpt_pretrained is not None:
            checkpoint = torch.load(
                ckpt_pretrained, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            if "optimizer" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "wandb_id" in checkpoint and self.wandb_vis:
                self.wandb_setup["id"] = checkpoint["wandb_id"]
                self.wandb_setup["resume"] = "allow"
            if "epoch" in checkpoint:
                self.epoch_start = checkpoint["epoch"]

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
        data = next(
            iterators[g]
        )  # In this case the dataloader outputs also a class label

        if type(data) is tuple or type(data) is list:
            x = data[0]
        else:
            x = data

        x = x.to(self.device)
        physics = self.physics[g]

        if self.physics_generator is not None:
            params = self.physics_generator[g].step(batch_size=x.size(0))
            # Update parameters both via update_parameters and, if implemented in physics, via forward pass
            physics.update_parameters(**params)
            y = physics(x, **params)
        else:
            y = physics(x)

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
        elif len(data) == 3:
            x, y, params = data

        if type(x) is list or type(x) is tuple:
            x = [s.to(self.device) for s in x]
        else:
            x = x.to(self.device)

        y = y.to(self.device)
        physics = self.physics[g]

        if params is not None:
            params = {k: p.to(self.device) for k, p in params.items()}
            physics.update_parameters(**params)

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
            samples = self.get_samples_online(iterators, g)
        else:  # the measurements y were pre-computed
            samples = self.get_samples_offline(iterators, g)

        return samples

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

    def compute_loss(self, physics, x, y, train=True, epoch: int = None):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        if train:
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
                    cur_loss = meters.avg
                    logs[l.__class__.__name__] = cur_loss

            meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
            meters.update(loss_total.item())
            logs[f"TotalLoss"] = meters.avg

        if train:
            loss_total.backward()  # Backward the total loss

            norm = self.check_clip_grad()  # Optional gradient clipping
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            # Optimizer step
            self.optimizer.step()

        return x_net, logs

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
                    x_net=x_net,
                    x=x,
                    epoch=epoch,
                )

                current_log = (
                    self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                )
                current_log.update(metric.detach().cpu().numpy())
                logs[l.__class__.__name__] = current_log.avg

                if not train and self.compare_no_learning:
                    x_lin = self.no_learning_inference(y, physics)
                    metric = l(x=x, x_net=x_lin, y=y, physics=physics, model=self.model)
                    self.logs_metrics_linear[k].update(metric.detach().cpu().numpy())
                    logs[f"{l.__class__.__name__} no learning"] = (
                        self.logs_metrics_linear[k].avg
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
                f"No learning reconstruction method {self.no_learning_method} not recognized"
            )

        return x_nl

    def step(self, epoch, progress_bar, train_ite=None, train=True, last_batch=False):
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

        # random permulation of the dataloaders
        G_perm = np.random.permutation(self.G)

        if self.log_train_batch and train:
            self.reset_metrics()

        for g in G_perm:  # for each dataloader
            x, y, physics_cur = self.get_samples(
                self.current_train_iterators if train else self.current_eval_iterators,
                g,
            )

            # Compute loss and perform backprop
            x_net, logs = self.compute_loss(physics_cur, x, y, train=train, epoch=epoch)

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
                x, y, x_net, x_nl, rescale_mode=self.rescale_mode
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

    def save_model(self, epoch, eval_metrics=None, state={}):
        r"""
        Save the model.

        It saves the model every ``ckp_interval`` epochs.

        :param int epoch: Current epoch.
        :param None, float eval_metrics: Evaluation metrics across epochs.
        :param dict state: custom objects to save with model
        """

        if not self.save_path:
            return

        if (epoch > 0 and epoch % self.ckp_interval == 0) or epoch + 1 == self.epochs:
            os.makedirs(str(self.save_path), exist_ok=True)
            state = state | {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "loss": self.loss_history,
                "optimizer": self.optimizer.state_dict(),
            }
            if eval_metrics is not None:
                state["eval_metrics"] = eval_metrics
            if self.wandb_vis:
                state["wandb_id"] = wandb.run.id
            torch.save(
                state,
                os.path.join(
                    Path(self.save_path), Path("ckp_{}.pth.tar".format(epoch))
                ),
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

        for epoch in range(self.epoch_start, self.epochs):
            self.reset_metrics()

            ## Training
            self.current_train_iterators = [
                iter(loader) for loader in self.train_dataloader
            ]

            batches = min(
                [len(loader) - loader.drop_last for loader in self.train_dataloader]
            )

            if self.loop_physics_generator and self.physics_generator is not None:
                for physics_generator in self.physics_generator:
                    physics_generator.reset_rng()

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
                    for j in (
                        eval_progress_bar := tqdm(
                            range(eval_batches),
                            ncols=150,
                            disable=(not self.verbose or not self.show_progress_bar),
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

                    for l in self.logs_losses_eval:
                        self.eval_metrics_history[l.__class__.__name__] = l.avg

            self.loss_history.append(self.logs_total_loss_train.avg)

            if self.scheduler:
                self.scheduler.step()

            # Saving the model
            self.save_model(epoch, self.eval_metrics_history if perform_eval else None)

        if self.wandb_vis:
            wandb.save("model.h5")
            wandb.finish()

        return self.model

    def test(self, test_dataloader, save_path=None, compare_no_learning=True) -> dict:
        r"""
        Test the model, compute metrics and plot images.

        :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] test_dataloader: Test data loader(s) should provide a
            a signal x or a tuple of (x, y) signal/measurement pairs.
        :param str save_path: Directory in which to save the plotted images.
        :param bool compare_no_learning: If ``True``, the linear reconstruction is compared to the network reconstruction.
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
                out[name] = self.logs_metrics_linear[k].avg
                out[name + "_std"] = self.logs_metrics_linear[k].std
                if self.verbose:
                    print(
                        f"{name}: {self.logs_metrics_linear[k].avg:.3f} +- {self.logs_metrics_linear[k].std:.3f}"
                    )

            name = self.metrics[k].__class__.__name__
            out[name] = l.avg
            out[name + "_std"] = l.std
            if self.verbose:
                print(f"{name}: {l.avg:.3f} +- {l.std:.3f}")

        return out


def train(
    model: torch.nn.Module,
    physics: Physics,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int = 100,
    losses: Union[Loss, List[Loss]] = SupLoss(),
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

    :param torch.nn.Module model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s) used by the reconstruction network.
    :param int epochs: Number of training epochs. Default is 100.
    :param torch.optim.Optimizer optimizer: Torch optimizer for training the network.
    :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s) should provide a
        a signal x or a tuple of (x, y) signal/measurement pairs.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        :ref:`See the libraries' training losses <loss>`. By default, it uses the supervised mean squared error.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] eval_dataloader: Evaluation data loader(s)
        should provide a signal x or a tuple of (x, y) signal/measurement pairs.
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
