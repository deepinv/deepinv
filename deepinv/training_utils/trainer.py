import warnings
import torchvision.utils
from deepinv.utils import AverageMeter, get_timestamp, plot, rescale_img
import os
import numpy as np
from tqdm import tqdm
import torch
import wandb
from pathlib import Path
from typing import Union, List
from dataclasses import dataclass, field
from deepinv.loss import PSNR, Loss, SupLoss
from deepinv.physics import Physics
from .testing import test


@dataclass
class Trainer:
    r"""
    Trainer class for training a reconstruction network.

    Training can be done by calling the :meth:`deepinv.Trainer.train` method, whereas
    testing can be done by calling the :meth:`deepinv.Trainer.test` method.

    Training details are saved every ``ckp_interval`` epochs in the following format

    ::

        save_path/yyyy-mm-dd_hh-mm-ss/ckp_{epoch}.pth.tar

    where ``.pth.tar`` file contains a dictionary with the keys: ``epoch`` current epoch, ``state_dict`` the state
    dictionary of the model, ``loss`` the loss history, and ``optimizer`` the state dictionary of the optimizer.

    The class provides a flexible training loop that can be customized by the user. In particular, the user can
    rewrite the :meth:`deepinv.Trainer.compute_loss` method to define their custom training step without having
    to write all the training code from scratch:


    ::

        class CustomTrainer(Trainer):
            def compute_loss(self, physics, x, y, train=True):
                logs = {}

                self.optimizer.zero_grad() # Zero the gradients

                # Evaluate reconstruction network
                x_net = self.model_inference(y=y, physics=physics)

                # Compute the losses
                loss_total = 0
                for k, l in enumerate(self.losses):
                    loss = l(x=x, x_net=x_net, y=y, physics=physics, model=self.model)
                    loss_total += loss.mean()

                current_log = self.logs_total_loss_train if train else self.logs_total_loss_eval
                current_log.update(loss_total.item())
                logs[f"TotalLoss"] = current_log.avg

                if train:
                    loss_total.backward()  # Backward the total loss
                    self.optimizer.step() # Optimizer step

                return x_net, logs


    If the user wants to change the way the metrics are computed, they can rewrite the
    :meth:`deepinv.Trainer.compute_metrics` method.
    The user can also change the way the samples are generated by rewriting the :meth:`deepinv.Trainer.get_samples_online`
    or :meth:`deepinv.Trainer.get_samples_offline` methods.

    .. note::

        The training code can synchronize with `Weights & Biases <https://wandb.ai/site>`_ for visualization
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

    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``. This results in a wider range of measurements if the physics' parameters, such as
        parameters of the forward operator or noise realizations, can change between each sample;
        the measurements are loaded from the training dataset.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] metrics: Metric or list of metrics used for evaluating the model.
        :ref:`See the libraries' evaluation metrics <loss>`.
    :param float grad_clip: Gradient clipping value for the optimizer. If None, no gradient clipping is performed.
    :param torch.device device: gpu or cpu.
    :param int ckp_interval: The model is saved every ``ckp_interval`` epochs.
    :param int eval_interval: Number of epochs between each evaluation of the model on the evaluation set.
    :param str save_path: Directory in which to save the trained model.
    :param bool verbose: Output training progress information in the console.
    :param bool plot_images: Plots reconstructions every ``ckp_interval`` epochs.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    :param dict wandb_setup: Dictionary with the setup for wandb, see https://docs.wandb.ai/quickstart for more details.
    :param bool plot_measurements: Plot the measurements y. default=True.
    :param bool check_grad: Compute and print the gradient norm at each iteration.
    :param str ckpt_pretrained: path of the pretrained checkpoint. If None, no pretrained checkpoint is loaded.
    :param int freq_plot: Frequency of plotting images to wandb during train evaluation (at the end of each epoch).
        If ``1``, plots at each epoch.
    :param bool verbose_individual_losses: If ``True``, the value of individual losses are printed during training.
        Otherwise, only the total loss is printed.
    """

    metrics: Union[Loss, List[Loss]] = PSNR()
    online_measurements: bool = False
    grad_clip: float = None
    device: Union[str, torch.device] = "cpu"
    ckp_interval: int = 1
    eval_interval: int = 1
    save_path: Union[str, Path] = "."
    verbose: bool = True
    plot_images: bool = False
    plot_metrics: bool = False
    wandb_vis: bool = False
    wandb_setup: dict = field(default_factory=dict)
    plot_measurements: bool = True
    check_grad: bool = False
    ckpt_pretrained: Union[str, None] = None
    freq_plot: int = 1
    verbose_individual_losses: bool = True
    display_losses_eval: bool = False

    def setup_train(self, losses, physics):
        r"""
        Set up the training process.

        It initializes the wandb logging, the different metrics, the save path, the physics and dataloaders,
        and the pretrained checkpoint if given.
        """

        self.save_path = Path(self.save_path)

        if self.wandb_setup is not None and not self.wandb_vis:
            warnings.warn(
                "wandb_vis is False but wandb_setup is provided. Wandb visualization deactivated (wandb_vis=False)."
            )

        # wandb initialization
        if self.wandb_vis:
            if wandb.run is None:
                wandb.init(**self.wandb_setup)

        if not isinstance(losses, list) or isinstance(losses, tuple):
            self.losses = [losses]
        else:
            self.losses = losses

        if not isinstance(self.metrics, list) or isinstance(self.metrics, tuple):
            self.metrics = [self.metrics]

        # losses
        self.logs_total_loss_train = AverageMeter("Training loss", ":.2e")
        self.logs_losses_train = [
            AverageMeter("Training loss " + l.name, ":.2e") for l in self.losses
        ]

        self.logs_total_loss_eval = AverageMeter("Validation loss", ":.2e")
        self.logs_losses_eval = [
            AverageMeter("Validation loss " + l.name, ":.2e") for l in self.losses
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

        # gradient clipping
        if self.check_grad:
            self.check_grad_val = AverageMeter("Gradient norm", ":.2e")

        self.save_path = f"{self.save_path}/{get_timestamp()}"

        # count the overall training parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {params} trainable parameters")

        # make physics and data_loaders of list type
        if type(physics) is not list:
            self.physics = [physics]
        else:
            self.physics = physics

        self.loss_history = []

        self.epoch_start = 0
        if self.ckpt_pretrained is not None:
            checkpoint = torch.load(self.ckpt_pretrained)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch_start = checkpoint["epoch"]

    def prepare_images(self, physics_cur, x, y, x_net):
        r"""
        Prepare the images for plotting.

        It prepares the images for plotting by rescaling them and concatenating them in a grid.

        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Reconstruction network output.
        :returns: The images, the titles, the grid image, and the caption.
        """
        with torch.no_grad():
            if (
                self.plot_measurements
                and len(y.shape) == len(x.shape)
                and y.shape != x.shape
            ):
                y_reshaped = torch.nn.functional.interpolate(y, size=x.shape[2])
                if hasattr(physics_cur, "A_adjoint"):
                    imgs = [y_reshaped, physics_cur.A_adjoint(y), x_net, x]
                    caption = (
                        "From top to bottom: input, backprojection, output, target"
                    )
                    titles = ["Input", "Backprojection", "Output", "Target"]
                else:
                    imgs = [y_reshaped, x_net, x]
                    titles = ["Input", "Output", "Target"]
                    caption = "From top to bottom: input, output, target"
            else:
                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        back = physics_cur.module.A_adjoint(y)
                    else:
                        back = physics_cur.A_adjoint(y)
                    imgs = [back, x_net, x]
                    titles = ["Backprojection", "Output", "Target"]
                    caption = "From top to bottom: backprojection, output, target"
                else:
                    imgs = [x_net, x]
                    caption = "From top to bottom: output, target"
                    titles = ["Output", "Target"]

            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])

        return imgs, titles, grid_image, caption

    def log_metrics_wandb(self, log_dict_iter):
        r"""
        Log the metrics to wandb.
        """
        if self.wandb_vis:
            wandb.log(log_dict_iter)

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

        if type(data) is tuple and len(data) == 2:
            x, _ = data
        else:
            x = data

        x = x.to(self.device)

        physics = self.physics[g]

        y = physics(x)

        return x, y, physics

    def get_samples_offline(self, iterators, g):
        r"""
        Get the samples for the offline measurements.

        In this setting, samples have been generated offline and are loaded from the dataloader.
        This function returns a dictionary containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a dictionary containing at least: the ground truth, the measurement, and the current physics operator.
        """
        data = next(iterators[g])
        if (type(data) is not tuple and type(data) is not list) or len(data) != 2:
            raise ValueError(
                "If online_measurements=True, the dataloader should output a tuple (x, y)"
            )

        x, y = data
        if type(x) is list or type(x) is tuple:
            x = [s.to(self.device) for s in x]
        else:
            x = x.to(self.device)

        y = y.to(self.device)
        physics = self.physics[g]

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

    def model_inference(self, y, physics):
        r"""
        Perform the model inference.

        It returns the network reconstruction given the samples.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :returns: The network reconstruction.
        """
        y = y.to(self.device)
        x_net = self.model(y, physics)
        return x_net

    def compute_loss(self, physics, x, y, train=True):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        self.optimizer.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics)

        if train or self.display_losses_eval:
            # Compute the losses
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(x=x, x_net=x_net, y=y, physics=physics, model=self.model)
                loss_total += loss.mean()
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    current_log = (
                        self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                    )
                    current_log.update(loss.detach().cpu().numpy())
                    cur_loss = current_log.avg
                    logs[l.__class__.__name__] = cur_loss

            current_log = (
                self.logs_total_loss_train if train else self.logs_total_loss_eval
            )
            current_log.update(loss_total.item())
            logs[f"TotalLoss"] = current_log.avg

        if train:
            loss_total.backward()  # Backward the total loss

            norm = self.check_clip_grad()  # Optional gradient clipping
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            # Optimizer step
            self.optimizer.step()

        return x_net, logs

    def compute_metrics(self, x, x_net, y, physics, logs, train=True):
        r"""
        Compute the metrics.

        It computes the metrics over the batch.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param dict logs: Dictionary containing the logs for printing the training progress.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :returns: The logs with the metrics.
        """
        # Compute the metrics over the batch
        with torch.no_grad():
            for k, l in enumerate(self.metrics):
                metric = l(x=x, x_net=x_net, y=y, physics=physics)

                current_log = (
                    self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                )
                current_log.update(metric.detach().cpu().numpy())
                logs[l.__class__.__name__] = current_log.avg

        return logs

    def step(self, epoch, progress_bar, train=True, last_batch=False):
        r"""
        Train/Eval a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration.

        :param int epoch: Current epoch.
        :param tqdm progress_bar: Progress bar.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        :returns: The current physics operator, the ground truth, the measurement, and the network reconstruction.
        """

        # random permulation of the dataloaders
        G_perm = np.random.permutation(self.G)

        for g in G_perm:  # for each dataloader
            x, y, physics_cur = self.get_samples(self.current_iterators, g)

            # Compute loss and perform backprop
            x_net, logs = self.compute_loss(physics_cur, x, y, train=train)

            # Log metrics
            logs = self.compute_metrics(x, x_net, y, physics_cur, logs)

            # Update the progress bar
            progress_bar.set_postfix(logs)

        if last_batch:
            logs["step"] = epoch
            self.log_metrics_wandb(logs)  # Log metrics to wandb
            self.plot(epoch, physics_cur, x, y, x_net, train=train)  # Plot images

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot the images.

        It plots the images at the end of each epoch.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        post_str = "Training" if train else "Eval"
        if self.plot_images and ((epoch + 1) % self.freq_plot == 0):
            imgs, titles, grid_image, caption = self.prepare_images(
                physics, x, y, x_net
            )

            plot(
                imgs,
                titles=titles,
                show=self.plot_images,
                return_fig=True,
                rescale_mode="clip",
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch)

    def save_model(self, epoch, eval_psnr=None):
        r"""
        Save the model.

        It saves the model every ``ckp_interval`` epochs.

        :param int epoch: Current epoch.
        :param None, float eval_psnr: Evaluation PSNR.
        """
        if (epoch > 0 and epoch % self.ckp_interval == 0) or epoch + 1 == self.epochs:
            os.makedirs(str(self.save_path), exist_ok=True)
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "loss": self.loss_history,
                "optimizer": self.optimizer.state_dict(),
            }
            if eval_psnr is not None:
                state["eval_psnr"] = eval_psnr
            torch.save(
                state, os.path.join(str(self.save_path), "ckp_{}.pth.tar".format(epoch))
            )

    def train(
        self,
        model: torch.nn.Module,
        physics: Physics,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        epochs: int = 100,
        losses: Union[Loss, List[Loss]] = SupLoss(),
        eval_dataloader=None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        :param torch.nn.Module model: Reconstruction network, which can be PnP, unrolled, artifact removal
            or any other custom reconstruction network.
        :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s) used by the reconstruction network.
        :param int epochs: Number of training epochs. Default is 100.
        :param torch.nn.optim.Optimizer optimizer: Torch optimizer for training the network.
        :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s) should provide a
            a signal x or a tuple of (x, y) signal/measurement pairs.
        :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
            :ref:`See the libraries' training losses <loss>`. By default, it uses the supervised mean squared error.
        :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] eval_dataloader: Evaluation data loader(s)
            should provide a signal x or a tuple of (x, y) signal/measurement pairs.
        :param None, torch.optim.lr_scheduler.LRScheduler scheduler: Torch scheduler for changing the learning rate across iterations.
        :returns: The trained model.
        """

        self.epochs = epochs
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        if type(train_dataloader) is not list:
            train_dataloader = [train_dataloader]
        if eval_dataloader and type(eval_dataloader) is not list:
            eval_dataloader = [eval_dataloader]

        self.G = len(train_dataloader)

        self.setup_train(losses=losses, physics=physics)
        for epoch in range(self.epoch_start, self.epochs):
            ## Evaluation
            perform_eval = eval_dataloader and (
                (epoch + 1) % self.eval_interval == 0 or epoch + 1 == self.epochs
            )
            if perform_eval:
                self.current_iterators = [iter(loader) for loader in eval_dataloader]
                batches = len(eval_dataloader[self.G - 1]) - int(
                    eval_dataloader[self.G - 1].drop_last
                )

                self.model.eval()
                for i in (
                    progress_bar := tqdm(
                        range(batches), ncols=150, disable=not self.verbose
                    )
                ):
                    progress_bar.set_description(f"Eval epoch {epoch + 1}")
                    self.step(
                        epoch, progress_bar, train=False, last_batch=(i == batches - 1)
                    )

                self.eval_psnr = self.logs_metrics_eval[0].avg

            ## Training
            self.current_iterators = [iter(loader) for loader in train_dataloader]
            batches = len(train_dataloader[self.G - 1]) - int(
                train_dataloader[self.G - 1].drop_last
            )

            self.model.train()
            for i in (
                progress_bar := tqdm(
                    range(batches), ncols=150, disable=not self.verbose
                )
            ):
                progress_bar.set_description(f"Train epoch {epoch + 1}")
                self.step(
                    epoch, progress_bar, train=True, last_batch=(i == batches - 1)
                )

            self.loss_history.append(self.logs_total_loss_train.avg)

            if self.scheduler:
                self.scheduler.step()

            # Saving the model
            self.save_model(epoch, eval_psnr=self.eval_psnr if perform_eval else None)

        if self.wandb_vis:
            wandb.save("model.h5")

        return self.model

    def test(self, model, physics, test_dataloader):
        r"""
        Test the model.

        It computes the quality metrics of the reconstruction network on the test set, and it
        compares the performance of a simple reconstruction that does not learn.

        :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP,
            unrolled, artifact removal or any other custom reconstruction network.
        :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics:
            Forward operator(s) used by the reconstruction network.
        :param torch.utils.data.DataLoader test_dataloader: Test data loader, which should provide a tuple of (x, y) pairs.
        :returns tuple[float]: The PSNR of the model, the standard deviation of the PSNR of the model,
            the PSNR of the model without learning, and the standard deviation of the PSNR of the model without learning.
        """

        return test(
            model,
            physics=physics,
            test_dataloader=test_dataloader,
            online_measurements=self.online_measurements,
            plot_images=self.plot_images,
            plot_metrics=self.plot_metrics,
            save_folder=self.save_path + "/test",
            metrics=self.metrics,
            device=self.device,
            verbose=self.verbose,
        )


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
    Alias function for training a model using :class:`deepinv.training_utils.Trainer` class.

    This function creates a Trainer instance and returns the trained model.

    .. warning::

        This function is deprecated and will be removed in future versions. Please use
        :class:`deepinv.Trainer` instead.

    :param torch.nn.Module model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s) used by the reconstruction network.
    :param int epochs: Number of training epochs. Default is 100.
    :param torch.nn.optim.Optimizer optimizer: Torch optimizer for training the network.
    :param torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] train_dataloader: Train data loader(s) should provide a
        a signal x or a tuple of (x, y) signal/measurement pairs.
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] losses: Loss or list of losses used for training the model.
        :ref:`See the libraries' training losses <loss>`. By default, it uses the supervised mean squared error.
    :param None, torch.utils.data.DataLoader, list[torch.utils.data.DataLoader] eval_dataloader: Evaluation data loader(s)
        should provide a signal x or a tuple of (x, y) signal/measurement pairs.
    :param args: Other positional arguments to pass to Trainer constructor. See :meth:`deepinv.Trainer`.
    :param kwargs: Keyword arguments to pass to Trainer constructor. See :meth:`deepinv.Trainer`.
    :return: Trained model.
    """
    trainer = Trainer(*args, **kwargs)
    trained_model = trainer.train(
        model,
        physics=physics,
        optimizer=optimizer,
        epochs=epochs,
        losses=losses,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    return trained_model
