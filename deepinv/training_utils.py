import warnings
import torchvision.utils
from deepinv.utils import (
    save_model,
    AverageMeter,
    get_timestamp,
    cal_psnr,
)
from deepinv.utils import plot, plot_curves, wandb_plot_curves, rescale_img, zeros_like
from deepinv.physics import Physics
import numpy as np
from tqdm import tqdm
import torch
import wandb
from pathlib import Path
from typing import Union
from dataclasses import dataclass, field


@dataclass
class Trainer:
    r"""
    Trainer class for training a reconstruction network.

    .. note::

        The losses can be chosen from :ref:`the libraries' training losses <loss>`, or can be a custom loss function,
        as long as it takes as input ``(x, x_net, y, physics, model)`` and returns a scalar, where ``x`` is the ground
        reconstruction, ``x_net`` is the network reconstruction :math:`\inversef{y}{A}`,
        ``y`` is the measurement vector, ``physics`` is the forward operator
        and ``model`` is the reconstruction network. Note that not all inpus need to be used by the loss,
        e.g., self-supervised losses will not make use of ``x``.


    :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param torch.utils.data.DataLoader train_dataloader: Train dataloader.
    :param int epochs: Number of training epochs.
    :param torch.nn.Module, list of torch.nn.Module losses: Loss or list of losses used for training the model.
    :param torch.utils.data.DataLoader eval_dataloader: Evaluation dataloader.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s)
        used by the reconstruction network at train time.
    :param torch.nn.optim optimizer: Torch optimizer for training the network.
    :param float grad_clip: Gradient clipping value for the optimizer. If None, no gradient clipping is performed.
    :param torch.nn.optim scheduler: Torch scheduler for changing the learning rate across iterations.
    :param torch.device device: gpu or cpu.
    :param int ckp_interval: The model is saved every ``ckp_interval`` epochs.
    :param int eval_interval: Number of epochs between each evaluation of the model on the evaluation set.
    :param str save_path: Directory in which to save the trained model.
    :param bool verbose: Output training progress information in the console.
    :param bool unsupervised: Train an unsupervised network, i.e., uses only measurement vectors y for training.
    :param bool plot_images: Plots reconstructions every ``ckp_interval`` epochs.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    :param dict wandb_setup: Dictionary with the setup for wandb, see https://docs.wandb.ai/quickstart for more details.
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``. This results in a wider range of measurements if the physics' parameters, such as
        parameters of the forward operator or noise realizations, can change between each sample; these are updated
        with the ``physics.reset()`` method. If ``online_measurements=False``, the measurements are loaded from the training dataset
    :param bool plot_measurements: Plot the measurements y. default=True.
    :param bool check_grad: Check the gradient norm at each iteration.
    :param str ckpt_pretrained: path of the pretrained checkpoint. If None, no pretrained checkpoint is loaded.
    :param list fact_losses: List of factors to multiply the losses. If None, all losses are multiplied by 1.
    :param int freq_plot: Frequency of plotting images to wandb during train evaluation (at the end of each epoch). If 1, plots at each epoch.
    """

    model: torch.nn.Module
    train_dataloader: torch.utils.data.DataLoader
    epochs: int
    losses: list
    eval_dataloader: torch.utils.data.DataLoader = None
    physics: Physics = None
    optimizer: torch.optim.Optimizer = None
    grad_clip: float = None
    scheduler: torch.optim.lr_scheduler.LRScheduler = None
    device: Union[str, torch.device] = "cpu"
    ckp_interval: int = 1
    eval_interval: int = 1
    save_path: Union[str, Path] = "."
    verbose: bool = False
    unsupervised: bool = False
    plot_images: bool = False
    plot_metrics: bool = False
    wandb_vis: bool = False
    wandb_setup: dict = field(default_factory=dict)
    online_measurements: bool = False
    plot_measurements: bool = True
    check_grad: bool = False
    ckpt_pretrained: Union[str, None] = None
    fact_losses: list = None
    freq_plot: int = 1

    def setup_train(self):
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

        # wandb initialiation
        if self.wandb_vis:
            if wandb.run is None:
                wandb.init(**self.wandb_setup)

        self.total_loss = AverageMeter("loss", ":.2e")
        if not isinstance(self.losses, list) or isinstance(self.losses, tuple):
            self.losses = [self.losses]
        if self.fact_losses is None:
            self.fact_losses = [1] * len(self.losses)
        self.losses_verbose = [
            AverageMeter("Loss_" + l.name, ":.2e") for l in self.losses
        ]
        self.train_metric = AverageMeter("Train_psnr_model", ":.2f")
        if self.eval_dataloader:
            self.eval_psnr = AverageMeter("Eval_psnr_model", ":.2f")
        if self.check_grad:
            self.check_grad_val = AverageMeter("Gradient norm", ":.2e")

        self.save_path = f"{self.save_path}/{get_timestamp()}"

        # count the overall training parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {params} trainable parameters")

        # make physics and data_loaders of list type
        if type(self.physics) is not list:
            self.physics = [self.physics]
        if type(self.train_dataloader) is not list:
            self.train_dataloader = [self.train_dataloader]
        if self.eval_dataloader and type(self.eval_dataloader) is not list:
            self.eval_dataloader = [self.eval_dataloader]

        self.G = len(self.train_dataloader)

        self.loss_history = []

        self.epoch_start = 0
        if self.ckpt_pretrained is not None:
            checkpoint = torch.load(self.ckpt_pretrained)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch_start = checkpoint["epoch"]

    def make_grid_image(self, physics_cur, x, y, x_net):

        with torch.no_grad():
            if self.plot_measurements and y.shape != x.shape:
                y_reshaped = torch.nn.functional.interpolate(y, size=x.shape[2])
                if hasattr(physics_cur, "A_adjoint"):
                    imgs = [y_reshaped, physics_cur.A_adjoint(y), x_net, x]
                    caption = (
                        "From top to bottom: input, backprojection, output, target"
                    )
                else:
                    imgs = [y_reshaped, x_net, x]
                    caption = "From top to bottom: input, output, target"
            else:
                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        back = physics_cur.module.A_adjoint(y)
                    else:
                        back = physics_cur.A_adjoint(y)
                    imgs = [back, x_net, x]
                    caption = "From top to bottom: backprojection, output, target"
                else:
                    imgs = [x_net, x]
                    caption = "From top to bottom: output, target"

            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])

        return grid_image, caption

    def epoch_wandb_vis(self, epoch, physics_cur, x, y, x_net):
        r"""
        Perform visualization at the end of each epoch.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.

        """

        if self.wandb_vis:
            # log average training metrics
            log_dict_post_epoch = {}
            log_dict_post_epoch["mean training loss"] = self.total_loss.avg
            log_dict_post_epoch["mean training psnr"] = self.train_metric.avg
            if self.check_grad:
                log_dict_post_epoch["mean gradient norm"] = self.check_grad_val.avg

            grid_image, caption = self.make_grid_image(physics_cur, x, y, x_net)

            if self.plot_images and ((epoch + 1) % self.freq_plot == 0):
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch["Training samples"] = images

            wandb.log(log_dict_post_epoch)

    def batch_metric(self, x, x_net, y, physics, train=True, log=True):
        r"""
        Compute metrics over the training batch at each iteration and logs them.

        It computes the PSNR of the network reconstruction and logs the training metrics.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: Whether the model is being trained. (Necessary to attribute to relevant log key)
        :param bool log: Whether to log the metrics. (Only if train is True)
        """

        assert (
            not self.unsupervised
        ), "batch_metric should not be called when self.unsupervised is True."

        with torch.no_grad():
            metric = cal_psnr(x_net, x)

        if train and log:
            self.train_metric.update(metric)
            self.log_dict_iter["train_psnr"] = self.train_metric.val

        return metric

    def log_metrics(self):
        r"""
        Log the metrics to wandb.
        """
        if self.wandb_vis:
            wandb.log(self.log_dict_iter)

    def check_clip_grad(self):
        r"""
        Check the gradient norm and perform gradient clipping if necessary.

        """
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
            self.log_dict_iter["gradient norm"] = norm_grads.item()
            self.check_grad_val.update(norm_grads.item())

    def backward_pass(self, g, x, y, x_net):
        r"""
        Perform the backward pass.

        It computes the losses and the total loss, and performs the backward pass.

        :param int g: Current dataloader index.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        """
        # Compute the losses
        loss_total = 0
        for k, l in enumerate(self.losses):
            loss = l(x=x, x_net=x_net, y=y, physics=self.physics[g], model=self.model)
            loss_total += self.fact_losses[k] * loss
            self.losses_verbose[k].update(loss.item())
            if len(self.losses) > 1:
                self.log_dict_iter["loss_" + l.name] = self.losses_verbose[k].val

        self.total_loss.update(loss_total.item())

        self.log_dict_iter["training loss"] = self.total_loss.val

        # Backward the total loss
        loss_total.backward()
        self.check_clip_grad()

        # Optimizer step
        self.optimizer.step()

    def get_samples_online(self, iterators, g):
        r"""
        Get the samples for the online measurements.

        In this setting, a new sample is generated at each iteration by calling the physics operator.
        This function returns a dictionary containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data.

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a dictionary containing at least: the ground truth, the measurement, and the current physics operator.
        """
        x, _ = next(
            iterators[g]
        )  # In this case the dataloader outputs also a class label
        x = x.to(self.device)
        physics_cur = self.physics[g]

        y = physics_cur(x)

        return x, y, physics_cur

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
        if self.unsupervised:
            y = next(iterators[g])
            x = None
        else:
            x, y = next(iterators[g])
            if type(x) is list or type(x) is tuple:
                x = [s.to(self.device) for s in x]
            else:
                x = x.to(self.device)

        physics_cur = self.physics[g]

        return x, y, physics_cur

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

    def model_inference(self, y, physics_cur):
        r"""
        Perform the model inference.

        It returns the network reconstruction given the samples.

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :returns: The network reconstruction.
        """
        y = y.to(self.device)
        x_net = self.model(y, physics_cur)
        return x_net

    def forward_pass(self, iterator, g):
        r"""
        Perform the forward pass.

        It returns the ground truth, the measurement, the current physics operator, and the network reconstruction.
        """
        x, y, physics_cur = self.get_samples(iterator, g)

        y = y.to(self.device)

        # Run the forward model
        x_net = self.model_inference(y=y, physics_cur=physics_cur)

        return x, y, physics_cur, x_net

    def train_step(self, epoch, progress_bar):
        r"""
        Train a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration.

        :param int epoch: Current epoch.
        :param tqdm progress_bar: Progress bar.
        :returns: The current physics operator, the ground truth, the measurement, and the network reconstruction.
        """

        progress_bar.set_description(f"Epoch {epoch + 1}")

        self.log_dict_iter = {}

        # random permulation of the dataloaders
        G_perm = np.random.permutation(self.G)

        for g in G_perm:  # for each dataloader

            self.optimizer.zero_grad()

            # Forward step
            x, y, physics_cur, x_net = self.forward_pass(self.train_iterators, g)

            # Backward step
            self.backward_pass(g=g, x=x, y=y, x_net=x_net)

            # Compute the metrics over the batch
            _ = self.batch_metric(x=x, x_net=x_net, y=y, physics=physics_cur)

            # Log metrics
            self.log_metrics()

            progress_bar.set_postfix(self.log_dict_iter)

        return physics_cur, x, y, x_net

    def train(self):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        :returns: The trained model.
        """

        self.setup_train()
        for epoch in range(self.epoch_start, self.epochs):
            self.epoch_eval(epoch)

            self.model.train()

            self.train_iterators = [iter(loader) for loader in self.train_dataloader]
            batches = len(self.train_dataloader[self.G - 1])

            for i in (progress_bar := tqdm(range(batches), disable=not self.verbose)):
                physics_cur, x, y, x_net = self.train_step(epoch, progress_bar)

            self.epoch_wandb_vis(epoch, physics_cur, x=x, y=y, x_net=x_net)

            self.loss_history.append(self.total_loss.avg)

            if self.scheduler:
                self.scheduler.step()

            # Saving the model
            save_model(
                epoch,
                self.model,
                self.optimizer,
                self.ckp_interval,
                self.epochs,
                self.loss_history,
                str(self.save_path),
                eval_psnr=self.eval_psnr if self.perform_eval else None,
            )

        if self.wandb_vis:
            wandb.save("model.h5")

        return self.model

    def validation_step(self, epoch):
        r"""
        Perform validation on the validation set.

        :param int epoch: Current epoch.
        """
        self.model.eval()

        self.val_iterators = [iter(loader) for loader in self.eval_dataloader]
        G = len(self.eval_dataloader)

        metrics = []
        images = []
        losses = []

        for g in range(G):
            dataloader = self.eval_dataloader[g]
            for i in (progress_bar := tqdm(dataloader, disable=not self.verbose)):

                progress_bar.set_description(f"Epoch {epoch + 1} (Validation)")

                with torch.no_grad():

                    # Forward step
                    x, y, physics_cur, x_net = self.forward_pass(self.val_iterators, g)

                    for k, l in enumerate(self.losses):
                        loss = l(
                            x=x, x_net=x_net, y=y, physics=physics_cur, model=self.model
                        )
                        losses.append(self.fact_losses[k] * loss)

                    metric = self.batch_metric(
                        x=x,
                        x_net=x_net,
                        y=y,
                        physics=physics_cur,
                        train=False,
                        log=False,
                    )
                    metrics.append(metric)

                    progress_bar.set_postfix(
                        {
                            "val_loss": torch.Tensor(losses).mean().item(),
                            "val_psnr": torch.Tensor(metrics).mean().item(),
                        }
                    )

            image, _ = self.make_grid_image(physics_cur, x, y, x_net)
            images.append(image)

        mean_metric = torch.Tensor(metrics).mean()
        mean_loss = torch.Tensor(losses).mean()

        return mean_metric, mean_loss, images

    def epoch_eval(self, epoch):
        r"""
        Perform evaluation at the end of each epoch.

        :param int epoch: Current epoch.
        """
        wandb_log_dict_epoch = {"epoch": epoch}

        # perform evaluation every eval_interval epoch
        self.perform_eval = (
            (not self.unsupervised)
            and self.eval_dataloader
            and ((epoch + 1) % self.eval_interval == 0 or epoch + 1 == self.epochs)
        )

        if self.perform_eval:
            mean_val_metric, mean_val_loss, images = self.validation_step(epoch)
            self.eval_psnr.update(mean_val_metric)
            wandb_log_dict_epoch["eval_psnr"] = mean_val_metric
            wandb_log_dict_epoch["eval_loss"] = mean_val_loss

            # wandb logging
            if self.plot_images and self.wandb_vis:
                last_lr = (
                    None if self.scheduler is None else self.scheduler.get_last_lr()[0]
                )
                wandb_log_dict_epoch["learning rate"] = last_lr

                for g in range(self.G):
                    wandb_log_dict_epoch[f"Val images batch (G={g}) "] = wandb.Image(
                        images[g]
                    )

                wandb.log(wandb_log_dict_epoch)


def test(
    model,
    test_dataloader,
    physics,
    device="cpu",
    plot_images=False,
    save_folder="results",
    plot_metrics=False,
    verbose=True,
    plot_only_first_batch=True,
    wandb_vis=False,
    wandb_setup={},
    step=0,
    online_measurements=False,
    plot_measurements=True,
    img_interval=1,
    **kwargs,
):
    r"""
    Tests a reconstruction network.

    This function computes the PSNR of the reconstruction network on the test set,
    and optionally plots the reconstructions as well as the metrics computed along the iterations.
    Note that by default only the first batch is plotted.

    :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param torch.utils.data.DataLoader test_dataloader: Test data loader, which should provide a tuple of (x, y) pairs.
        See :ref:`datasets <datasets>` for more details.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s)
        used by the reconstruction network at test time.
    :param torch.device device: gpu or cpu.
    :param bool plot_images: Plot the ground-truth and estimated images.
    :param str save_folder: Directory in which to save plotted reconstructions.
    :param bool plot_metrics: plot the metrics to be plotted w.r.t iteration.
    :param bool verbose: Output training progress information in the console.
    :param bool plot_only_first_batch: Plot only the first batch of the test set.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    :param dict wandb_setup: Dictionary with the setup for wandb, see https://docs.wandb.ai/quickstart for more details.
    :param int step: Step number for wandb visualization.
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``.
    :param bool plot_measurements: Plot the measurements y. default=True.
    :param int img_interval: how many steps between plotting images
    :returns: A tuple of floats (test_psnr, test_std_psnr, linear_std_psnr, linear_std_psnr) with the PSNR of the
        reconstruction network and a simple linear inverse on the test set.
    """
    save_folder = Path(save_folder)

    psnr_init = []
    psnr_net = []

    model.eval()

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)

    show_operators = 5

    if wandb_vis:
        if wandb.run is None:
            wandb.init(**wandb_setup)
        psnr_data = []

    for g in range(G):
        dataloader = test_dataloader[g]
        if verbose:
            print(f"Processing data of operator {g+1} out of {G}")
        for i, batch in enumerate(tqdm(dataloader, disable=not verbose)):
            with torch.no_grad():
                if online_measurements:
                    (
                        x,
                        _,
                    ) = batch  # In this case the dataloader outputs also a class label
                    x = x.to(device)
                    physics_cur = physics[g]
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        physics_cur.module.noise_model.__init__()
                    else:
                        physics_cur.reset()
                    y = physics_cur(x)
                else:
                    x, y = batch
                    if type(x) is list or type(x) is tuple:
                        x = [s.to(device) for s in x]
                    else:
                        x = x.to(device)
                    physics_cur = physics[g]

                    y = y.to(device)

                if plot_metrics:
                    x1, metrics = model(y, physics_cur, x_gt=x, compute_metrics=True)
                else:
                    x1 = model(y, physics[g])

                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        x_init = physics_cur.module.A_adjoint(y)
                    else:
                        x_init = physics_cur.A_adjoint(y)
                elif hasattr(physics_cur, "A_dagger"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        x_init = physics_cur.module.A_dagger(y)
                    else:
                        x_init = physics_cur.A_dagger(y)
                else:
                    x_init = zeros_like(x)

                cur_psnr_init = cal_psnr(x_init, x)
                cur_psnr = cal_psnr(x1, x)
                psnr_init.append(cur_psnr_init)
                psnr_net.append(cur_psnr)

                if wandb_vis:
                    psnr_data.append([g, i, cur_psnr_init, cur_psnr])

                if plot_images:
                    save_folder_im = (
                        (save_folder / ("G" + str(g))) if G > 1 else save_folder
                    ) / "images"
                    save_folder_im.mkdir(parents=True, exist_ok=True)
                else:
                    save_folder_im = None
                if plot_metrics:
                    save_folder_curve = (
                        (save_folder / ("G" + str(g))) if G > 1 else save_folder
                    ) / "curves"
                    save_folder_curve.mkdir(parents=True, exist_ok=True)

                if (plot_images or wandb_vis) and (step + 1) % img_interval == 0:
                    if g < show_operators:
                        if not plot_only_first_batch or (
                            plot_only_first_batch and i == 0
                        ):
                            if plot_measurements and len(y.shape) == 4:
                                imgs = [y, x_init, x1, x]
                                name_imgs = ["Input", "No learning", "Recons.", "GT"]
                            else:
                                imgs = [x_init, x1, x]
                                name_imgs = ["No learning", "Recons.", "GT"]
                            fig = plot(
                                imgs,
                                titles=name_imgs,
                                save_dir=save_folder_im if plot_images else None,
                                show=plot_images,
                                return_fig=True,
                            )
                            if wandb_vis:
                                wandb.log(
                                    {
                                        f"Test images batch_{i} (G={g}) ": wandb.Image(
                                            fig
                                        )
                                    }
                                )

                if plot_metrics:
                    plot_curves(metrics, save_dir=save_folder_curve, show=True)
                    if wandb_vis:
                        wandb_plot_curves(metrics, batch_idx=i, step=step)

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    linear_psnr = np.mean(psnr_init)
    linear_std_psnr = np.std(psnr_init)
    if verbose:
        print(
            f"Test PSNR: No learning rec.: {linear_psnr:.2f}+-{linear_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. "
        )
    if wandb_vis:
        wandb.log({"Test PSNR": test_psnr}, step=step)

    return test_psnr, test_std_psnr, linear_psnr, linear_std_psnr


def train(*args, **kwargs):
    """
    Alias function for training a model using :class:`deepinv.training_utils.Trainer` class.

    This function creates a Trainer instance and returns the trained model.

    .. warning::

        This function is deprecated and will be removed in future versions. Please use
        :class:`deepinv.training_utils.Trainer` instead.

    :param args: Positional arguments to pass to Trainer constructor.
    :param kwargs: Keyword arguments to pass to Trainer constructor.
    :return: Trained model.
    """
    trainer = Trainer(*args, **kwargs)
    trained_model = trainer.train()
    return trained_model


def train_normalizing_flow(
    model,
    dataloader,
    epochs=10,
    learning_rate=1e-3,
    device="cpu",
    jittering=1 / 255.0,
    verbose=False,
):
    r"""
    Trains a normalizing flow.

    Uses the Adam optimizer and the forward Kullback-Leibler (maximum likelihood) loss function given by

    .. math::
        \mathcal{L}(\theta)=\mathrm{KL}(P_X,{\mathcal{T}_\theta}_\#P_Z)=\mathbb{E}_{x\sim P_X}[p_{{\mathcal{T}_\theta}_\#P_Z}(x)]+\mathrm{const},

    where :math:`\mathcal{T}_\theta` is the normalizing flow with parameters :math:`\theta`, latent distribution :math:`P_Z`, data distribution :math:`P_X` and push-forward measure :math:`{\mathcal{T}_\theta}_\#P_Z`.

    :param torch.nn.Module model: Normalizing flow in the same format as in the `FrEIA <https://vislearn.github.io/FrEIA/_build/html/index.html>`_ framework (i.e., the forward method takes the data and the flag rev (default False) where rev=True indicates calling the inverse; the forward method returns the output of the network and the log-determinant of the Jacobian of the flow.
    :param torch.utils.data.DataLoader dataloader: contains training data.
    :param int epochs: number of epochs
    :param float learning_rate: learning rate
    :param str device: used device
    :param float jittering: adds uniform noise of range [-jittering,jittering] to the training data.
        This is a common trick for stabilizing the training of normalizing flows and to avoid overfitting
    :param bool verbose: Whether printing progress.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        mean_loss = 0.0
        for i, (x, _) in enumerate(
            progress_bar := tqdm(dataloader, disable=not verbose)
        ):
            x = x.to(device)
            x = x + jittering * (2 * torch.rand_like(x) - 1)
            optimizer.zero_grad()
            invs, jac_inv = model(x)
            loss = torch.mean(
                0.5 * torch.sum(invs.view(invs.shape[0], -1) ** 2, -1)
                - jac_inv.view(invs.shape[0])
            )
            loss.backward()
            optimizer.step()
            mean_loss = mean_loss / (i + 1) * i + loss.item() / (i + 1)
            progress_bar.set_description(
                "Epoch {}, Mean Loss: {:.2f}, Loss {:.2f}".format(
                    epoch + 1, mean_loss, loss.item()
                )
            )
