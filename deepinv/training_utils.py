import torchvision.utils

from torch.profiler import profile, record_function, ProfilerActivity  # to delete

from deepinv.utils import (
    save_model,
    AverageMeter,
    ProgressMeter,
    get_timestamp,
    cal_psnr,
)
from deepinv.utils import plot, plot_curves, wandb_imgs, wandb_plot_curves
import numpy as np
from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({"font.size": 17})
matplotlib.rcParams["lines.linewidth"] = 2
plt.rcParams["text.usetex"] = True


def train(
    model,
    train_dataloader,
    epochs,
    losses,
    eval_dataloader=None,
    physics=None,
    optimizer=None,
    grad_clip=None,
    scheduler=None,
    device="cpu",
    ckp_interval=1,
    eval_interval=1,
    log_interval=1,
    save_path=".",
    verbose=False,
    unsupervised=False,
    plot_images=False,
    plot_metrics=False,
    wandb_vis=False,
    wandb_setup={},
    n_plot_max_wandb=8,
    online_measurements=False,
):
    r"""
    Trains a reconstruction network.


    .. note::

        The losses can be chosen from :ref:`the libraries' training losses <loss>`, or can be a custom loss function,
        as long as it takes as input ``(x, x_net, y, physics, model)`` and returns a scalar, where ``x`` is the ground
        reconstruction, ``x_net`` is the network reconstruction :math:`\inversef{y, A}`,
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
    :param int n_plot_max_wandb: Maximum number of images to plot in wandb visualization.
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``. This results in a wider range of measurements if the physics' parameters, such as
         parameters of the forward operator or noise realizations, can change between each sample; these are updated
         with the ``physics.reset()`` method. If ``online_measurements=False``, the measurements are generated
        offline before training and saved in memory.
    :returns: Trained model.
    """
    save_path = Path(save_path)

    if wandb_vis:
        if wandb.run is None:
            wandb.init(**wandb_setup)

    if not isinstance(losses, list) or isinstance(losses, tuple):
        losses = [losses]

    loss_meter = AverageMeter("loss", ":.2e")
    meters = [loss_meter]
    eval_psnr_net = []

    losses_verbose = [AverageMeter("Loss_" + l.name, ":.2e") for l in losses]
    train_psnr = AverageMeter("Train_psnr_model", ":.2f")
    if eval_dataloader:
        eval_psnr_net = AverageMeter("Eval_psnr_model", ":.2f")

    for loss in losses_verbose:
        meters.append(loss)

    meters.append(train_psnr)
    if eval_dataloader:
        meters.append(eval_psnr_net)

    progress = ProgressMeter(epochs, meters)

    save_path = f"{save_path}/{get_timestamp()}"

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {params} trainable parameters")

    if type(physics) is not list:
        physics = [physics]

    if type(losses) is not list:
        losses = [losses]

    if type(train_dataloader) is not list:
        train_dataloader = [train_dataloader]

    if eval_dataloader and type(eval_dataloader) is not list:
        eval_dataloader = [eval_dataloader]

    G = len(train_dataloader)

    loss_history = []

    for epoch in range(epochs):
        model.train()

        for meter in meters:
            meter.reset()
        iterators = [iter(loader) for loader in train_dataloader]
        batches = len(train_dataloader[G - 1])

        for i in range(batches):
            G_perm = np.random.permutation(G)

            for g in G_perm:
                if online_measurements:
                    x, _ = next(
                        iterators[g]
                    )  # In this case the dataloader outputs also a class label
                    x = x.to(device)
                    physics_cur = physics[g]
                    physics_cur.reset()
                    y = physics_cur(x)
                else:
                    if unsupervised:
                        y = next(iterators[g])
                        x = None
                    else:
                        x, y = next(iterators[g])

                        if type(x) is list or type(x) is tuple:
                            x = [s.to(device) for s in x]
                        else:
                            x = x.to(device)

                    physics_cur = physics[g]

                y = y.to(device)

                optimizer.zero_grad()

                x_net = model(y, physics_cur)

                loss_total = 0
                for k, l in enumerate(losses):
                    loss = l(x=x, x_net=x_net, y=y, physics=physics[g], model=model)
                    loss_total += loss
                    losses_verbose[k].update(loss.item())

                loss_total.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                if wandb_vis:
                    wandb.log({"training loss": loss_total.item()})

                loss_meter.update(loss_total.item())

                if not unsupervised:
                    train_psnr.update(cal_psnr(x_net, x))

        if (
            wandb_vis
        ):  # Note that this may not be 16 images because the last batch may be smaller
            in_image = physics_cur.A_adjoint(y)
            vis_array = torch.cat((in_image, x_net, x), dim=0)
            vis_array = torch.clip(vis_array, 0, 1)
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])
            images = wandb.Image(
                grid_image, caption="Top: Input, Middle: Output, Bottom: target"
            )
            wandb.log({"Training samples": images})

        check = (
            (not unsupervised)
            and eval_dataloader is not None
            and ((epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs)
        )

        perform_eval = (
            (not unsupervised)
            and eval_dataloader
            and ((epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs)
        )
        if perform_eval:
            test_psnr, _, _, _ = test(
                model,
                eval_dataloader,
                physics,
                device,
                verbose=False,
                plot_images=plot_images,
                plot_metrics=plot_metrics,
                wandb_vis=wandb_vis,
                wandb_setup=wandb_setup,
                step=epoch,
                n_plot_max_wandb=n_plot_max_wandb,
                online_measurements=online_measurements,
            )

            eval_psnr_net.update(test_psnr)

        if scheduler:
            scheduler.step()

        loss_history.append(loss_meter.avg)

        if wandb_vis:
            last_lr = None if scheduler is None else scheduler.get_last_lr()[0]
            if not unsupervised and perform_eval:
                wandb.log(
                    {
                        "mean training loss": loss_meter.avg,
                        "mean training psnr": train_psnr.avg,
                        "mean eval psnr": eval_psnr_net.avg,
                        "epoch": epoch,
                        "learning rate": last_lr,
                    }
                )
            else:
                wandb.log(
                    {
                        "mean training loss": loss_meter.avg,
                        "mean training psnr": train_psnr.avg,
                        "epoch": epoch,
                        "learning rate": last_lr,
                    }
                )

        if (epoch + 1) % log_interval == 0:
            progress.display(epoch + 1)

        save_model(
            epoch,
            model,
            optimizer,
            ckp_interval,
            epochs,
            loss_history,
            str(save_path),
            eval_psnr_net,
        )

    if wandb_vis:
        wandb.save("model.h5")

    return model


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
    n_plot_max_wandb=8,
    online_measurements=False,
    **kwargs,
):
    r"""
    Tests a reconstruction network.

    This function computes the PSNR of the reconstruction network on the test set,
    and optionally plots the reconstructions as well as the metrics computed along the iterations.
    Note that by default only the batch is plotted.

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
    :param int n_plot_max_wandb: Maximum number of images to plot in wandb visualization.
    :returns: A tuple of floats (test_psnr, test_std_psnr, linear_std_psnr, linear_std_psnr) with the PSNR of the
        reconstruction network and a simple linear inverse on the test set.
    """
    save_folder = Path(save_folder)

    psnr_lin = []  # psnr with linear reconstruction A^Ty (for comparison)
    psnr_net = []  # psnr with model

    model.eval()

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)  # we can handle multiple operators.

    show_operators = 5  # maximum numbers of operators (test_dataloaders) to visualize.

    if wandb_vis:  # initialize wandb visualization.
        if wandb.run is None:
            wandb.init(**wandb_setup)
        psnr_data = []

    for g in range(G):  # for each operator
        dataloader = test_dataloader[g]
        if verbose and G > 1:
            print(f"Processing data of operator {g+1} out of {G}")
        for i, batch in enumerate(
            tqdm(dataloader, disable=not verbose)
        ):  # for each batch
            if verbose:
                print(f"Processing batch {i} out of {len(dataloader)}")
            if online_measurements:
                x, _ = batch  # In this case the dataloader outputs also a class label
                x = x.to(device)
                physics_cur = physics[g]
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

            with torch.no_grad():  # run the model
                if plot_metrics:
                    x1, metrics = model(y, physics_cur, x_gt=x, compute_metrics=True)
                else:
                    x1 = model(y, physics[g])

            x_lin = physics_cur.A_adjoint(y)  # linear reconstruction
            cur_psnr_lin = cal_psnr(x_lin, x)
            cur_psnr = cal_psnr(x1, x)
            psnr_lin.append(cur_psnr_lin)
            psnr_net.append(cur_psnr)

            if wandb_vis:
                psnr_data.append([g, i, cur_psnr_lin, cur_psnr])

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

            if plot_images or wandb_vis:
                if g < show_operators:
                    if not plot_only_first_batch or (plot_only_first_batch and i == 0):
                        if len(y.shape) == 4:
                            imgs = [y, x_lin, x1, x]
                            name_imgs = ["Input", "Linear", "Recons.", "GT"]
                        else:
                            imgs = [x_lin, x1, x]
                            name_imgs = ["Linear", "Recons.", "GT"]
                        if plot_images:
                            plot(
                                imgs,
                                titles=name_imgs,
                                save_dir=save_folder_im,
                                show=True,
                            )
                        if wandb_vis:
                            vis_array = torch.cat(imgs, dim=0)
                            vis_array = torch.clip(vis_array, 0, 1)
                            grid_image = torchvision.utils.make_grid(vis_array, nrow=4)
                            images = wandb.Image(
                                grid_image,
                                caption="Input (Left) / Backprojection (optional) / Output / Target (optional)",
                            )
                            wandb.log({f"Test images batch_{i} (G={g}) ": images})

            if plot_metrics:
                plot_curves(metrics, save_dir=save_folder_curve, show=True)
                if wandb_vis:
                    wandb_plot_curves(metrics, batch_idx=i, step=step)

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    linear_psnr = np.mean(psnr_lin)
    linear_std_psnr = np.std(psnr_lin)
    if verbose:
        print(
            f"Average test PSNR: Linear rec.: {linear_psnr:.2f}+-{linear_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. "
        )
    if wandb_vis:
        wandb.log({"Test PSNR": test_psnr}, step=step)

    return test_psnr, test_std_psnr, linear_psnr, linear_std_psnr
