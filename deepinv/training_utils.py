import torchvision.utils
from deepinv.utils import (
    save_model,
    AverageMeter,
    get_timestamp,
    cal_psnr,
)
from deepinv.utils import plot, plot_curves, wandb_plot_curves, rescale_img, zeros_like
import numpy as np
from tqdm import tqdm
import torch
import wandb
from pathlib import Path


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
    save_path=".",
    verbose=False,
    unsupervised=False,
    plot_images=False,
    plot_metrics=False,
    wandb_vis=False,
    wandb_setup={},
    online_measurements=False,
    plot_measurements=True,
    check_grad=False,
    ckpt_pretrained=None,
    fact_losses=None,
    freq_plot=1,
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
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
         ``physics(x)``. This results in a wider range of measurements if the physics' parameters, such as
         parameters of the forward operator or noise realizations, can change between each sample; these are updated
         with the ``physics.reset()`` method. If ``online_measurements=False``, the measurements are loaded from the training dataset
    :param bool plot_measurements: Plot the measurements y. default=True.
    :param bool check_grad: Check the gradient norm at each iteration.
    :param str ckpt_pretrained: path of the pretrained checkpoint. If None, no pretrained checkpoint is loaded.
    :param list fact_losses: List of factors to multiply the losses. If None, all losses are multiplied by 1.
    :param int freq_plot: Frequency of plotting images to wandb. If 1, plots at each epoch.
    :returns: Trained model.
    """
    save_path = Path(save_path)

    # wandb initialiation
    if wandb_vis:
        if wandb.run is None:
            wandb.init(**wandb_setup)

    # set the different metrics
    meters = []
    total_loss = AverageMeter("loss", ":.2e")
    meters.append(total_loss)
    if not isinstance(losses, list) or isinstance(losses, tuple):
        losses = [losses]
    if fact_losses is None:
        fact_losses = [1] * len(losses)
    losses_verbose = [AverageMeter("Loss_" + l.name, ":.2e") for l in losses]
    for loss in losses_verbose:
        meters.append(loss)
    train_psnr = AverageMeter("Train_psnr_model", ":.2f")
    meters.append(train_psnr)
    if eval_dataloader:
        eval_psnr = AverageMeter("Eval_psnr_model", ":.2f")
        meters.append(eval_psnr)
    if check_grad:
        check_grad_val = AverageMeter("Gradient norm", ":.2e")
        meters.append(check_grad_val)

    save_path = f"{save_path}/{get_timestamp()}"

    # count the overall training parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {params} trainable parameters")

    # make physics and data_loaders of list type
    if type(physics) is not list:
        physics = [physics]
    if type(train_dataloader) is not list:
        train_dataloader = [train_dataloader]
    if eval_dataloader and type(eval_dataloader) is not list:
        eval_dataloader = [eval_dataloader]

    G = len(train_dataloader)

    loss_history = []

    log_dict = {}

    epoch_start = 0
    if ckpt_pretrained is not None:
        checkpoint = torch.load(ckpt_pretrained)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_start = checkpoint["epoch"]

    for epoch in range(epoch_start, epochs):
        ### Evaluation

        if wandb_vis:
            wandb_log_dict_epoch = {"epoch": epoch}

        # perform evaluation every eval_interval epoch
        perform_eval = (
            (not unsupervised)
            and eval_dataloader
            and ((epoch + 1) % eval_interval == 0 or epoch + 1 == epochs)
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
                online_measurements=online_measurements,
            )
            eval_psnr.update(test_psnr)
            log_dict["eval_psnr"] = test_psnr
            if wandb_vis:
                wandb_log_dict_epoch["eval_psnr"] = test_psnr

        # wandb logging
        if wandb_vis:
            last_lr = None if scheduler is None else scheduler.get_last_lr()[0]
            wandb_log_dict_epoch["learning rate"] = last_lr

            wandb.log(wandb_log_dict_epoch)

        ### Training

        model.train()

        for meter in meters:
            meter.reset()  # reset the metric at each epoch

        iterators = [iter(loader) for loader in train_dataloader]
        batches = len(train_dataloader[G - 1])

        for i in (progress_bar := tqdm(range(batches), disable=not verbose)):
            progress_bar.set_description(f"Epoch {epoch + 1}")

            if wandb_vis:
                wandb_log_dict_iter = {}

            # random permulation of the dataloaders
            G_perm = np.random.permutation(G)

            for g in G_perm:  # for each dataloader
                if online_measurements:  # the measurements y are created on-the-fly
                    x, _ = next(
                        iterators[g]
                    )  # In this case the dataloader outputs also a class label
                    x = x.to(device)
                    physics_cur = physics[g]

                    if isinstance(physics_cur, torch.nn.DataParallel):
                        physics_cur.module.noise_model.__init__()
                    else:
                        physics_cur.reset()

                    y = physics_cur(x)

                else:  # the measurements y were pre-computed
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

                # run the forward model
                x_net = model(y, physics_cur)

                # compute the losses
                loss_total = 0
                for k, l in enumerate(losses):
                    loss = l(x=x, x_net=x_net, y=y, physics=physics[g], model=model)
                    loss_total += fact_losses[k] * loss
                    losses_verbose[k].update(loss.item())
                    if len(losses) > 1:
                        log_dict["loss_" + l.name] = losses_verbose[k].avg
                        if wandb_vis:
                            wandb_log_dict_iter["loss_" + l.name] = loss.item()
                if wandb_vis:
                    wandb_log_dict_iter["training loss"] = loss_total.item()
                total_loss.update(loss_total.item())
                log_dict["total_loss"] = total_loss.avg

                # backward the total loss
                loss_total.backward()

                # gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if check_grad:
                    # from https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/7
                    grads = [
                        param.grad.detach().flatten()
                        for param in model.parameters()
                        if param.grad is not None
                    ]
                    norm_grads = torch.cat(grads).norm()
                    wandb_log_dict_iter["gradient norm"] = norm_grads.item()
                    check_grad_val.update(norm_grads.item())

                # optimize step
                optimizer.step()

                # training psnr and logging
                if not unsupervised:
                    with torch.no_grad():
                        psnr = cal_psnr(x_net, x)
                        train_psnr.update(psnr)
                        if wandb_vis:
                            wandb_log_dict_iter["train_psnr"] = psnr
                            wandb.log(wandb_log_dict_iter)
                        log_dict["train_psnr"] = train_psnr.avg

                progress_bar.set_postfix(log_dict)

        # wandb plotting of training images
        if wandb_vis:
            # log average training metrics
            log_dict_post_epoch = {}
            log_dict_post_epoch["mean training loss"] = total_loss.avg
            log_dict_post_epoch["mean training psnr"] = train_psnr.avg
            if check_grad:
                log_dict_post_epoch["mean gradient norm"] = check_grad_val.avg

            with torch.no_grad():
                if plot_measurements and y.shape != x.shape:
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
                if epoch % freq_plot == 0:
                    images = wandb.Image(
                        grid_image,
                        caption=caption,
                    )
                    log_dict_post_epoch["Training samples"] = images

        if wandb_vis:
            wandb.log(log_dict_post_epoch)

        loss_history.append(total_loss.avg)

        if scheduler:
            scheduler.step()

        # Saving the model
        save_model(
            epoch,
            model,
            optimizer,
            ckp_interval,
            epochs,
            loss_history,
            str(save_path),
            eval_psnr=eval_psnr if perform_eval else None,
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
    online_measurements=False,
    plot_measurements=True,
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

                if plot_images or wandb_vis:
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
