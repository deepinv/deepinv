from deepinv.utils import (
    save_model,
    AverageMeter,
    ProgressMeter,
    get_timestamp,
    cal_psnr,
)
from deepinv.utils import plot, torch2cpu, wandb_imgs
import numpy as np
from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({"font.size": 17})
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.style.use("seaborn-darkgrid")
use_tex = matplotlib.checkdep_usetex(True)
if use_tex:
    plt.rcParams["text.usetex"] = True


def train(
    model,
    train_dataloader,
    epochs,
    losses,
    eval_dataloader=None,
    physics=None,
    optimizer=None,
    scheduler=None,
    device="cpu",
    ckp_interval=1,
    eval_interval=1,
    log_interval=1,
    save_path=".",
    verbose=False,
    unsupervised=False,
    plot_images=False,
    save_images=False,
    plot_metrics=False,
    wandb_vis=False,
    n_plot_max_wandb=8,
):
    r"""
    Trains a reconstruction network.


    :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param torch.utils.data.DataLoader train_dataloader: Train dataloader.
    :param int epochs: Number of training epochs.
    :param torch.nn.Module, list of torch.nn.Module losses: Loss or list of losses used for training the model.
    :param torch.utils.data.DataLoader eval_dataloader: Evaluation dataloader.
    :param deepinv.physics.Physics physics: Forward operator containing the physics of the inverse problem.
    :param torch.nn.optim optimizer: Torch optimizer for training the network.
    :param torch.nn.optim scheduler: Torch scheduler for changing the learning rate across iterations.
    :param torch.device device: gpu or cpu.
    :param int ckp_interval: The model is saved every ``ckp_interval`` epochs.
    :param int eval_interval: Number of epochs between each evaluation of the model on the evaluation set.
    :param str save_path: Directory in which to save the trained model.
    :param bool verbose: Output training progress information in the console.
    :param bool unsupervised: Train an unsupervised network, i.e., uses only measurement vectors y for training.
    :param bool plot_images: Plots reconstructions every ``ckp_interval`` epochs.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    """
    save_path = Path(save_path)

    if wandb_vis:
        wandb.init()

    if not isinstance(losses, list) or isinstance(losses, tuple):
        losses = [losses]

    loss_meter = AverageMeter("loss", ":.2e")
    meters = [loss_meter]
    eval_psnr_net = []

    losses_verbose = [AverageMeter("Loss_" + l.name, ":.2e") for l in losses]
    train_psnr_net = AverageMeter("Train_psnr_model", ":.2f")
    if eval_dataloader:
        eval_psnr_net = AverageMeter("Eval_psnr_model", ":.2f")

    for loss in losses_verbose:
        meters.append(loss)

    meters.append(train_psnr_net)
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
        if eval_dataloader:
            eval_psnr_net.reset()
        train_psnr_net.reset()
        iterators = [iter(loader) for loader in train_dataloader]
        batches = len(train_dataloader[G - 1])
        for i in range(batches):
            G_perm = np.random.permutation(G)

            for g in G_perm:
                if unsupervised:
                    y = next(iterators[g])
                else:
                    x, y = next(iterators[g])

                    if type(x) is list or type(x) is tuple:
                        x = [s.to(device) for s in x]
                    else:
                        x = x.to(device)

                y = y.to(device)

                optimizer.zero_grad()

                x1 = model(y, physics[g])  # Requires grad ok

                loss_total = 0
                for k, l in enumerate(losses):
                    if l.name in ["mc"]:
                        loss = l(y, x1, physics[g])
                    elif l.name in ["ms"]:
                        loss = l(y, physics[g], model)
                    elif l.name in ["jsn"]:
                        loss = l(x1, y)
                    elif not unsupervised and l.name in ["sup"]:
                        loss = l(x1, x)
                    elif l.name in ["moi"]:
                        loss = l(x1, physics, model)
                    elif l.name in ["tv"]:
                        loss = l(x1)
                    elif l.name.startswith("Sure"):
                        loss = l(y, x1, physics[g], model)
                    elif l.name in ["ei", "rei"]:
                        loss = l(x1, physics[g], model)
                    else:
                        raise Exception(
                            "The loss used is not recognized by the train function."
                        )
                    loss_total += loss

                    losses_verbose[k].update(loss.item())

                loss_meter.update(loss_total.item())

                if (not unsupervised) and verbose:
                    train_psnr_net.update(cal_psnr(x1, x))

                loss_total.backward()
                optimizer.step()

        if (
            (not unsupervised)
            and eval_dataloader
            and ((epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs)
        ):
            test_psnr, _, _, _ = test(
                model,
                eval_dataloader,
                physics,
                device,
                verbose=False,
                plot_images=plot_images,
                save_images=save_images,
                plot_metrics=plot_metrics,
                wandb_vis=wandb_vis,
                step=epoch,
                n_plot_max_wandb=n_plot_max_wandb,
            )

            eval_psnr_net.update(test_psnr)

        if scheduler:
            scheduler.step()

        loss_history.append(loss_total.detach().cpu().numpy())

        if wandb_vis:
            wandb.log({"training loss": loss_total}, step=epoch)

        if (epoch + 1) % log_interval == 0:
            progress.display(epoch + 1)

        save_model(
            epoch, model, optimizer, ckp_interval, epochs, loss_history, str(save_path)
        )

    if wandb_vis:
        wandb.save("model.h5")

    return model


def test(
    model,
    test_dataloader,
    physics,
    device=torch.device(f"cuda:0"),
    plot_images=False,
    save_images=True,
    save_folder="results",
    plot_metrics=False,
    verbose=True,
    wandb_vis=False,
    step=0,
    n_plot_max_wandb=8,
    **kwargs,
):
    r"""
    Tests a reconstruction network.

    :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param torch.utils.data.DataLoader test_dataloader:
    :param deepinv.physics.Physics physics:
    :param torch.device device: gpu or cpu.
    :param bool plot_images: Plot the ground-truth and estimated images.
    :param bool save_images: Save the images.
    :param str save_folder: Directory in which to save plotted reconstructions.
    :param bool plot_metrics: plot the metrics to be plotted w.r.t iteration.
    :param bool verbose: Output training progress information in the console.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    """
    save_folder = Path(save_folder)

    psnr_init = []
    psnr_net = []

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)
    imgs = []

    show_operators = 5

    if wandb_vis:
        wandb.init()
        psnr_data = []

    for g in range(G):
        dataloader = test_dataloader[g]
        if verbose:
            print(f"Processing data of operator {g+1} out of {G}")
        for i, (x, y) in enumerate(tqdm(dataloader, disable=not verbose)):
            if type(x) is list or type(x) is tuple:
                x = [s.to(device) for s in x]
            else:
                x = x.to(device)

            y = y.to(device)

            with torch.no_grad():
                if plot_metrics:
                    output_model = model(y, physics[g], x, **kwargs)
                    if len(output_model) == 1:
                        plot_metrics = False
                        print(
                            "plot_metrics is set to True but model does not returns metrics"
                        )
                        x1 = model(y, physics[g], **kwargs)
                    else:
                        x1, metrics = output_model
                else:
                    x1 = model(y, physics[g], **kwargs)

                if hasattr(model, "custom_init") and model.custom_init:
                    x_init = model.custom_init(y)
                else:
                    x_init = physics[g].A_adjoint(y)
            cur_psnr_init = cal_psnr(x_init, x)
            cur_psnr = cal_psnr(x1, x)

            psnr_init.append(cur_psnr_init)
            psnr_net.append(cur_psnr)

            if wandb_vis:
                psnr_data.append([g, i, cur_psnr_init, cur_psnr])

            if save_images or plot_metrics:
                save_folder_G = save_folder / ("G" + str(g))
                save_folder_G.mkdir(parents=True, exist_ok=True)

            if plot_images or save_images or wandb_vis:
                if g < show_operators:
                    imgs = [x_init, x1, x]
                    name_imgs = ["Linear", "Recons.", "GT"]

                    if save_images:
                        for img, name_im in zip(imgs, name_imgs):
                            plt.imsave(
                                save_folder_G / (name_im + "_" + str(i) + ".png"),
                                img,
                                cmap="gray")
                    if plot_images:
                        plot(imgs, titles=name_imgs)
                    if wandb_vis:
                        n_plot = min(n_plot_max_wandb, len(x))
                        captions = [
                            "Input",
                            f"Init (or Linear) PSNR:{cur_psnr_init:.2f}",
                            f"Estimated PSNR:{cur_psnr:.2f}",
                            "Ground Truth",
                        ]
                        imgs = wandb_imgs(imgs, captions=captions, n_plot=n_plot)
                        wandb.log({f"Images batch_{i} (G={g}) ": imgs}, step=step)

            if plot_metrics:
                for metric_name, metric_val in zip(metrics.keys(), metrics.values()):
                    if len(metric_val) > 0:
                        batch_size, n_iter = len(metric_val), len(metric_val[0])

                        if wandb_vis:
                            wandb.log(
                                {
                                    f"{metric_name} batch {i}": wandb.plot.line_series(
                                        xs=range(n_iter),
                                        ys=metric_val,
                                        keys=[f"image {j}" for j in range(batch_size)],
                                        title=f"{metric_name} batch {i}",
                                        xname="iteration",
                                    )
                                },
                                step=step,
                            )

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    init_psnr = np.mean(psnr_init)
    init_std_psnr = np.std(psnr_init)
    if verbose:
        print(
            f"Test PSNR: Init: {init_psnr:.2f}+-{init_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. "
        )
    if wandb_vis:
        wandb.log({"Test PSNR": test_psnr}, step=step)

    return test_psnr, test_std_psnr, init_psnr, init_std_psnr
