import os
import math
from deepinv.utils import (
    save_model,
    AverageMeter,
    ProgressMeter,
    get_timestamp,
    cal_psnr,
    investigate_model,
)
from deepinv.utils import plot_debug, torch2cpu, im_save, make_grid
import numpy as np
from tqdm import tqdm
import torch
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib

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
    ckp_interval=100,
    eval_interval=1,
    save_path=".",
    verbose=False,
    unsupervised=False,
    plot_images=False,
    wandb_vis=False,
    debug=False,
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
    :param bool debug: TODO
    """

    if wandb_vis:
        wandb.watch(model)

    loss_meter = AverageMeter("loss", ":.2e")
    meters = [loss_meter]
    losses_verbose = []
    train_psnr_net = []
    train_psnr_linear = []
    eval_psnr_net = []
    eval_psnr_linear = []

    if verbose:
        losses_verbose = [AverageMeter("loss_" + l.name, ":.2e") for l in losses]
        train_psnr_net = AverageMeter("train_psnr_net", ":.2f")
        train_psnr_linear = AverageMeter("train_psnr_linear", ":.2f")
        eval_psnr_net = AverageMeter("eval_psnr_net", ":.2f")
        eval_psnr_linear = AverageMeter("eval_psnr_linear", ":.2f")

        for loss in losses_verbose:
            meters.append(loss)
        meters.append(train_psnr_linear)
        meters.append(train_psnr_net)
        if eval_dataloader:
            meters.append(eval_psnr_linear)
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

                x1 = model(y, physics[g])  # Requires grad ok

                loss_total = 0
                for k, l in enumerate(losses):
                    if l.name in ["mc"]:
                        loss = l(y, x1, physics[g])
                    elif l.name in ["ms"]:
                        loss = l(y, physics[g], model)
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

                    if verbose:
                        losses_verbose[k].update(loss.item())

                loss_meter.update(loss_total.item())

                if i == 0 and g == 0 and plot_images and epoch % 499 == 0:
                    imgs = [
                        physics[g].A_adjoint(y)[0, :, :, :].unsqueeze(0),
                        x1[0, :, :, :].unsqueeze(0),
                    ]
                    titles = ["Linear Inv.", "Estimated"]
                    if not unsupervised:
                        imgs.append(x[0, :, :, :].unsqueeze(0))
                        titles.append("Ground Truth")
                    plot_debug(imgs, titles=titles)

                if (not unsupervised) and verbose:
                    train_psnr_linear.update(cal_psnr(physics[g].A_adjoint(y), x))
                    train_psnr_net.update(cal_psnr(x1, x))

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                if debug and i == 0:
                    investigate_model(model)

        if (not unsupervised) and eval_dataloader and (epoch + 1) % eval_interval == 0:
            test_psnr, test_std_psnr, pinv_psnr, pinv_std_psnr = test(
                model,
                eval_dataloader,
                physics,
                device,
                verbose=False,
                wandb_vis=wandb_vis,
            )
            if verbose:
                eval_psnr_linear.update(test_psnr)
                eval_psnr_net.update(pinv_psnr)

        if scheduler:
            scheduler.step()

        loss_history.append(loss_total.detach().cpu().numpy())

        if wandb_vis:
            wandb.log({"training loss": loss_total})

        progress.display(epoch + 1)
        save_model(
            epoch, model, optimizer, ckp_interval, epochs, loss_history, save_path
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
    plot_input=True,
    save_folder="results",
    plot_metrics=False,
    verbose=True,
    wandb_vis=False,
    **kwargs,
):
    r"""
    Tests a reconstruction network.

    :param torch.nn.Module, deepinv.models.ArtifactRemoval model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network.
    :param torch.utils.data.DataLoader test_dataloader:
    :param deepinv.physics.Physics physics:
    :param torch.device device: gpu or cpu.
    :param bool plot_images: Plot and save estimated images.
    :param bool plot_input: Plot the input image along with the estimated images.
    :param str save_folder: Directory in which to save plotted reconstructions.
    :param bool plot_metrics: the model returns a dictionary with metrics to be plotted w.r.t iteration.
    :param bool verbose: Output training progress information in the console.
    :param bool wandb_vis: Use Weights & Biases visualization, see https://wandb.ai/ for more details.
    """

    psnr_linear = []
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
        for i, (x, y) in enumerate(tqdm(dataloader)):
            if type(x) is list or type(x) is tuple:
                x = [s.to(device) for s in x]
            else:
                x = x.to(device)

            y = physics[g](x)

            # y = y.to(device)

            with torch.no_grad():
                if plot_metrics:
                    output_model = model(y, physics[g], x, **kwargs)
                    assert (
                        len(output_model) == 2
                    ), "plot_metrics is set to True but model does not returns metrics"
                    x1, metrics = output_model
                else:
                    x1 = model(y, physics[g], **kwargs)

            cur_psnr_linear = cal_psnr(physics[g].A_adjoint(y), x)
            cur_psnr = cal_psnr(x1, x)
            psnr_linear.append(cur_psnr_linear)
            psnr_net.append(cur_psnr)
            if verbose:
                print(
                    f"Test PSNR: Linear Inv: {cur_psnr_linear:.2f}| Model: {cur_psnr:.2f} dB. "
                )
            if wandb_vis:
                psnr_data.append([g, i, cur_psnr_linear, cur_psnr])

            if plot_images or plot_metrics:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_folder_G = save_folder + "G" + str(g)
                if not os.path.exists(save_folder_G):
                    os.makedirs(save_folder_G)

            if plot_images:
                if g < show_operators:
                    imgs = []
                    name_imgs = []
                    xlin = physics[g].A_adjoint(y)
                    if len(y[0].shape) == 3:
                        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
                        name_imgs.append("y")
                    imgs.append(torch2cpu(xlin[0, :, :, :].unsqueeze(0)))
                    name_imgs.append("xlin")
                    imgs.append(torch2cpu(x1[0, :, :, :].unsqueeze(0)))
                    name_imgs.append("xest")
                    imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))
                    name_imgs.append("x")
                    for img, name_im in zip(imgs, name_imgs):
                        im_save(
                            os.path.join(
                                save_folder_G, name_im + "_" + str(i) + ".png"
                            ),
                            img,
                        )
                    if wandb_vis:
                        n_plot = min(8, len(x))
                        imgs = []
                        if plot_input:
                            imgs.append(
                                wandb.Image(
                                    make_grid(
                                        y[:n_plot], nrow=int(math.sqrt(n_plot)) + 1
                                    ),
                                    caption="Input",
                                )
                            )
                        imgs.append(
                            wandb.Image(
                                make_grid(
                                    xlin[:n_plot], nrow=int(math.sqrt(n_plot)) + 1
                                ),
                                caption=f"Linear PSNR:{cur_psnr_linear:.2f}",
                            )
                        )
                        imgs.append(
                            wandb.Image(
                                make_grid(x1[:n_plot], nrow=int(math.sqrt(n_plot)) + 1),
                                caption=f"Estimated PSNR:{cur_psnr:.2f}",
                            )
                        )
                        imgs.append(
                            wandb.Image(
                                make_grid(x[:n_plot], nrow=int(math.sqrt(n_plot)) + 1),
                                caption="Ground Truth",
                            )
                        )
                        wandb.log({f"Image (G={g}) ": imgs}, step=i)

            if plot_metrics:
                for metric_name, metric_val in zip(metrics.keys(), metrics.values()):
                    plt.figure()
                    fig, ax = plt.subplots()
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)
                    plt.plot(metric_val, "o-")
                    plt.xlabel("iteration")
                    plt.ylabel(metric_name)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.savefig(
                        os.path.join(
                            save_folder_G, metric_name + "_" + "im_" + str(i) + ".png"
                        ),
                        bbox_inches="tight",
                    )
                    if wandb_vis:
                        wandb.log({f"Plot {metric_name}": fig}, step=i)

    test_psnr = np.mean(psnr_net)
    test_std_psnr = np.std(psnr_net)
    pinv_psnr = np.mean(psnr_linear)
    pinv_std_psnr = np.std(psnr_linear)
    if verbose:
        print(
            f"Test PSNR: Linear Inv: {pinv_psnr:.2f}+-{pinv_std_psnr:.2f} dB | Model: {test_psnr:.2f}+-{test_std_psnr:.2f} dB. "
        )
    if wandb_vis:
        table_psnr = wandb.Table(
            data=np.array(psnr_data),
            columns=["operator", "image", "Linear PSNR", "Estimated PSNR"],
        )
        wandb.log({"table_psnr": table_psnr})

    return test_psnr, test_std_psnr, pinv_psnr, pinv_std_psnr
