from deepinv.utils import (
    AverageMeter,
)
from deepinv.utils import plot, plot_curves, zeros_like
from tqdm import tqdm
import torch
from pathlib import Path
from deepinv.loss import PSNR
import warnings


def test(
    model,
    test_dataloader,
    physics,
    metrics=PSNR(),
    online_measurements=False,
    physics_generator=None,
    device="cpu",
    plot_images=False,
    save_folder="results",
    plot_metrics=False,
    verbose=True,
    plot_only_first_batch=True,
    plot_measurements=True,
    show_progress_bar=True,
    **kwargs,
):
    r"""
    Tests a reconstruction model (algorithm or network).

    This function computes the chosen metrics of the reconstruction network on the test set,
    and optionally plots the reconstructions as well as the metrics computed along the iterations.
    Note that by default only the first batch is plotted.

    :param torch.nn.Module model: Reconstruction network, which can be PnP, unrolled, artifact removal
        or any other custom reconstruction network (unfolded, plug-and-play, etc).
    :param torch.utils.data.DataLoader test_dataloader: Test data loader, which should provide a tuple of (x, y) pairs.
        See :ref:`datasets <datasets>` for more details.
    :param deepinv.physics.Physics, list[deepinv.physics.Physics] physics: Forward operator(s)
        used by the reconstruction network at test time.
    :param deepinv.loss.Loss, list[deepinv.Loss] metrics: Metric or list of metrics used for evaluating the model.
        :ref:`See the libraries' evaluation metrics <loss>`.
    :param bool online_measurements: Generate the measurements in an online manner at each iteration by calling
        ``physics(x)``.
    :param None, deepinv.physics.generator.PhysicsGenerator physics_generator: Optional physics generator for generating
        the physics operators. If not None, the physics operators are randomly sampled at each iteration using the generator.
        Should be used in conjunction with ``online_measurements=True``.
    :param torch.device device: gpu or cpu.
    :param bool plot_images: Plot the ground-truth and estimated images.
    :param str save_folder: Directory in which to save plotted reconstructions.
        Images are saved in the ``save_folder/images`` directory
    :param bool plot_metrics: plot the metrics to be plotted w.r.t iteration.
    :param bool verbose: Output training progress information in the console.
    :param bool plot_only_first_batch: Plot only the first batch of the test set.
    :param bool plot_measurements: Plot the measurements y. default=True.
    :param bool show_progress_bar: Show progress bar.
    :returns: A tuple of floats (test_psnr, test_std_psnr, linear_std_psnr, linear_std_psnr) with the PSNR of the
        reconstruction network and a simple linear inverse on the test set.
    """

    if physics_generator is not None and not online_measurements:
        warnings.warn(
            "Physics generator is provided but online_measurements is False. Physics generator will not be used."
        )

    save_folder = Path(save_folder)

    model.eval()

    if type(physics) is not list:
        physics = [physics]

    if type(test_dataloader) is not list:
        test_dataloader = [test_dataloader]

    G = len(test_dataloader)

    show_operators = 5

    if type(metrics) is not list:
        metrics = [metrics]

    logs_metrics = [
        AverageMeter("Test " + l.__class__.__name__, ":.2e") for l in metrics
    ]

    logs_metrics_init = [
        AverageMeter("Test (no learning) " + l.__class__.__name__, ":.2e")
        for l in metrics
    ]

    for g in range(G):
        dataloader = test_dataloader[g]

        batches = len(dataloader) - int(dataloader.drop_last)
        iterator = iter(dataloader)
        for i in (
            progress_bar := tqdm(
                range(batches),
                ncols=150,
                disable=(not verbose or not show_progress_bar),
            )
        ):
            desc = f"Test operator {g + 1}" if G > 1 else "Test "
            progress_bar.set_description(desc)
            with torch.no_grad():
                if online_measurements:
                    data = next(
                        iterator
                    )  # In this case the dataloader outputs also a class label

                    if type(data) is tuple or type(data) is list:
                        x = data[0]
                    else:
                        x = data

                    x = x.to(device)
                    physics_cur = physics[g]

                    if physics_generator is not None:
                        params = physics_generator.step()
                        y = physics_cur(x, **params)
                    else:
                        y = physics_cur(x)
                else:
                    x, y = next(
                        iterator
                    )  # In this case the dataloader outputs also a class label

                    if type(x) is list or type(x) is tuple:
                        x = [s.to(device) for s in x]
                    else:
                        x = x.to(device)
                    physics_cur = physics[g]

                    y = y.to(device)

            if plot_metrics:
                x_net, optim_metrics = model(
                    y, physics_cur, x_gt=x, compute_metrics=True
                )
            else:
                x_net = model(y, physics_cur)

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

            # Compute the metrics over the batch
            for k, l in enumerate(metrics):
                loss = l(x=x, x_net=x_net, y=y, physics=physics)
                logs_metrics[k].update(loss.detach().cpu().numpy())
                loss = l(x=x, x_net=x_init, y=y, physics=physics)
                logs_metrics_init[k].update(loss.detach().cpu().numpy())

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

            if plot_images:
                if g < show_operators:
                    if not plot_only_first_batch or (plot_only_first_batch and i == 0):
                        if plot_measurements and len(y.shape) == 4:
                            imgs = [y, x_init, x_net, x]
                            name_imgs = ["Input", "No learning", "Recons.", "GT"]
                        else:
                            imgs = [x_init, x_net, x]
                            name_imgs = ["No learning", "Recons.", "GT"]
                        plot(
                            imgs,
                            titles=name_imgs,
                            save_dir=save_folder_im if plot_images else None,
                            show=plot_images,
                            return_fig=True,
                            rescale_mode="clip",
                        )

                if plot_metrics:
                    plot_curves(optim_metrics, save_dir=save_folder_curve, show=True)

    if verbose:
        for k, l in enumerate(metrics):
            print(
                f"Test {l.__class__.__name__}: No learning rec.: {logs_metrics_init[k].avg:.3f}+-{logs_metrics_init[k].std:.3f} "
                f"| Model: {logs_metrics[k].avg:.3f}+-{logs_metrics[k].std:.3f}. "
            )

    return (
        logs_metrics[0].avg,
        logs_metrics[0].std,
        logs_metrics_init[0].avg,
        logs_metrics_init[0].std,
    )
