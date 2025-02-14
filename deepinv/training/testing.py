import torch
from deepinv.loss.metric import PSNR
from deepinv.training import Trainer


def test(
    model,
    test_dataloader,
    physics,
    metrics=PSNR(),
    online_measurements=False,
    physics_generator=None,
    device="cpu",
    plot_images=False,
    save_folder=None,
    plot_convergence_metrics=False,
    verbose=True,
    rescale_mode="clip",
    show_progress_bar=True,
    no_learning_method="A_dagger",
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
    :param deepinv.loss.Loss, list[deepinv.loss.Loss] metrics: Metric or list of metrics used for evaluating the model.
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
    :param bool plot_convergence_metrics: plot the metrics to be plotted w.r.t iteration.
    :param bool verbose: Output training progress information in the console.
    :param bool plot_measurements: Plot the measurements y. default=True.
    :param bool show_progress_bar: Show progress bar.
    :param str no_learning_method: Reconstruction method used for the no learning comparison. Options are ``'A_dagger'``,
        ``'A_adjoint'``, ``'prox_l2'``, or ``'y'``. Default is ``'A_dagger'``. The user can modify the no-learning method
        by overwriting the :func:`no_learning_inference <deepinv.Trainer.no_learning_inference>` method
    :returns: A dictionary with the metrics computed on the test set, where the keys are the metric names, and include
        the average and standard deviation of the metric.
    """
    trainer = Trainer(
        model,
        physics=physics,
        train_dataloader=None,
        eval_dataloader=None,
        optimizer=None,
        metrics=metrics,
        online_measurements=online_measurements,
        physics_generator=physics_generator,
        device=device,
        plot_images=plot_images,
        plot_convergence_metrics=plot_convergence_metrics,
        verbose=verbose,
        rescale_mode=rescale_mode,
        no_learning_method=no_learning_method,
        show_progress_bar=show_progress_bar,
        **kwargs,
    )
    return trainer.test(test_dataloader, save_path=save_folder)
