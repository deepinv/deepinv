# test

### deepinv.test(model, test_dataloader, physics, metrics=None, online_measurements=False, physics_generator=None, device='cpu', plot_images=False, save_folder=None, plot_convergence_metrics=False, verbose=True, rescale_mode='clip', show_progress_bar=True, compare_no_learning=True, no_learning_method='A_adjoint', \*\*kwargs)

Tests a reconstruction model (algorithm or network).

This function computes the chosen metrics of the reconstruction network on the test set,
and optionally plots the reconstructions as well as the metrics computed along the iterations.
Note that by default only the first batch is plotted.

* **Parameters:**
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction network, which can be PnP, unrolled, artifact removal
    or any other custom reconstruction network (unfolded, plug-and-play, etc).
  * **test_dataloader** ([*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)) – Test data loader, which should provide a tuple of (x, y) pairs.
    See [datasets](https://deepinv.org/user_guide/training/datasets.md#datasets) for more details.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.md#deepinv.physics.Physics) *]*) – Forward operator(s)
    used by the reconstruction network at test time.
  * **metrics** ([*deepinv.loss.Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.md#deepinv.loss.Loss) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.loss.Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.md#deepinv.loss.Loss) *]*) – Metric or list of metrics used for evaluating the model. Defaults to [`deepinv.loss.metric.PSNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.PSNR.md#deepinv.loss.metric.PSNR).
    [See the libraries’ evaluation metrics](https://deepinv.org/user_guide/training/loss.md#loss).
  * **online_measurements** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Generate the measurements in an online manner at each iteration by calling
    `physics(x)`.
  * **physics_generator** (*None* *,* [*deepinv.physics.generator.PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.md#deepinv.physics.generator.PhysicsGenerator)) – Optional physics generator for generating
    the physics operators. If not None, the physics operators are randomly sampled at each iteration using the generator.
    Should be used in conjunction with `online_measurements=True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – gpu or cpu.
  * **plot_images** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Plot the ground-truth and estimated images.
  * **save_folder** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Directory in which to save plotted reconstructions.
    Images are saved in the `save_folder/images` directory
  * **plot_convergence_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – plot the metrics to be plotted w.r.t iteration.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output training progress information in the console.
  * **plot_measurements** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Plot the measurements y. default=True.
  * **show_progress_bar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Show progress bar.
  * **compare_no_learning** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the linear reconstruction is compared to the network reconstruction.
  * **no_learning_method** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Reconstruction method used for the no learning comparison. Options are `'A_dagger'`,
    `'A_adjoint'`, `'prox_l2'`, or `'y'`. Default is `'A_adjoint'`. The user can modify the no-learning method
    by overwriting the [`no_learning_inference`](https://deepinv.org/api/stubs/deepinv.Trainer.md#deepinv.Trainer.no_learning_inference) method
* **Returns:**
  A dictionary with the metrics computed on the test set, where the keys are the metric names, and include
  the average and standard deviation of the metric, timing and peak GPU memory usage information. Timings correspond to total test time in seconds.

## Examples using `test`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_dataset_thumb.png)

[Bring your own dataset](https://deepinv.org/auto_examples/basics/demo_custom_dataset.md)

  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_optim_thumb.png)

[Use iterative reconstruction algorithms](https://deepinv.org/auto_examples/basics/demo_custom_optim.md)

  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_training_thumb.png)

[Training a reconstruction model](https://deepinv.org/auto_examples/models/demo_training.md)

  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_custom_prior_thumb.png)

[Image deblurring with custom deep explicit prior.](https://deepinv.org/auto_examples/optimization/demo_custom_prior.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in zhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_DPIR_deblur_thumb.png)

[DPIR method for PnP image deblurring.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.md)

  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_multiscale_thumb.png)

[Multi-scale Plug-and-Play for Inpainting](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_multiscale.md)

  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of romano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) hurault2021gradient which provides an explicit prior.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_RED_GSPnP_SR_thumb.png)

[Regularization by Denoising (RED) for Super-Resolution.](https://deepinv.org/auto_examples/plug-and-play/demo_RED_GSPnP_SR.md)

  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constrained_LISTA_thumb.png)

[Unfolded Chambolle-Pock for constrained image inpainting](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constrained_LISTA.md)

  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
