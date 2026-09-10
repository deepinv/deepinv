# Trainer

### *class* deepinv.Trainer(model, physics, optimizer, train_dataloader, ...)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Trainer class for training a reconstruction network.

#### SEE ALSO
See the [User Guide](https://deepinv.org/user_guide/training/trainer.html.md#trainer) for more details and for how to adapt the trainer to your needs.

See [Training a reconstruction model](https://deepinv.org/auto_examples/models/demo_training.html.md#sphx-glr-auto-examples-models-demo-training-py) for a simple example of how to use the trainer.

Training can be done by calling the [`deepinv.Trainer.train()`](#deepinv.Trainer.train) method, whereas
testing can be done by calling the [`deepinv.Trainer.test()`](#deepinv.Trainer.test) method.

<hr />

#### TIP
The training code can synchronize with MLOps tools like [Weights & Biases](https://wandb.ai/site) and [MLflow](https://mlflow.org)
for logging and visualization by setting `wandb_vis=True` or `mlflow_vis=True`.

Parameters are described below, grouped into **Basics**, **Optimization**, **Evaluation**, **Physics Generators**,
**Model Saving**, **Comparing with Pseudoinverse Baseline**, **Plotting**, **Verbose** and **Weights & Biases**.

* **Basics:**

The **dataloaders** should return data in the correct format for DeepInverse: see [datasets user guide](https://deepinv.org/user_guide/training/datasets.html.md#datasets) for
how to use predefined datasets, create datasets, or generate datasets. These will be checked automatically with [`deepinv.datasets.check_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.check_dataset.html.md#deepinv.datasets.check_dataset).

If the dataloaders do not return
measurements `y`, then you should use the `online_measurements=True` option which generates measurements in an online manner (optionally with parameters), running
under the hood `y=physics(x)` or `y=physics(x, **params)`. Otherwise if dataloaders do return measurements `y`, set `online_measurements=False` (default) otherwise
`y` will be ignored and new measurements will be generated online.

#### TIP
If your dataloaders do not return `y` but you do not want online measurements, use [`deepinv.datasets.generate_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset) to generate a dataset
of offline measurements from a dataset of `x` and a `physics`.

* **Parameters:**
  * **model** ([*deepinv.models.Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction network, which can be [any reconstruction network](https://deepinv.org/user_guide/reconstruction/introduction.html.md#reconstructors).
    or any other custom reconstruction network.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *]*) – [Forward operator(s)](https://deepinv.org/user_guide/physics/physics.html.md#physics-list).
  * **train_dataloader** ([*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *]*) – Train data loader(s), see [datasets user guide](https://deepinv.org/user_guide/training/datasets.html.md#datasets)
    for how we expect data to be provided.
  * **online_measurements** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Generate new measurements `y` in an online manner at each iteration by calling
    `y=physics(x)`. If `False` (default), the measurements are loaded from the training dataset.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – Device on which to run the training (e.g., ‘cuda’, ‘mps’ or ‘cpu’). Default is first ‘cuda’ and second ‘mps’ if available, otherwise ‘cpu’.
  * **mixed_precision** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Mixed precision to use. If False, standard float32 training is performed. If True, defaults to float16.
    If a string, that dtype will be used (only ‘float16’ / ‘fp16’ and ‘bfloat16’ / ‘bf16’ are supported.) Mixed-precision is only used for training steps, not eval or test.

<hr />

* **Optimization:**
* **Parameters:**
  * **optimizer** (*None* *,* [*torch.optim.Optimizer*](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)) – Torch optimizer for training the network. Default is `None`.
  * **epochs** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of training epochs.
    Default is 100. The trainer will perform gradient steps equal to the `min(epochs*n_batches, max_batch_steps)`.
  * **max_batch_steps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of gradient steps per iteration.
    Default is `1e10`. The trainer will perform batch steps equal to the `min(epochs*n_batches, max_batch_steps)`.
  * **scheduler** (*None* *,* [*torch.optim.lr_scheduler.LRScheduler*](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html#torch.optim.lr_scheduler.LRScheduler)) – Torch scheduler for changing the learning rate across iterations. Default is `None`.
  * **losses** ([*deepinv.loss.Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*deepinv.loss.Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) *]*) – Loss or list of losses used for training the model.
    Optionally wrap losses using a loss scheduler for more advanced training.
    [See the libraries’ training losses](https://deepinv.org/user_guide/training/loss.html.md#loss).
    Where relevant, the underlying metric should have `reduction=None` as we perform the averaging
    using [`deepinv.utils.AverageMeter`](https://deepinv.org/api/stubs/deepinv.utils.AverageMeter.html.md#deepinv.utils.AverageMeter) to deal with uneven batch sizes. Default is [`supervised loss`](https://deepinv.org/api/stubs/deepinv.loss.SupLoss.html.md#deepinv.loss.SupLoss).
  * **grad_clip** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Gradient clipping value for the optimizer. If None, no gradient clipping is performed. Default is None.
  * **optimizer_step_multi_dataset** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the optimizer step is performed once on all datasets. If `False`, the optimizer step is performed on each dataset separately.

#### NOTE
The losses and evaluation metrics can be chosen from [our training losses](https://deepinv.org/user_guide/training/loss.html.md#loss) or [our metrics](https://deepinv.org/user_guide/training/metric.html.md#metric)

Custom losses can be used, as long as it takes as input `(x, x_net, y, physics, model)`
and returns a tensor of length `batch_size` (i.e. `reduction=None` in the underlying metric, as we perform averaging to deal with uneven batch sizes),
where `x` is the ground truth, `x_net` is the network reconstruction $\inversef{y}{A}$,
`y` is the measurement vector, `physics` is the forward operator
and `model` is the reconstruction network. Note that not all inputs need to be used by the loss,
e.g., self-supervised losses will not make use of `x`.

Custom metrics can also be used in the exact same way as custom losses.

#### NOTE
When `optimizer_step_multidataset=True` and `grad_clip` is performed, mixed-precision with `float16` will clip the gradient from each dataset separately `clip(g1) + clip(g2)`,
while in other modes, the sum of all gradients is clipped `clip(g1 + g2)`. Therefore, different behavior is possible in this setting between `float16` and other precision modes.

<hr />

* **Evaluation:**

#### NOTE
- **Supervised evaluation**: If ground-truth data is available for validation, use any
  [full reference metric](https://deepinv.org/user_guide/training/metric.html.md#full-reference-metrics), e.g. [`PSNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.PSNR.html.md#deepinv.loss.metric.PSNR).
- **Self-supervised evaluation**: If no ground-truth data is available for validation, it is
  still possible to validate using:
  1. [no reference metrics](https://deepinv.org/user_guide/training/metric.html.md#no-reference-metrics), e.g. [`NIQE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NIQE.html.md#deepinv.loss.metric.NIQE)
  2. [self-supervised losses](https://deepinv.org/user_guide/training/loss.html.md#self-supervised-losses) with
     `compute_eval_losses=True` and `metrics=None`.

     Additionally, in this case, we recommend setting `compute_train_metrics=False` to avoid computing
     metrics in `model.train()` mode. This is required by many self-supervised losses,
     such as [`SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss) or
     [`R2RLoss`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.html.md#deepinv.loss.R2RLoss), which behave differently in
     `model.train()` and `model.eval()` modes.

     For early-stopping with self-supervised losses, `early_stop_on_losses` must also be `True`.

* **Parameters:**
  * **eval_dataloader** (*None* *,* [*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *]*) – Evaluation data loader(s),
    see [datasets user guide](https://deepinv.org/user_guide/training/datasets.html.md#datasets) for how we expect data to be provided.
  * **metrics** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *]* *,* *None*) – Metric or list of metrics used for evaluating the model.
    They should have `reduction=None` as we perform the averaging using [`deepinv.utils.AverageMeter`](https://deepinv.org/api/stubs/deepinv.utils.AverageMeter.html.md#deepinv.utils.AverageMeter) to deal with uneven batch sizes.
    [See the libraries’ evaluation metrics](https://deepinv.org/user_guide/training/metric.html.md#metric). Default is [`PSNR`](https://deepinv.org/api/stubs/deepinv.loss.metric.PSNR.html.md#deepinv.loss.metric.PSNR).
  * **eval_interval** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of epochs (or train iters, if `log_train_batch=True`) between each evaluation of
    the model on the evaluation set. Default is `1`.
  * **compute_train_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – 

    If `False`, do not compute metrics during training on train set.
    If `True` (default), during training all metrics are computed on the training dataloader.

    #### WARNING
    If `compute_train_metrics=True` the metrics are computed using the model prediction during training (i.e., in `model.train()` mode) to avoid an additional
    forward pass. This can lead to metrics that are different at test time when the model is in `model.eval()` mode,
    and/or produce errors if the network does not provide the same output shapes under train and eval modes (e.g., which is the case of [`some self-supervised losses`](https://deepinv.org/api/stubs/deepinv.loss.ReducedResolutionLoss.html.md#deepinv.loss.ReducedResolutionLoss)).
  * **compute_eval_losses** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the losses are computed during evaluation. Default is `False`. This is useful
    when using self-supervised losses for evaluation and early-stopping or to make sure that the model is performing
    similarly on losses on the train and eval sets.
  * **early_stop** (*None* *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – If not `None`, the training stops when the first evaluation metric is not improving
    after `early_stop` passes over the eval dataset. Default is `None` (no early stopping).
    The user can modify the strategy for saving the best model by overriding the [`deepinv.Trainer.stop_criterion()`](#deepinv.Trainer.stop_criterion) method.
  * **early_stop_on_losses** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Early stop using losses computed on the eval set instead of metrics: useful for stopping when
    using a self-supervised loss or when ground truth is unavailable. Default is `False`.
    If `True`, requires `compute_eval_losses` to be `True`.
  * **log_train_batch** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, log train batch and eval-set metrics and losses for each train batch during training.
    This is useful for visualizing train progress inside an epoch, not just over epochs.
    If `False` (default), log average over dataset per epoch (standard training).

#### TIP
If a validation dataloader `eval_dataloader` is provided, the trainer will also **save the best model** according to the
first metric in the list, using the following format:
`save_path/yyyy-mm-dd_hh-mm-ss/ckp_best.pth.tar`. The user can modify the strategy for saving the best model
by overriding the [`deepinv.Trainer.save_best_model()`](#deepinv.Trainer.save_best_model) method.
The best model can be also loaded using the [`deepinv.Trainer.load_best_model()`](#deepinv.Trainer.load_best_model) method.

<hr />

* **Physics Generators:**
* **Parameters:**
  * **physics_generator** (*None* *,* [*deepinv.physics.generator.PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)) – Optional [physics generator](https://deepinv.org/user_guide/physics/intro.html.md#physics-generators) for generating
    the physics operators. If not `None`, the physics operators are randomly sampled at each iteration using the generator.
    Should be used in conjunction with `online_measurements=True`, no effect when `online_measurements=False`. Also see `loop_random_online_physics`. Default is `None`.
  * **loop_random_online_physics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, resets the physics generator **and** noise model back to its initial state at the beginning of each epoch,
    so that the same measurements are generated each epoch. Requires `shuffle=False` in dataloaders. If `False`, generates new physics every epoch.
    Used in conjunction with `online_measurements=True` and `physics_generator` or noise model in `physics`, no effect when `online_measurements=False`. Default is `False`.

#### WARNING
If the physics changes at each iteration for online measurements (e.g. if `physics_generator` is used to generate random physics operators or noise model is used),
the generated measurements will randomly vary each epoch.
If this is not desired (i.e. you want the same online measurements each epoch), set `loop_random_online_physics=True`.
This resets the physics generator and noise model’s random generators every epoch.

**Caveat**: this requires `shuffle=False` in your dataloaders.

An alternative, safer solution is to generate and save params offline using [`deepinv.datasets.generate_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset).
The params dict will then be automatically updated every time data is loaded.

<hr />

* **Model Saving:**
* **Parameters:**
  * **save_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Directory in which to save the trained model. Default is `"."` (current folder).
  * **ckp_interval** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – The model is saved every `ckp_interval` epochs. Default is `1`.
  * **ckpt_pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – path of the pretrained checkpoint. If `None` (default), no pretrained checkpoint is loaded.

Training details are saved every `ckp_interval` epochs in the following format

```default
save_path/yyyy-mm-dd_hh-mm-ss/ckp_{epoch}.pth.tar
```

where `.pth.tar` file contains a dictionary with the keys:

- `epoch`: current epoch number when saved
- `state_dict`: model parameters state dictionary
- `loss`: loss history on train set
- `train_metrics`: metric history on train set
- `eval_loss`: loss history on eval set
- `eval_metrics`: metric history on eval set
- `optimizer`: optimizer state dictionary, or `None` if not used
- `scheduler`: learning rate scheduler state dictionary, or `None` if not used

<hr />

* **Comparison with Pseudoinverse Baseline:**
* **Parameters:**
  * **compare_no_learning** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the no learning method is compared to the network reconstruction. Default is `False`.
  * **no_learning_method** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)) – Reconstruction method used for the no learning comparison. Options are `'A_dagger'`, `'A_adjoint'`,
    `'prox_l2'`, or `'y'`. Default is `'A_adjoint'`. The user can also provide a custom method by overriding the
    [`no_learning_inference`](#deepinv.Trainer.no_learning_inference) method. Default is `'A_adjoint'`.

<hr />

* **Plotting:**
* **Parameters:**
  * **plot_images** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Plots reconstructions every `ckp_interval` epochs. Default is `False`.
  * **plot_measurements** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Plot the measurements y, default is `True`.
  * **plot_convergence_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Plot convergence metrics for model, default is `False`.
  * **rescale_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Rescale mode for plotting images. Default is `'clip'`.

<hr />

* **Verbose:**
* **Parameters:**
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output training progress information in the console. Default is `True`.
  * **verbose_individual_losses** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – **Deprecated.** This parameter is deprecated and will be removed in a future version.
    Individual losses are now always added to logs when multiple losses are present. Default is `None`.
  * **show_progress_bar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Show a progress bar during training. Default is `True`.
  * **freq_update_progress_bar** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – progress bar postfix update frequency (measured in iterations). Defaults to 1.
    Increasing this may speed up training.
  * **check_grad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Compute and print the gradient norm at each iteration. Default is `False`.

<hr />

* **Weights & Biases:**
* **Parameters:**
  * **wandb_vis** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Logs data onto Weights & Biases, see [https://wandb.ai/](https://wandb.ai/) for more details. Default is `False`.
  * **wandb_setup** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary with the setup for wandb, see [https://docs.wandb.ai/quickstart](https://docs.wandb.ai/quickstart) for more details. Default is `{}`.
  * **plot_interval** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Frequency of plotting images to MLOps tools (wandb or MLflow) during evaluation (at the end of each epoch).
    If `1`, plots at each epoch. Default is `1`.

<hr />

* **MLflow:**
* **Parameters:**
  * **mlflow_vis** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Logs data onto MLflow, see [https://mlflow.org/](https://mlflow.org/) for more details. Default is `False`.
  * **mlflow_setup** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary with the setup for mlflow, see [https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) for more details. Default is `{}`.
  * **non_blocking_transfers** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Use non-blocking host-to-device transfers for data loading. Default is `True` only when device is cuda. If device is not cuda, then this is forced to `False`.
    See [PyTorch docs](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) for more details.
    It is advised to enable `pin_memory=True` in the dataloader when using this option for best performance.

#### check_clip_grad()

Check the gradient norm and perform gradient clipping if necessary.

#### compute_loss(physics, x, y, train=True, epoch=None, step=False)

Compute the loss and perform the backward pass.

It evaluates the reconstruction network, computes the losses, and performs the backward pass.

* **Parameters:**
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Current physics operator.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Ground truth.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the model is trained, otherwise it is evaluated.
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current epoch.
  * **step** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to perform an optimization step when computing the loss.
* **Returns:**
  (tuple) The network reconstruction x_net (for plotting and computing metrics) and
  the logs (for printing the training progress).

#### compute_metrics(x, x_net, y, physics, logs, train=True, epoch=None)

Compute the metrics.

It computes the metrics over the batch.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Ground truth.
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Network reconstruction.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Current physics operator.
  * **logs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the logs for printing the training progress.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the model is trained, otherwise it is evaluated.
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – current epoch.
* **Returns:**
  The reconstructed signal during eval (if `x_net=None`) and the logs with the metrics

#### get_samples(iterators, g)

Get the samples.

This function returns a dictionary containing necessary data for the model inference. It needs to contain
the measurement, the ground truth, and the current physics operator, but can also contain additional data.

* **Parameters:**
  * **iterators** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – List of dataloader iterators.
  * **g** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current dataloader index.
* **Returns:**
  the tuple returned by the get_samples_online or get_samples_offline function.

#### get_samples_offline(iterators, g)

Get the samples for the offline measurements.

In this setting, samples have been generated offline and are loaded from the dataloader.
This function returns a tuple `(x, y, physics)` for the model inference. You can override this function to add custom data.

If the dataloader returns 3-tuples, this is assumed to be `(x, y, params)` where
`params` is a dict of physics generator params. These params are then used to update
the physics. The dataloader batch can also be a dict with `"x"`, `"y"` and optional
`"params"` keys (see [`deepinv.datasets.check_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.check_dataset.html.md#deepinv.datasets.check_dataset)).

* **Parameters:**
  * **iterators** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – List of dataloader iterators.
  * **g** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current dataloader index.
* **Returns:**
  a dictionary containing at least: the ground truth, the measurement, and the current physics operator.

#### get_samples_online(iterators, g)

Get the samples for the online measurements.

In this setting, a new sample is generated at each iteration by calling the physics operator.
This function returns a tuple `(x, y, physics)` for the model inference.

Assumes the dataloader returns ground truth `x`, or tuples of (`x`, `params`), or a dict with `"x"` key and optional `"params"` key. Note that `params` are ignored if a physics generator is provided.

* **Parameters:**
  * **iterators** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – List of dataloader iterators.
  * **g** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current dataloader index.
* **Returns:**
  a tuple containing at least: the ground truth, the measurement, and the current physics operator.

#### load_best_model()

Load the best model.

It loads the model from the checkpoint saved during training.

* **Returns:**
  The model.

#### load_model(ckpt_pretrained=None, strict=True)

Load model from checkpoint.

* **Parameters:**
  * **ckpt_pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – checkpoint filename. If `None`, use checkpoint passed to class init.
    If not `None`, override checkpoint passed to class.
  * **strict** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – strict load weights to model.
* **Returns:**
  if checkpoint loaded, return checkpoint dict, else return `None`
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### log_metrics_mlops(logs, step, train=True)

Log the metrics to MLOps tools including wandb and MLflow.

It logs the metrics to wandb and MLflow.

* **Parameters:**
  * **logs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the metrics to log.
  * **step** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current step to log. If `Trainer.log_train_batch=True`, this is the batch iteration, if `False` (default), this is the epoch.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the model is trained, otherwise it is evaluated.

#### model_inference(y, physics, x=None, train=True, \*\*kwargs)

Perform the model inference.

It returns the network reconstruction given the samples.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Current physics operator.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Optional ground truth, used for computing convergence metrics.
* **Returns:**
  The network reconstruction.

#### no_learning_inference(y, physics)

Perform the no learning inference.

By default it returns the (linear) pseudo-inverse reconstruction given the measurement.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Current physics operator.
* **Returns:**
  Reconstructed image.

#### plot(epoch, physics, x, y, x_net, train=True)

Plot ground truths, measurements and reconstructions and at test time, optionally save them.

#### NOTE
Images can be saved to disk at test time by providing a value for the parameter `save_folder_im`
when calling the method [`deepinv.Trainer.test()`](#deepinv.Trainer.test). Note that in that case, every test sample is saved and not only the first ones.

* **Parameters:**
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current epoch.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Current physics operator.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Ground truth.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Network reconstruction.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the model is trained, otherwise it is evaluated.

#### reset_metrics()

Reset the metrics.

#### save_best_model(epoch, train_ite, \*\*kwargs)

Save the best model using validation metrics.

By default, uses validation based on first metric. If no metric is provided (e.g. in self-supervised learning),
uses the first loss on the eval dataset instead (requires having `compute_eval_losses=True`).

Override this method to provide custom criterion.

* **Parameters:**
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current epoch.
  * **train_ite** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current training batch iteration, equal to (current epoch $\times$
    number of batches) + current batch within epoch

#### save_model(filename, epoch, state=None)

Save the model.

It saves the model every `ckp_interval` epochs in `save_path/filename`.

* **Parameters:**
  * **filename** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – checkpoint filename.
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current epoch.
  * **state** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – custom objects to save with model

#### setup_train(train=True, \*\*kwargs)

Set up the training process.

It initializes the wandb logging, the different metrics, the save path, the physics and dataloaders,
and the pretrained checkpoint if given.

* **Parameters:**
  **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether model is being trained.

#### step(epoch, progress_bar, train_ite=None, train=True, last_batch=False, update_progress_bar=False)

Train/Eval a batch.

It performs the forward pass, the backward pass, and the evaluation at each iteration.

* **Parameters:**
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current epoch.
  * **progress_bar** – [tqdm](https://tqdm.github.io/docs/tqdm/) progress bar.
  * **train_ite** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – train iteration, only needed for logging if `Trainer.log_train_batch=True`
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the model is trained, otherwise it is evaluated.
  * **last_batch** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the last batch of the epoch is being processed.
* **Returns:**
  The current physics operator, the ground truth, the measurement, and the network reconstruction.

#### stop_criterion(epoch, train_ite, \*\*kwargs)

Stop criterion for early stopping.

By default, stops optimization when first eval metric doesn’t improve in the last 3 evaluations.

If `early_stop_on_losses=True` (default is `False`)
uses the first loss on the eval dataset instead (requires having `compute_eval_losses=True`).

Override this method to early stop on a custom condition.

* **Parameters:**
  * **epoch** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current epoch.
  * **train_ite** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Current training batch iteration, equal to (current epoch $\times$ number
    of batches) + current batch within epoch
  * **metric_history** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – Dictionary containing the metrics history, with the metric name as key.
  * **metrics** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – List of metrics used for evaluation.

#### test(test_dataloader, save_path=None, compare_no_learning=True, log_raw_metrics=False, metrics=None)

Test the model, compute metrics and plot images.

#### NOTE
It is possible to save the reconstructed images along with the ground truths and measurements by specifying a value for the parameter `save_path`. Note that in this case, every test sample is saved and not only the first ones.

* **Parameters:**
  * **test_dataloader** ([*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *]*) – Test data loader(s), see [datasets user guide](https://deepinv.org/user_guide/training/datasets.html.md#datasets)
    for how we expect data to be provided.
  * **save_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to the directory where to save the plotted images if desired (optional).
  * **compare_no_learning** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the linear reconstruction is compared to the network reconstruction.
  * **log_raw_metrics** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, also return non-aggregated metrics as a list.
  * **metrics** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *]* *,* *None*) – Metric or list of metrics used for evaluation. If
    `None`, uses the metrics provided during Trainer initialization.
* **Returns:**
  dict of metrics, timings (in sec) and peak GPU memory usage (in GB) results with means and stds.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### NOTE
Timings correspond to total time for `test` to run, which also includes time to load data and compute metrics.
Therefore, the reported runtime will be greater than just model inference timings.

#### train()

Train the model.

It performs the training process, including the setup, the evaluation, the forward and backward passes,
and the visualization.

* **Returns:**
  The trained model.

<a id="sphx-glr-backref-deepinv-trainer"></a>

## Examples using `Trainer`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
