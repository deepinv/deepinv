# plot_parameters

### deepinv.utils.plot_parameters(model, init_params=None, save_dir=None, show=True)

Plot the parameters of the model before and after training.
This can be used after training Unfolded optimization models.

* **Parameters:**
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – the model whose parameters are plotted. The parameters are contained in the dictionary
    `params_algo` attribute of the model.
  * **init_params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – the initial parameters of the model, before training. Defaults to `None`.
  * **save_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – the directory where to save the plot. Defaults to `None`.
  * **show** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to show the plot. Defaults to `True`.

## Examples using `plot_parameters`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div>
<!-- thumbnail-parent-div-close --></div>
