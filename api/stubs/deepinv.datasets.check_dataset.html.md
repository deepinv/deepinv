# check_dataset

### deepinv.datasets.check_dataset(dataset, allow_non_tensor=True)

Check that a torch dataset is compatible with DeepInverse.

For details of what is compatible, see [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset).

* **Parameters:**
  * **dataset** ([*torch.utils.data.Dataset*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) – torch dataset.
  * **allow_non_tensor** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – allow image types that are not tensors (i.e. numpy ndarrays and PIL Images). Default `False`, which
    is recommended so that the dataset is asserted to return tensors to be compatible with deepinv.

## Examples using `check_dataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div>
<!-- thumbnail-parent-div-close --></div>
