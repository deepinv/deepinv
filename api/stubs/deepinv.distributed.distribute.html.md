# distribute

### deepinv.distributed.distribute(object, ctx, , num_operators=None, type_object='auto', dtype=torch.float32, gather_strategy='concatenated', tiling_strategy=None, tiling_dims=None, patch_size=256, overlap=64, max_batch_size=None, checkpoint_batches='auto', checkpoint_use_reentrant=False, checkpoint_preserve_rng_state=True, \*\*kwargs)

Distribute a DeepInverse object across multiple devices.

This function takes a DeepInverse object and distributes it using the provided DistributedContext.

The list of supported DeepInverse objects includes:

> - Physics operators: a list of [`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics), [`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) or [`deepinv.physics.StackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedLinearPhysics.html.md#deepinv.physics.StackedLinearPhysics).
> - Data fidelity terms: a list of [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) or [`deepinv.optim.StackedPhysicsDataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.html.md#deepinv.optim.StackedPhysicsDataFidelity).
> - Priors/Denoisers: [`deepinv.models.Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) or [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) objects.
* **Parameters:**
  * **object** ([*StackedPhysics*](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) *|* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) *]*  *|* *Callable* *|* [*Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) *|* [*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *|* [*StackedPhysicsDataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.html.md#deepinv.optim.StackedPhysicsDataFidelity) *|* [*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*DataFidelity*](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) *]*  *|* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) *|* [*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *|* *Sequence* *[*[*torch.nn.parameter.Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) *]*) – DeepInverse object to distribute.
  * **ctx** ([*DistributedContext*](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext)) – distributed context manager.
  * **num_operators** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – number of physics operators when using a factory for physics, otherwise inferred. Default is `None`.
  * **type_object** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – type of object to distribute. Options are `'physics'`, `'linear_physics'`,
    `'data_fidelity'`, `'denoiser'`, `'module'`, `'parameters'`, or `'auto'` for automatic detection.
    Default is `'auto'`.
    `'module'` is restricted to [`BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) models built
    with `unfold=True` (including the legacy `BaseUnfold` subclass).
    Generic `torch.nn.Module` instances are intentionally not supported by this
    API, to avoid ambiguous partial auto-distribution.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *|* *None*) – data type for distributed object. Default is `torch.float32`.
  * **gather_strategy** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – 

    strategy for gathering distributed results.

    Options are:
    : - `'naive'`: Simple object serialization (best for small tensors)
      - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
      - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)

    Default is `'concatenated'`.
  * **tiling_strategy** ([*DistributedSignalStrategy*](https://deepinv.org/api/stubs/deepinv.distributed.strategies.DistributedSignalStrategy.html.md#deepinv.distributed.strategies.DistributedSignalStrategy) *|* *None*) – strategy for tiling the signal (for Denoiser).
    Options are either a custom strategy instance or `None`, which corresponds to the default tiling strategy.
  * **tiling_dims** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*  *|* *None*) – 

    dimensions to tile over (for Denoiser).

    Can be one of the following:
    : - If `None` (default), tiles the last N-2 dimensions of your input tensor.
      - If an int `N`, only tiles over the specified dimension.
      - If a tuple, specifies exact dimensions to tile.

    Examples:
    : - For `(B, C, H, W)` image: `tiling_dims=(2, 3)` tiles over H and W.
      - For `(B, C, D, H, W)` volume: `tiling_dims=(2, 3, 4)` tiles over D, H, W.
      - For `(B, C, H, W)` image: `tiling_dims=2` tiles only over H dimension.
      - For `(B, C, D, H, W)` volume: `tiling_dims=None` tiles over D, H, W dimensions.
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of patches for tiling strategies (for Denoiser).
    Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is `256`.
  * **overlap** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – receptive field size for overlap in tiling strategies (for Denoiser).
    Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is `64`.
  * **max_batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – maximum number of patches to process in a single batch (for Denoiser). If `None`, all patches are batched together. Set to `1` for sequential processing. Default is `None`.
  * **checkpoint_batches** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – activation checkpointing mode for
    patch-batches during backward (for Denoiser).
    Supported values are `'auto'`, `'always'` and `'never'`.
    Default is `'auto'`.
  * **checkpoint_use_reentrant** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – reentrant mode for activation
    checkpointing in denoiser processing. Default is `False`.
  * **checkpoint_preserve_rng_state** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – preserve RNG state during
    checkpoint recomputation in denoiser processing. Default is `True`.
  * **kwargs** – additional keyword arguments for specific distributed classes.
* **Returns:**
  Distributed version of the input object.
* **Return type:**
  [*DistributedStackedPhysics*](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedPhysics.html.md#deepinv.distributed.framework.DistributedStackedPhysics) | [*DistributedStackedLinearPhysics*](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedLinearPhysics.html.md#deepinv.distributed.framework.DistributedStackedLinearPhysics) | [*DistributedProcessing*](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedProcessing.html.md#deepinv.distributed.framework.DistributedProcessing) | [*DistributedDataFidelity*](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedDataFidelity.html.md#deepinv.distributed.framework.DistributedDataFidelity) | [*Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) | [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[*Parameter*](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)] | [*BaseOptim*](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim)

<hr />

* **Examples:**
  Distribute a Physics object:
  ```pycon
  >>> from deepinv.physics import Blur, StackedLinearPhysics
  >>> from deepinv.distributed import DistributedContext, distribute
  >>> with DistributedContext() as ctx:
  ...     physics = StackedLinearPhysics([Blur(kernel_size=5), Blur(kernel_size=9)])
  ...     dphysics = distribute(physics, ctx)
  ```

  Distribute a DataFidelity object:
  ```pycon
  >>> from deepinv.optim.data_fidelity import L2
  >>> from deepinv.distributed import DistributedContext, distribute
  >>> with DistributedContext() as ctx:
  ...     data_fidelity = L2()
  ...     ddata_fidelity = distribute(data_fidelity, ctx)
  ```

  Distribute a Prior object:
  ```pycon
  >>> from deepinv.models import DnCNN
  >>> from deepinv.distributed import DistributedContext, distribute
  >>> with DistributedContext() as ctx:
  ...     denoiser = DnCNN()
  ...     ddenoiser = distribute(denoiser, ctx)
  ```

  Distribute a full unfolded PGD model in one call:
  ```pycon
  >>> from deepinv.models import DnCNN
  >>> from deepinv.optim import PGD
  >>> from deepinv.optim.data_fidelity import L2
  >>> from deepinv.optim.prior import PnP
  >>> from deepinv.distributed import DistributedContext, distribute
  >>> with DistributedContext() as ctx:
  ...     model = PGD(
  ...         data_fidelity=L2(),
  ...         prior=PnP(DnCNN(in_channels=1, out_channels=1)),
  ...         stepsize=[0.9, 0.8],
  ...         max_iter=2,
  ...         unfold=True,
  ...     )
  ...     distribute(model, ctx, patch_size=64, overlap=8)
  ```

## Examples using `distribute`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In many imaging problems, the data to be processed can be very large, making it challenging to fit the denoising process into the memory of a single device. For instance, medical imaging or satellite imagery often involves processing gigapixel images that cannot be processed as a whole.">  <div class="sphx-glr-thumbnail-title">Distributed Denoiser with Image Tiling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
