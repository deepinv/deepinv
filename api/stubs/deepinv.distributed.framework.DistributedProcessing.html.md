# DistributedProcessing

### *class* deepinv.distributed.framework.DistributedProcessing(ctx, processor, , strategy=None, strategy_kwargs=None, max_batch_size=None, checkpoint_batches='auto', checkpoint_use_reentrant=False, checkpoint_preserve_rng_state=True, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Distributed signal processing using pluggable tiling and reduction strategies.

This class enables distributed processing of large signals (images, volumes, etc.) by:

1. Splitting the signal into patches using a chosen strategy
2. Distributing patches across multiple processes/GPUs
3. Processing each patch independently using a provided processor function
4. Combining processed patches back into the full signal with proper overlap handling

The processor can be any callable that operates on tensors (e.g., denoisers, priors,
neural networks, etc.). The class handles all distributed coordination automatically.

<hr />

**Example:**

```python
import torch
from deepinv.distributed import DistributedContext
from deepinv.distributed.framework import DistributedProcessing

x = torch.randn(1, 3, 1024, 1024)

with DistributedContext() as ctx:
    processor = torch.nn.Identity()
    distributed_processor = DistributedProcessing(
        ctx,
        processor,
        strategy_kwargs={"patch_size": 256, "overlap": 32},
        max_batch_size=1,
    )
    output = distributed_processor(x.to(ctx.device))
```

* **Parameters:**
  * **ctx** ([*DistributedContext*](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext)) – distributed context manager.
  * **processor** (*Callable* *[* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – processing function to apply to signal patches.
    Should accept a batched tensor of shape `(B, C, ...)` and return a tensor of the same shape.
    Examples: denoiser, neural network, prior gradient function, etc.
  * **strategy** ([*DistributedSignalStrategy*](https://deepinv.org/api/stubs/deepinv.distributed.strategies.DistributedSignalStrategy.html.md#deepinv.distributed.strategies.DistributedSignalStrategy) *|* *None*) – signal processing strategy for patch extraction
    and reduction. Either a custom strategy instance or `None`, which corresponds to the default tiling strategy.
  * **strategy_kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *|* *None*) – additional keyword arguments passed to the strategy constructor
    when using string strategy names. Examples: `patch_size`, `overlap`, `tiling_dims`. Default is `None`.
  * **max_batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – maximum number of patches to process in a single batch.
    If `None`, all local patches are batched together. Set to `1` for sequential processing
    (useful for memory-constrained scenarios). Higher values increase throughput but require more memory. Default is `None`.
  * **checkpoint_batches** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – activation checkpointing mode for patch batches
    during the backward pass. Checkpointing saves memory by recomputing
    activations instead of storing them. Use `"auto"` (default) to enable it
    only when gradients are enabled and there are multiple local patch batches,
    `"always"` to enable it whenever gradients are enabled, or `"never"`
    to disable it.
  * **checkpoint_use_reentrant** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – reentrant mode passed to [`torch.utils.checkpoint.checkpoint()`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint).
    Default is `False` (recommended by PyTorch).
  * **checkpoint_preserve_rng_state** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to preserve RNG state across forward recomputation when
    checkpointing. Default is `True`.

#### forward(x, \*args, gather=True, \*\*kwargs)

Apply distributed processing to input signal.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input signal tensor to process, typically of shape `(B, C, H, W)` for 2D
    or `(B, C, D, H, W)` for 3D signals.
  * **args** – additional positional arguments passed to the processor.
  * **gather** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to gather results across ranks. If False, returns local contribution. Default is `True`.
  * **kwargs** – additional keyword arguments passed to the processor.
* **Returns:**
  processed signal with the same shape as input.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-distributed-framework-distributedprocessing"></a>

## Examples using `DistributedProcessing`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In many imaging problems, the data to be processed can be very large, making it challenging to fit the denoising process into the memory of a single device. For instance, medical imaging or satellite imagery often involves processing gigapixel images that cannot be processed as a whole.">  <div class="sphx-glr-thumbnail-title">Distributed Denoiser with Image Tiling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div>
<!-- thumbnail-parent-div-close --></div>
