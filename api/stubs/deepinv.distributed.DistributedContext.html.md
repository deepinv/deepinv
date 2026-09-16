# DistributedContext

### *class* deepinv.distributed.DistributedContext(backend=None, cleanup=True, seed=None, seed_offset=True, deterministic=False, device_mode=None)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Context manager for distributed computing.

Handles:
- Initialization/destruction of the process group (if `RANK` / `WORLD_SIZE` environment variables exist)
- Backend choice: NCCL when one-GPU-per-process per node, else Gloo.
- Device selection based on `LOCAL_RANK` and visible GPUs
- Sharding helpers and tiny communication helpers

#### NOTE
The world size refers to the total number of processes (usually one per GPU).
The rank of a process refers to its unique ID in the range [0, world_size-1].

* **Parameters:**
  * **backend** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – backend to use for distributed communication. If `None` (default), automatically selects NCCL for GPU or Gloo for CPU.
  * **cleanup** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to clean up the process group on exit. Default is `True`.
  * **seed** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *|* *None*) – random seed for reproducible results. If provided, behavior depends on `seed_offset`. Default is `None`.
  * **seed_offset** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to add rank offset to seed (each rank gets `seed + rank`). Default is `True`.
    When `True`: each process uses a unique seed for diverse random sequences.
    When `False`: all processes share the same seed for synchronized randomness.
  * **deterministic** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use deterministic cuDNN operations. Default is `False`.
  * **device_mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *|* *None*) – device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic. Default is `None`.

#### all_gather(x, group=None)

Gather one tensor per rank and return a stacked tensor of shape
`(world_size, *x.shape)`.

#### local_indices(num_items)

Get local indices for this rank based on round robin sharding.

* **Parameters:**
  **num_items** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – total number of items to shard.
* **Returns:**
  list of indices assigned to this rank.
* **Return type:**
  [list](https://docs.python.org/3.9/library/stdtypes.html#list)[[int](https://docs.python.org/3.9/library/functions.html#int)]

<a id="sphx-glr-backref-deepinv-distributed-distributedcontext"></a>

## Examples using `DistributedContext`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In many imaging problems, the data to be processed can be very large, making it challenging to fit the denoising process into the memory of a single device. For instance, medical imaging or satellite imagery often involves processing gigapixel images that cannot be processed as a whole.">  <div class="sphx-glr-thumbnail-title">Distributed Denoiser with Image Tiling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div>
<!-- thumbnail-parent-div-close --></div>
