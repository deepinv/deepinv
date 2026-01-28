.. _distributed:

Distributed Computing
=====================

For large-scale inverse problems, the memory and compute of a single device might not be enough.
The distributed computing framework enables efficient parallel processing across multiple GPUs by distributing physics operators and computations across multiple processes.

The framework provides an API centered around two key functions:

1. :class:`deepinv.distributed.DistributedContext` - manages distributed execution
2. :func:`deepinv.distributed.distribute` - converts regular objects to distributed versions

.. note::

    The distributed framework is particularly useful when:
    
    - *Multiple physics operators* with individual measurements need to be processed in parallel
    - *Large images* are too large to fit in a single device's memory
    - *Denoising priors* need to be applied to large images using spatial tiling
    - You want to *accelerate reconstruction* by leveraging multiple devices


.. warning::

    This module is in beta and may undergo significant changes in future releases.
    Some features are experimental and only supported for specific use cases.
    Please report any issues you encounter on our `GitHub repository <https://github.com/deepinv/deepinv>`_.

Quick Start
-----------

Here's a minimal example that shows the complete workflow:

.. testcode::

    from deepinv.physics import Blur, stack
    from deepinv.physics.blur import gaussian_blur
    from deepinv.optim.data_fidelity import L2
    from deepinv.models import DRUNet
    from deepinv.distributed import DistributedContext, distribute
    from deepinv.utils.demo import load_example

    # Step 1: Create distributed context
    with DistributedContext() as ctx:

        # Load an example image
        x = load_example(
            "CBSD_0010.png", grayscale=False, device=str(ctx.device)  # Make sure the image is on the correct device
        )

        # Step 2: Create and stack your physics operators
        physics_list = [
            Blur(
                filter=gaussian_blur(sigma=1.0), padding="circular"
            ),
            Blur(
                filter=gaussian_blur(sigma=2.0), padding="circular"
            ),
            Blur(
                filter=gaussian_blur(sigma=3.0), padding="circular"
            ),
        ]
        stacked_physics = stack(*physics_list)

        # Step 3: Distribute physics
        distributed_physics = distribute(stacked_physics, ctx)  # Distribute physics operators, transfers to correct devices

        # Use it like regular physics
        y = distributed_physics(x)  # Forward operation
        x_adj = distributed_physics.A_adjoint(y)  # Adjoint

        # Step 4: Distribute a denoiser for large images
        denoiser = DRUNet()
        distributed_denoiser = distribute(
            denoiser,
            ctx,
            patch_size=256,  # Split image into patches
            overlap=64,  # Overlap for smooth blending
        )

        # Use it like regular denoiser
        denoised = distributed_denoiser(x_adj, sigma=0.1)

        # Step 5: Distribute a data fidelity term
        data_fidelity = L2()
        distributed_data_fidelity = distribute(data_fidelity, ctx)

        # Use it like regular data fidelity
        loss = distributed_data_fidelity.fn(denoised, y, distributed_physics)

        # Step 6: debug and print on rank 0 only
        if ctx.rank == 0:
            print("Distributed physics output shape:", y.shape)
            print("Distributed physics adjoint output shape:", x_adj.shape)
            print("Distributed denoiser output shape:", denoised.shape)
            print(f"Distributed data fidelity loss: {loss.item():.6f}")

.. testoutput::
    :options: +ELLIPSIS

    Distributed physics output shape: [torch.Size([1, 3, 481, 321]), torch.Size([1, 3, 481, 321]), torch.Size([1, 3, 481, 321])]
    Distributed physics adjoint output shape: torch.Size([1, 3, 481, 321])
    Distributed denoiser output shape: torch.Size([1, 3, 481, 321])
    Distributed data fidelity loss: ...

**That's the entire API!** The :func:`deepinv.distributed.distribute()` function handles all the complexity of distributed computing.
You can choose to distribute some components and not others, depending on your needs. 
For instance, you might only want to distribute the denoiser for large images, while keeping the physics and data fidelity local.

When to Use Distributed Computing
----------------------------------

**Multi-Operator Problems**: many inverse problems involve multiple physics operators with corresponding measurements:

- *Multi-view imaging*: Different camera angles or viewpoints
- *Multi-frequency acquisitions*: Different measurement frequencies or channels  
- *Multi-blur deconvolution*: Different blur kernels applied to the same scene
- *Tomography*: Different projection angles

The distributed framework automatically splits these operators across processes, computing forward operations,
adjoints, and data fidelity gradients in parallel.

**Large-Scale Images**: for very large images (e.g., high-resolution medical scans, satellite imagery, radio interferometry),
the distributed framework uses spatial tiling to:

- Split the image into overlapping patches
- Process each patch independently across multiple devices
- Reconstruct the full image with smooth blending at boundaries

This enables handling arbitrarily large images that wouldn't fit in a single device's memory.


Simple Two-Step Pattern
-----------------------

**Step 1: Create a distributed context**

.. code-block:: python

    from deepinv.distributed import DistributedContext
    
    with DistributedContext() as ctx:
        # All distributed operations go here
        pass

The context:

- Works seamlessly in both single-process and multi-process modes
- Automatically initializes process groups when running with ``torchrun`` or on a slurm cluster with one task per gpu.
- Assigns devices based on available GPUs
- Cleans up resources on exit

**Step 2: Distribute your objects**

.. code-block:: python

    # Distribute physics operators
    distributed_physics = distribute(physics, ctx)
    
    # Distribute denoisers with tiling parameters
    distributed_denoiser = distribute(denoiser, ctx, patch_size=256, overlap=64)

    # Distribute data fidelity
    distributed_data_fidelity = distribute(data_fidelity, ctx)

The :func:`deepinv.distributed.distribute()` function:

- Auto-detects the object type (physics, denoiser, prior, data fidelity)
- Creates the appropriate distributed version
- Handles all parallelization logic internally

Distributed Physics
-------------------

Large-scale physics operators can sometimes be separated into blocks:

.. math::
       A(x) = \begin{bmatrix} A_1(x) \\  \vdots \\ A_N(x) \end{bmatrix}

for sub-operators :math:`A_i`.

The distributed framework allows you to compute each sub-operator in parallel to speed up the computation of the global forward or adjoint operator.

.. note::
    Check out the :ref:`distributed physics example <sphx_glr_auto_examples_distributed_demo_physics_distributed.py>` for a complete demo.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from deepinv.physics import Blur, stack
    from deepinv.distributed import DistributedContext, distribute
    
    with DistributedContext() as ctx:
        # Create multiple operators
        physics_list = [operator1, operator2, operator3, ...]
        stacked_physics = stack(*physics_list)
        
        # Distribute them
        dist_physics = distribute(stacked_physics, ctx)
        
        # Use like regular physics
        y = dist_physics(x)              # Forward (parallel)
        x_adj = dist_physics.A_adjoint(y)  # Adjoint (parallel)
        x_ata = dist_physics.A_adjoint_A(x)  # Composition (parallel)

How It Works
~~~~~~~~~~~~

1. **Operator Sharding**: Operators are divided across processes using round-robin assignment
2. **Parallel Forward**: Each process computes :math:`A_i(x)` for its local operators
3. **Parallel Adjoint**: Each process computes local adjoints, then results are summed via ``all_reduce``

Input Formats
~~~~~~~~~~~~~

The ``distribute()`` function accepts multiple formats:

.. code-block:: python

    physics_list = [operator1, operator2, operator3, ...]

    # From StackedPhysics
    stacked = stack(*physics_list)
    dist_physics = distribute(stacked, ctx)
    
    # From list of physics
    dist_physics = distribute(physics_list, ctx)
    
    # From factory function
    def physics_factory(idx, device, shared):
        # idx is the index of the operator to create
        # device is the assigned device for this process
        # shared is a dict for sharing parameters across operators (optional)
        return create_physics(idx, device)
    
    dist_physics = distribute(physics_factory, ctx, num_operators=10)

    # With shared parameters
    shared_params = {"common_param": value}

    dist_physics = distribute(
        physics_factory, ctx, num_operators=10, shared=shared_params
    )

Gather Strategies
~~~~~~~~~~~~~~~~~

You can control how results are gathered from different processes:

.. code-block:: python

    # Concatenated (default): most efficient for similar-sized tensors
    dist_physics = distribute(physics, ctx, gather_strategy="concatenated")
    
    # Naive: simple serialization, good for small tensors
    dist_physics = distribute(physics, ctx, gather_strategy="naive")
    
    # Broadcast: good for heterogeneous sizes
    dist_physics = distribute(physics, ctx, gather_strategy="broadcast")


Distributed Denoisers
---------------------

Denoisers can be distributed using **spatial tiling** to handle large images.

.. note::
    Check out the :ref:`distributed denoiser example <sphx_glr_auto_examples_distributed_demo_denoiser_distributed.py>` for a complete demo.

Basic Usage
~~~~~~~~~~~

.. testcode::

    from deepinv.models import DRUNet
    from deepinv.distributed import DistributedContext, distribute
    
    with DistributedContext() as ctx:
        # Load your denoiser
        denoiser = DRUNet()
        
        # Distribute with tiling parameters
        dist_denoiser = distribute(
            denoiser,
            ctx,
            patch_size=256,           # Size of each patch
            overlap=64,  # Overlap for smooth boundaries
        )

        # Process image
        image = torch.randn(1, 3, 512, 512, device=ctx.device)

        with torch.no_grad():
            denoised = dist_denoiser(image, sigma=0.05)

        if ctx.rank == 0:
            print("Denoised image shape:", denoised.shape)

.. testoutput::

    Denoised image shape: torch.Size([1, 3, 512, 512])

How It Works
~~~~~~~~~~~~

1. **Patch Extraction**: Image is split into overlapping patches
2. **Distributed Processing**: Patches are distributed across processes
3. **Parallel Denoising**: Each process denoises its local patches
4. **Reconstruction**: Patches are blended back into full image, each rank has access to the full output

Tiling Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``patch_size``
     - Size of each patch (default: 256). Larger patches = less communication, more memory
   * - ``overlap``
     - Overlap radius for smooth blending (default: 64).
   * - ``tiling_strategy``
     - Strategy for tiling: ``'overlap_tiling'`` (default), or ``'basic'``
   * - ``max_batch_size``
     - Max patches per batch (default: all). Set to 1 for sequential processing (lowest memory)

Tiling Strategies
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Tiling with overlap (default)
    dist_denoiser = distribute(denoiser, ctx, tiling_strategy="overlap_tiling")
    
    # Basic (no overlap blending)
    dist_denoiser = distribute(denoiser, ctx, tiling_strategy="basic")


Running Multi-Process
---------------------

Use ``torchrun`` to launch multiple processes. Examples:

4 GPUs on one machine:

.. code-block:: bash

    torchrun --nproc_per_node=4 my_script.py

2 machines with 2 GPUs each:

.. code-block:: bash

    # On machine 1 (rank 0)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr="192.168.1.1" --master_port=29500 my_script.py

    # On machine 2 (rank 1):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
             --master_addr="192.168.1.1" --master_port=29500 my_script.py

Alternatively, use the ``-m torch.distributed.run`` syntax to run as a module:

.. code-block:: bash

    python -m torch.distributed.run --nproc_per_node=4 my_script.py

The ``DistributedContext`` automatically detects the settings from environment variables.


Advanced Features
-----------------

Local vs Reduced Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, distributed methods return fully gathered and reduced results (combined from all processes).
You can get local-only results with ``gather=False`` and you can choose to skip reduction with ``reduce_op=None``:

.. code-block:: python

    # Get local results without local reduction
    y_local = dist_physics.A(x, gather=False, reduce_op=None)

    # Get local results with reduction (sum by default)
    y_local = dist_physics.A(x, gather=False)

    # Get gathered results without reduction
    y_gathered = dist_physics.A(x, reduce_op=None)
    
    # Get gathered results (default)
    y_all = dist_physics.A(x)

This is useful for:

- Custom reduction strategies
- Debugging distributed execution
- Optimizing communication patterns

Custom Tiling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

You can implement custom tiling strategies by subclassing
:class:`deepinv.distributed.strategies.DistributedSignalStrategy`:

.. code-block:: python

    from deepinv.distributed.strategies import DistributedSignalStrategy
    
    class MyCustomStrategy(DistributedSignalStrategy):
        def get_local_patches(self, X, local_indices):
            # Your patch extraction logic
            pass
        
        def reduce_patches(self, out_tensor, local_pairs):
            # Your patch reduction logic
            pass
        
        def get_num_patches(self):
            # Total number of patches
            pass
    
    # Use it
    dist_denoiser = distribute(
        denoiser, ctx,
        tiling_strategy=MyCustomStrategy(img_size),
    )


Performance Tips
----------------

**Choosing the Right Number of Processes**

- *Multi-operator problems*: Use as many processes as operators (up to available devices)
- *Spatial tiling*: Balance parallelism vs communication overhead
- *Rule of thumb*: Start with number of GPUs, experiment from there

**Optimizing Patch Size**

- *Larger patches* (512+): Less communication, more memory per process
- *Smaller patches* (128-256): More parallelism, more communication  
- *Recommendation*: 256-512 pixels for deep denoisers on natural images

**Receptive Field Padding**

- Set ``overlap`` to match your denoiser's receptive field
- Ensures smooth blending at patch boundaries
- *Typical values*: 32-64 pixels for U-Net style denoisers

**Gather Strategies**

- *Concatenated* (default): Best for most cases, minimal communication
- *Naive*: Use for small tensors or debugging
- *Broadcast*: Use when operator outputs have very different sizes



Key Classes
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`deepinv.distributed.DistributedContext`
     - Manages distributed execution, process groups, and devices
   * - :class:`deepinv.distributed.DistributedStackedPhysics`
     - Distributes physics operators across processes (auto-created by ``distribute()``)
   * - :class:`deepinv.distributed.DistributedStackedLinearPhysics`
     - Extends DistributedStackedPhysics for linear operators with adjoint operations
   * - :class:`deepinv.distributed.DistributedProcessing`
     - Distributes denoisers/priors using spatial tiling (auto-created by ``distribute()``)
   * - :class:`deepinv.distributed.DistributedDataFidelity`
     - Distributes data fidelity `fn` and `grad`` (if needed, auto-created by ``distribute()``)

**You typically won't need to instantiate these classes directly.** Use the :func:`deepinv.distributed.distribute()` function instead.



Troubleshooting
---------------

**Out of memory errors**

- Reduce ``patch_size`` for distributed denoisers
- Set ``max_batch_size=1`` for sequential patch processing

**Results differ slightly from non-distributed**

- This is normal for tiling strategies due to boundary blending
- Differences are typically very small
- The distributed implementation of ``A_dagger`` and ``compute_norm`` in ``LinearDistributedPhysics`` uses approximations that lead to differences compared to the non-distributed versions.


See Also
--------

- **API Reference**: :doc:`/api/deepinv.distributed`
- **Examples**: 

  - :ref:`sphx_glr_auto_examples_distributed_demo_physics_distributed.py`
  - :ref:`sphx_glr_auto_examples_distributed_demo_denoiser_distributed.py`
  - :ref:`sphx_glr_auto_examples_distributed_demo_pnp_distributed.py`

- **Related**: 
  
  - :class:`deepinv.physics.StackedPhysics` for multi-operator physics
  - :ref:`Optimization algorithms <optim>` for reconstruction methods
