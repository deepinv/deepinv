.. _distributed:

Distributed Computing
=====================

For large-scale inverse problems, single-device memory and compute limitations can become a bottleneck.
The distributed computing framework enables efficient parallel processing across multiple GPUs by distributing physics operators and computations across multiple processes.

The framework provides an API centered around two key functions:

1. :class:`~deepinv.distrib.DistributedContext` - manages distributed execution  
2. :func:`~deepinv.distrib.distribute` - converts regular objects to distributed versions

.. note::

    The distributed framework is particularly useful when:
    
    - **Multiple physics operators** with individual measurements need to be processed in parallel
    - **Large images** are too large to fit in a single device's memory  
    - **Denoising priors** need to be applied to large images using spatial tiling
    - You want to **accelerate reconstruction** by leveraging multiple devices


Quick Start
-----------

Here's a minimal example that shows the complete workflow:

.. code-block:: python

    import torch
    from deepinv.physics import Blur, stack
    from deepinv.physics.blur import gaussian_blur
    from deepinv.models import DRUNet
    from deepinv.distrib import DistributedContext, distribute
    
    # Step 1: Create distributed context
    with DistributedContext() as ctx:
        
        # Step 2: Create and stack your physics operators  
        physics_list = [
            Blur(filter=gaussian_blur(sigma=1.0), padding="circular"),
            Blur(filter=gaussian_blur(sigma=2.0), padding="circular"),
            Blur(filter=gaussian_blur(sigma=3.0), padding="circular"),
        ]
        stacked_physics = stack(*physics_list)
        
        # Step 3: Distribute physics - that's it!
        distributed_physics = distribute(stacked_physics, ctx)
        
        # Use it like regular physics
        y = distributed_physics(x)  # Forward operation (parallel across operators)
        x_adj = distributed_physics.A_adjoint(y)  # Adjoint (parallel)
        
        # Step 4: Distribute a denoiser for large images
        denoiser = DRUNet()
        distributed_denoiser = distribute(
            denoiser,
            ctx, 
            patch_size=256,           # Split image into patches
            receptive_field_size=64,  # Overlap for smooth blending
        )
        
        # Use it like regular denoiser
        denoised = distributed_denoiser(noisy_image)

**That's the entire API!** The ``distribute()`` function handles all the complexity of distributed computing.


When to Use Distributed Computing
----------------------------------

**Multi-Operator Problems**

Many inverse problems involve multiple physics operators with corresponding measurements:

- **Multi-view imaging**: Different camera angles or viewpoints
- **Multi-frequency acquisitions**: Different measurement frequencies or channels  
- **Multi-blur deconvolution**: Different blur kernels applied to the same scene
- **Tomography**: Different projection angles

The distributed framework automatically splits these operators across processes, computing forward operations,
adjoints, and data fidelity gradients **in parallel**.

**Large-Scale Images**

For very large images (e.g., high-resolution medical scans, satellite imagery, radio interferometry),
the distributed framework uses **spatial tiling** to:

- Split the image into overlapping patches
- Process each patch independently across multiple devices
- Reconstruct the full image with smooth blending at boundaries

This enables handling **arbitrarily large images** that wouldn't fit in a single device's memory.


Core Components
---------------

Simple Two-Step Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Create a distributed context**

.. code-block:: python

    from deepinv.distrib import DistributedContext
    
    with DistributedContext() as ctx:
        # All distributed operations go here
        pass

The context:

- Works seamlessly in **both single-process and multi-process** modes
- Automatically initializes process groups when running with ``torchrun`` or on a slurm cluster with one task per gpu.
- Assigns devices based on available GPUs
- Cleans up resources on exit

**Step 2: Distribute your objects**

.. code-block:: python

    # Distribute physics operators
    distributed_physics = distribute(physics, ctx)
    
    # Distribute denoisers/priors
    distributed_denoiser = distribute(denoiser, ctx, patch_size=256)

    # Disitribute data fidelity (if needed)
    distributed_data_fidelity = distribute(data_fidelity, ctx)

The ``distribute()`` function:

- **Auto-detects** the object type (physics, denoiser, prior, data fidelity)
- Creates the appropriate distributed version
- Handles all parallelization logic internally

Key Classes
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - :class:`~deepinv.distrib.DistributedContext`
     - Manages distributed execution, process groups, and devices
   * - :class:`~deepinv.distrib.DistributedPhysics`
     - Distributes physics operators across processes (auto-created by ``distribute()``)
   * - :class:`~deepinv.distrib.DistributedLinearPhysics`
     - Extends DistributedPhysics for linear operators with adjoint operations
   * - :class:`~deepinv.distrib.DistributedProcessing`
     - Distributes denoisers/priors using spatial tiling (auto-created by ``distribute()``)
   * - :class:`~deepinv.distrib.DistributedDataFidelity`
     - Distributes data fidelity fn and grad (if needed, auto-created by ``distribute()``)


Distributed Physics
-------------------

Physics operators can be distributed across processes to parallelize forward and adjoint operations.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from deepinv.physics import Blur, stack
    from deepinv.distrib import DistributedContext, distribute
    
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
2. **Parallel Forward**: Each process computes ``A_i(x)`` for its local operators
3. **Parallel Adjoint**: Each process computes local adjoints, then results are summed via ``all_reduce``

Input Formats
~~~~~~~~~~~~~

The ``distribute()`` function accepts multiple formats:

.. code-block:: python

    # From StackedPhysics
    stacked = stack(*physics_list)
    dist_physics = distribute(stacked, ctx)
    
    # From list of physics
    dist_physics = distribute(physics_list, ctx)
    
    # From factory function
    def physics_factory(idx, device, shared):
        return create_physics(idx, device)
    
    dist_physics = distribute(physics_factory, ctx, num_operators=10)

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


Distributed Denoisers & Priors
-------------------------------

Denoisers and priors can be distributed using **spatial tiling** to handle large images.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from deepinv.models import DRUNet
    from deepinv.distrib import DistributedContext, distribute
    
    with DistributedContext() as ctx:
        # Load your denoiser
        denoiser = DRUNet(pretrained="download").to(ctx.device)
        
        # Distribute with tiling parameters
        dist_denoiser = distribute(
            denoiser,
            ctx,
            patch_size=256,           # Size of each patch
            receptive_field_size=64,  # Overlap for smooth boundaries
        )
        
        # Process large images
        large_image = torch.randn(1, 3, 2048, 2048, device=ctx.device)
        denoised = dist_denoiser(large_image, sigma=0.05)

How It Works
~~~~~~~~~~~~

1. **Patch Extraction**: Image is split into overlapping patches
2. **Distributed Processing**: Patches are distributed across processes
3. **Parallel Denoising**: Each process denoises its local patches
4. **Reconstruction**: Patches are blended back into full image with smooth transitions

Tiling Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``patch_size``
     - Size of each patch (default: 256). Larger patches = less communication, more memory
   * - ``receptive_field_size``
     - Overlap radius for smooth blending (default: 64). Should match denoiser's receptive field
   * - ``tiling_strategy``
     - Strategy for tiling: ``'smart_tiling'`` (default), or ``'basic'``
   * - ``max_batch_size``
     - Max patches per batch (default: all). Set to 1 for sequential processing (lowest memory)

Tiling Strategies
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Tiling with overlap (default)
    dist_denoiser = distribute(denoiser, ctx, tiling_strategy="smart_tiling")
    
    # Basic (no overlap blending)
    dist_denoiser = distribute(denoiser, ctx, tiling_strategy="basic")


Complete PnP Example
---------------------

Here's a complete example of distributed PnP reconstruction:

.. code-block:: python

    import torch
    import torch.nn.functional as F
    from deepinv.physics import Blur, GaussianNoise, stack
    from deepinv.physics.blur import gaussian_blur
    from deepinv.models import DRUNet
    from deepinv.optim.data_fidelity import L2
    from deepinv.distrib import DistributedContext, distribute
    
    with DistributedContext(seed=42) as ctx:
        
        # ===== Setup =====
        
        # Create multiple physics operators
        kernels = [
            gaussian_blur(sigma=1.0, device=str(ctx.device)),
            gaussian_blur(sigma=2.0, device=str(ctx.device)),
            gaussian_blur(sigma=(1.5, 3.0), angle=30, device=str(ctx.device)),
        ]
        
        physics_list = []
        for kernel in kernels:
            blur = Blur(filter=kernel, padding="circular", device=str(ctx.device))
            blur.noise_model = GaussianNoise(sigma=0.03)
            physics_list.append(blur)
        
        stacked_physics = stack(*physics_list)
        
        # Generate measurements
        clean_image = load_image()  # Your image loading function
        measurements = stacked_physics(clean_image)
        
        # ===== Distribute Components =====
        
        # Distribute physics
        dist_physics = distribute(stacked_physics, ctx)
        
        # Distribute denoiser
        denoiser = DRUNet(pretrained="download").to(ctx.device)
        dist_denoiser = distribute(
            denoiser, ctx,
            patch_size=256,
            receptive_field_size=64,
        )
        
        # Create data fidelity (not distributed, works with distributed physics)
        data_fidelity = L2()
        
        # ===== PnP Iterations =====
        
        x = torch.zeros_like(clean_image)
        step_size = 0.5
        denoiser_sigma = 0.05
        
        for iteration in range(20):
            # Data fidelity gradient (uses distributed physics)
            grad = data_fidelity.grad(x, measurements, dist_physics)
            
            # Gradient step
            x = x - step_size * grad
            
            # Denoising step (distributed)
            x = dist_denoiser(x, sigma=denoiser_sigma)
            
            if ctx.rank == 0:
                print(f"Iteration {iteration+1}/20")
        
        # Final result
        if ctx.rank == 0:
            save_result(x)


Running Multi-Process
---------------------

Single Process (Development/Testing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just run your script normally:

.. code-block:: bash

    python my_script.py

The framework detects single-process mode and disables distributed features automatically.

Multi-Process (Production)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``torchrun`` to launch multiple processes:

.. code-block:: bash

    # 4 processes on one machine
    torchrun --nproc_per_node=4 my_script.py

    # 2 machines with 4 GPUs each
    # On machine 1 (rank 0):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr="192.168.1.1" --master_port=29500 my_script.py
    
    # On machine 2 (rank 1):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
             --master_addr="192.168.1.1" --master_port=29500 my_script.py

The ``DistributedContext`` automatically detects these settings from environment variables.


Advanced Features
-----------------

Local vs Reduced Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, distributed methods return fully reduced results (combined from all processes).
You can get local-only results with ``reduce=False``:

.. code-block:: python

    # Get local results only (no communication)
    y_local = dist_physics.A(x, reduce=False)  # List of local measurements
    
    # Get reduced results (default, with communication)
    y_all = dist_physics.A(x, reduce=True)  # TensorList of all measurements

This is useful for:

- Custom reduction strategies
- Debugging distributed execution
- Optimizing communication patterns

Custom Tiling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

You can implement custom tiling strategies by subclassing
:class:`~deepinv.distrib.distribution_strategies.strategies.DistributedSignalStrategy`:

.. code-block:: python

    from deepinv.distrib.distribution_strategies.strategies import DistributedSignalStrategy
    
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
        tiling_strategy=MyCustomStrategy(signal_shape),
    )


Performance Tips
----------------

**Choosing the Right Number of Processes**

- **Multi-operator problems**: Use as many processes as operators (up to available devices)
- **Spatial tiling**: Balance parallelism vs communication overhead
- **Rule of thumb**: Start with number of GPUs, experiment from there

**Optimizing Patch Size**

- **Larger patches** (512+): Less communication, more memory per process
- **Smaller patches** (128-256): More parallelism, more communication  
- **Recommendation**: 256-512 pixels for deep denoisers on natural images

**Receptive Field Padding**

- Set ``receptive_field_size`` to match your denoiser's receptive field
- Ensures smooth blending at patch boundaries
- **Typical values**: 32-64 pixels for U-Net style denoisers

**Gather Strategies**

- **Concatenated** (default): Best for most cases, minimal communication
- **Naive**: Use for small tensors or debugging
- **Broadcast**: Use when operator outputs have very different sizes


Troubleshooting
---------------

**Out of memory errors**

- Reduce ``patch_size`` for distributed denoisers
- Set ``max_batch_size=1`` for sequential patch processing

**Results differ slightly from non-distributed**

- This is normal for tiling strategies due to boundary blending
- Differences are typically very small
- The distributed implementation of ``A_dagger``and ``compute_norm``in ``LinearDistributedPhysics`` uses approximations that lead to differences compared to the non-distributed versions.


See Also
--------

- **API Reference**: :doc:`/api/deepinv.distrib`
- **Examples**: 
  
  - :ref:`sphx_glr_auto_examples_distrib_demo_physics_distributed.py`
  - :ref:`sphx_glr_auto_examples_distrib_demo_denoiser_distributed.py`
  - :ref:`sphx_glr_auto_examples_distrib_demo_pnp_distributed.py`

- **Related**: 
  
  - :class:`deepinv.physics.StackedPhysics` for multi-operator physics
  - :ref:`Optimization algorithms <optim>` for reconstruction methods
