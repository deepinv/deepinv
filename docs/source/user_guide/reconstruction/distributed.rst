.. _distributed:

Distributed Computing
=====================

For large-scale inverse problems, single-device memory and compute limitations can become a bottleneck.
The distributed computing framework enables efficient parallel processing across multiple GPUs or CPU cores
by distributing physics operators, measurements, and computations across multiple processes.

.. note::

    The distributed framework is particularly useful when:
    
    - Multiple physics operators with individual measurements need to be processed in parallel
    - Images are too large to fit in a single device's memory
    - Denoising priors need to be applied to large images using spatial tiling
    - You want to accelerate reconstruction by leveraging multiple devices


When to Use Distributed Computing
----------------------------------

**Multi-Operator Problems**

Many inverse problems involve multiple physics operators with corresponding measurements. For example:

- **Multi-view imaging**: Different camera angles or viewpoints
- **Multi-frequency acquisitions**: Different measurement frequencies
- **Multi-blur deconvolution**: Different blur kernels applied to the same scene

The distributed framework automatically splits these operators and measurements across processes,
computing gradients and data fidelity terms in parallel.

**Large-Scale Signals**

For very large images (e.g., high-resolution medical scans, radio interferometry), the distributed framework
uses spatial tiling to:

- Split the image into overlapping patches
- Process each patch with a denoising prior independently
- Reconstruct the full image by combining processed patches

This enables handling arbitrarily large images that wouldn't fit in a single device's memory.


Core Concepts
-------------

The distributed framework is built around several key components:

.. list-table::
   :header-rows: 1

   * - Component
     - Purpose
   * - :class:`~deepinv.distrib.DistributedContext`
     - Manages process groups, device assignment, and communication
   * - :class:`~deepinv.distrib.DistributedPhysics`
     - Distributes physics operators across processes
   * - :class:`~deepinv.distrib.DistributedMeasurements`
     - Distributes measurement data across processes
   * - :class:`~deepinv.distrib.DistributedDataFidelity`
     - Computes data fidelity gradients in parallel and reduces them
   * - :class:`~deepinv.distrib.DistributedSignal`
     - Represents the signal (image) being reconstructed, synchronized across processes
   * - :class:`~deepinv.distrib.DistributedPrior`
     - Applies priors using spatial tiling for large images


Quick Start Example
-------------------

Here's a minimal example showing how to use the distributed framework for PnP reconstruction
with multiple blur operators:

.. code-block:: python

    import torch
    from deepinv.physics import Blur, GaussianNoise, stack
    from deepinv.physics.blur import gaussian_blur
    from deepinv.optim import L2, PnP
    from deepinv.models import DRUNet
    from deepinv.distrib import (
        DistributedContext,
        FactoryConfig,
        TilingConfig,
        make_distrib_bundle,
    )

    # Initialize distributed context (works for single or multi-process)
    with DistributedContext(seed=42) as ctx:
        
        # Create multiple physics operators
        kernels = [
            gaussian_blur(sigma=1.0),
            gaussian_blur(sigma=2.0),
            gaussian_blur(sigma=(1.5, 3.0), angle=30),
        ]
        
        physics_list = []
        for kernel in kernels:
            blur_op = Blur(filter=kernel, padding="circular")
            blur_op.noise_model = GaussianNoise(sigma=0.03)
            physics_list.append(blur_op)
        
        # Stack physics operators and generate measurements
        stacked_physics = stack(*physics_list)
        measurements = stacked_physics(clean_image)
        
        # Configure distributed components
        factory_config = FactoryConfig(
            physics=stacked_physics,
            measurements=measurements,
            data_fidelity=L2(),
        )
        
        tiling_config = TilingConfig(
            patch_size=128,
            receptive_field_size=32,
        )
        
        # Build distributed bundle
        denoiser = DRUNet(pretrained="download")
        pnp_prior = PnP(denoiser=denoiser)
        
        bundle = make_distrib_bundle(
            ctx,
            factory_config=factory_config,
            signal_shape=clean_image.shape,
            prior=pnp_prior,
            tiling=tiling_config,
        )
        
        # Run PnP iterations
        bundle.signal.update_(torch.zeros_like(clean_image))
        
        for it in range(num_iterations):
            # Data fidelity gradient (distributed)
            grad = bundle.data_fidelity.grad(bundle.signal)
            
            # Gradient step
            new_data = bundle.signal.data - lr * grad
            bundle.signal.update_(new_data)
            
            # Denoising step (distributed)
            denoised = bundle.prior.prox(bundle.signal, sigma_denoiser=0.05)
            bundle.signal.update_(denoised)
        
        # Final result is in bundle.signal.data


Detailed Workflow
-----------------

1. Distributed Context Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~deepinv.distrib.DistributedContext` manages all distributed computing aspects:

.. code-block:: python

    from deepinv.distrib import DistributedContext
    
    with DistributedContext(
        backend=None,           # Auto-selects NCCL for GPU, Gloo for CPU
        sharding='round_robin', # or 'block' for contiguous sharding
        seed=42,                # Reproducible but rank-specific RNG
        device_mode=None,       # Auto-selects GPU/CPU, or force 'cpu'/'gpu'
    ) as ctx:
        # ctx.rank: current process rank
        # ctx.world_size: total number of processes
        # ctx.device: assigned device for this process
        # ctx.is_dist: whether running in distributed mode
        
        # Your distributed code here
        pass

The context automatically:

- Initializes the process group if ``RANK`` and ``WORLD_SIZE`` environment variables are set
- Assigns devices based on ``LOCAL_RANK`` and available GPUs
- Cleans up the process group on exit
- Works seamlessly in single-process mode (no process group needed)

**Running Multi-Process:**

.. code-block:: bash

    # Using torchrun
    torchrun --nproc_per_node=4 your_script.py
    
    # Using torch.distributed.run
    python -m torch.distributed.run --nproc_per_node=4 your_script.py


2. Creating Distributed Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Factory API** provides the simplest way to create all distributed components at once:

.. code-block:: python

    from deepinv.distrib import FactoryConfig, TilingConfig, make_distrib_bundle
    
    # Configure physics and measurements
    factory_config = FactoryConfig(
        physics=stacked_physics,      # StackedPhysics or list of Physics
        measurements=measurements,     # TensorList or list of tensors
        data_fidelity=L2(),           # Optional, defaults to L2
    )
    
    # Configure spatial tiling for priors
    tiling_config = TilingConfig(
        patch_size=256,               # Size of each patch
        receptive_field_size=64,      # Padding for overlap
        overlap=False,                # Whether patches overlap
        strategy='smart_tiling',      # or 'basic'
    )
    
    # Build everything at once
    bundle = make_distrib_bundle(
        ctx,
        factory_config=factory_config,
        signal_shape=(B, C, H, W),
        prior=pnp_prior,              # Optional
        tiling=tiling_config,         # Optional, only needed if prior is provided
    )
    
    # Access components
    bundle.physics         # DistributedLinearPhysics
    bundle.measurements    # DistributedMeasurements
    bundle.data_fidelity   # DistributedDataFidelity
    bundle.signal          # DistributedSignal
    bundle.prior           # DistributedPrior (if provided)

**Alternative: Manual Creation**

For more control, you can create components individually:

.. code-block:: python

    from deepinv.distrib import (
        DistributedLinearPhysics,
        DistributedMeasurements,
        DistributedDataFidelity,
        DistributedSignal,
        DistributedPrior,
    )
    
    # Define factory functions
    def physics_factory(idx, device, shared):
        return physics_list[idx].to(device)
    
    def measurements_factory(idx, device, shared):
        return measurements[idx].to(device)
    
    # Create distributed physics
    dphysics = DistributedLinearPhysics(
        ctx, num_ops=len(physics_list), factory=physics_factory
    )
    
    # Create distributed measurements
    dmeasurements = DistributedMeasurements(
        ctx, num_items=len(measurements), factory=measurements_factory
    )
    
    # Create distributed data fidelity
    ddata_fidelity = DistributedDataFidelity(
        ctx, dphysics, dmeasurements, data_fidelity_factory=lambda idx, dev, s: L2()
    )
    
    # Create distributed signal
    dsignal = DistributedSignal(ctx, shape=(B, C, H, W))


3. Distributed Data Fidelity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~deepinv.distrib.DistributedDataFidelity` class computes data fidelity gradients in parallel:

.. math::

    \nabla f(x) = \sum_{i=1}^{N} \nabla f_i(x)

where each :math:`f_i(x) = \distance{A_i(x)}{y_i}` is computed on a different process, and the results
are summed via ``allreduce``.

.. code-block:: python

    # Each process computes gradients for its local operators
    grad = bundle.data_fidelity.grad(bundle.signal)
    
    # grad is automatically reduced across all processes
    # and is identical on all ranks
    
    # You can also compute the full loss
    loss = bundle.data_fidelity(bundle.signal)

**How it works:**

1. Each process has a subset of physics operators and measurements (via sharding)
2. When ``grad()`` is called, each process computes gradients for its local operators
3. Gradients are summed across all processes using ``dist.all_reduce()``
4. The final gradient represents the sum over all operators


4. Distributed Priors with Spatial Tiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large images, the :class:`~deepinv.distrib.DistributedPrior` uses spatial tiling to distribute
denoising computations:

.. code-block:: python

    # Create distributed prior
    dprior = DistributedPrior(
        ctx=ctx,
        prior=pnp_prior,
        strategy='smart_tiling',
        signal_shape=(B, C, H, W),
        strategy_kwargs={
            'patch_size': 256,
            'receptive_field_size': 64,
        }
    )
    
    # Apply denoising
    denoised = dprior.prox(signal, sigma_denoiser=0.05)

**How it works:**

1. The image is split into overlapping patches (with receptive field padding)
2. Patches are distributed across processes via round-robin or block sharding
3. Each process denoise its local patches
4. Patches are reduced back to the full image with proper blending in overlap regions

**Available Strategies:**

.. list-table::
   :header-rows: 1

   * - Strategy
     - Description
     - Best For
   * - ``'smart_tiling'``
     - Uniform patches with padding, efficient batching
     - Large images, deep priors with large receptive fields
   * - ``'basic'``
     - Simple splitting along specified dimensions
     - Custom splitting patterns


5. Distributed Signal
~~~~~~~~~~~~~~~~~~~~~

The :class:`~deepinv.distrib.DistributedSignal` represents the reconstruction signal and keeps it synchronized
across all processes:

.. code-block:: python

    # Create signal
    signal = DistributedSignal(ctx, shape=(B, C, H, W))
    
    # Initialize with data
    signal.update_(torch.zeros(B, C, H, W, device=ctx.device))
    
    # Access the data
    current_data = signal.data
    
    # Update with new data (automatically broadcasts to all ranks)
    signal.update_(new_data)

The signal ensures all processes have the same reconstruction state, which is essential for distributed gradients.


Complete PnP Example
--------------------

Here's a complete example of distributed PnP reconstruction:

.. code-block:: python

    import torch
    from deepinv.physics import Blur, GaussianNoise, stack
    from deepinv.physics.blur import gaussian_blur
    from deepinv.optim import L2, PnP
    from deepinv.models import DRUNet
    from deepinv.distrib import (
        DistributedContext,
        FactoryConfig,
        TilingConfig,
        make_distrib_bundle,
    )
    
    def main():
        # Configuration
        num_iterations = 10
        lr = 1.0
        denoiser_sigma = 0.05
        img_size = (512, 512)
        
        with DistributedContext(seed=42) as ctx:
            # Load or create clean image
            clean_image = load_your_image(img_size, device=ctx.device)
            
            # Create multiple blur operators
            kernels = [
                gaussian_blur(sigma=1.0, device=str(ctx.device)),
                gaussian_blur(sigma=2.0, device=str(ctx.device)),
                gaussian_blur(sigma=(1.5, 3.0), angle=30, device=str(ctx.device)),
            ]
            
            physics_list = []
            for kernel in kernels:
                blur_op = Blur(filter=kernel, padding="circular")
                blur_op.noise_model = GaussianNoise(sigma=0.03)
                physics_list.append(blur_op.to(ctx.device))
            
            # Stack and generate measurements
            stacked_physics = stack(*physics_list)
            measurements = stacked_physics(clean_image)
            
            # Load denoiser
            denoiser = DRUNet(pretrained="download").to(ctx.device)
            pnp_prior = PnP(denoiser=denoiser)
            
            # Configure and build distributed components
            factory_config = FactoryConfig(
                physics=stacked_physics,
                measurements=measurements,
                data_fidelity=L2(),
            )
            
            tiling_config = TilingConfig(
                patch_size=128,
                receptive_field_size=32,
            )
            
            bundle = make_distrib_bundle(
                ctx,
                factory_config=factory_config,
                signal_shape=clean_image.shape,
                prior=pnp_prior,
                tiling=tiling_config,
            )
            
            # Initialize reconstruction
            bundle.signal.update_(torch.zeros_like(clean_image))
            
            # PnP iterations
            with torch.no_grad():
                for it in range(num_iterations):
                    # Data fidelity gradient step
                    grad = bundle.data_fidelity.grad(bundle.signal)
                    new_data = bundle.signal.data - lr * grad
                    bundle.signal.update_(new_data)
                    
                    # Denoising step
                    denoised = bundle.prior.prox(
                        bundle.signal, sigma_denoiser=denoiser_sigma
                    )
                    bundle.signal.update_(denoised)
                    
                    # Log progress (only on rank 0)
                    if ctx.rank == 0 and (it + 1) % 2 == 0:
                        print(f"Iteration {it+1}/{num_iterations}")
            
            # Get final result
            reconstruction = bundle.signal.data
            
            # Save or visualize (only on rank 0)
            if ctx.rank == 0:
                save_image(reconstruction, "result.png")
    
    if __name__ == "__main__":
        main()


Running the script:

.. code-block:: bash

    # Single process
    python script.py
    
    # Multiple processes
    torchrun --nproc_per_node=4 script.py


Advanced Features
-----------------


Custom Distribution Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can implement custom spatial distribution strategies by subclassing
:class:`~deepinv.distrib.distribution_strategies.strategies.DistributedSignalStrategy`:

.. code-block:: python

    from deepinv.distrib.distribution_strategies.strategies import (
        DistributedSignalStrategy
    )
    
    class MyCustomStrategy(DistributedSignalStrategy):
        def __init__(self, signal_shape, **kwargs):
            super().__init__(signal_shape)
            # Your initialization
        
        def get_local_patches(self, X, local_indices):
            # Extract patches for this rank
            pass
        
        def apply_batching(self, patches):
            # Batch patches for efficient processing
            pass
        
        def reduce_patches(self, out_tensor, local_pairs):
            # Reduce patches back to full tensor
            pass
        
        def get_num_patches(self):
            # Return total number of patches
            pass


Performance Tips
----------------

**1. Choose the Right Number of Processes**

- For multi-operator problems: Use as many processes as you have operators (up to available devices)
- For spatial tiling: Balance between parallelism and communication overhead
- Rule of thumb: Start with the number of GPUs you have

**2. Optimize Patch Size**

- Larger patches: Less communication, more memory per process
- Smaller patches: More parallelism, more communication
- Recommended: 128-512 pixels for natural images with deep denoisers

**3. Use Receptive Field Padding**

- Set ``receptive_field_size`` to match your denoiser's receptive field
- This ensures proper blending at patch boundaries
- Typical values: 32-64 pixels for U-Net style denoisers

**4. Monitor Communication**

- Most communication happens in ``all_reduce`` operations
- Minimize the number of ``signal.update_()`` calls



See Also
--------

- :doc:`API Reference </api/deepinv.distrib>`
- Example: :ref:`sphx_glr_auto_examples_distrib_demo_pnp_distributed.py`
- :class:`deepinv.physics.StackedPhysics` for multi-operator physics
- :ref:`Optimization algorithms <optim>` for non-distributed reconstruction
