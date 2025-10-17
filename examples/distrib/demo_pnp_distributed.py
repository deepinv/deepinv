"""
Large-scale plug-and-play methods using distributed computing
===============================================

This example demonstrates how to use the distributed framework for PnP reconstruction.
The framework automatically distributes physics operators and priors across multiple processes.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distrib/demo_pnp_distributed.py

    # Multi-process with torchrun
    python -m torch.distributed.run --nproc_per_node=2 examples/distrib/demo_pnp_distributed.py

**Key Steps:**

1. Create physics operators and measurements
2. Initialize distributed context
3. Configure distributed components with FactoryConfig and TilingConfig
4. Build distributed bundle with make_distrib_bundle()
5. Run PnP iterations
6. Visualize results
"""

import torch
from deepinv.physics import GaussianNoise, stack
from deepinv.physics.blur import Blur, gaussian_blur
from deepinv.utils.demo import load_example
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.loss.metric import PSNR
from deepinv.utils.plotting import plot
from deepinv.models import DRUNet

# Import the distributed framework
from deepinv.distrib import (
    DistributedContext,
    FactoryConfig,
    TilingConfig,
    make_distrib_bundle,
)


# ============================================================================
# DATA SETUP
# ============================================================================

def create_physics_and_measurements(device, img_size=(256, 256)):
    """
    Create stacked physics operators and measurements using example images.
    
    :param device: Device to create operators on
    :param tuple img_size: Size of the image (H, W)
    
    :returns: Tuple of (stacked_physics, measurements, clean_image)
    """
    # Load example image
    clean_image = load_example(
        "CBSD_0010.png", grayscale=False, device=device, img_size=img_size
    )
    
    # Create different Gaussian blur kernels
    kernels = [
        gaussian_blur(sigma=1.0, device=str(device)),           # Small blur
        gaussian_blur(sigma=2.0, device=str(device)),           # Medium blur  
        gaussian_blur(sigma=(1.5, 3.0), angle=30, device=str(device))  # Anisotropic blur
    ]
    
    # Noise levels for each operator
    noise_levels = [0.03, 0.05, 0.04]
    
    # Create physics operators
    physics_list = []
    
    for kernel, noise_level in zip(kernels, noise_levels):
        # Create blur operator with circular padding
        blur_op = Blur(filter=kernel, padding="circular", device=str(device))
        
        # Set the noise model
        blur_op.noise_model = GaussianNoise(sigma=noise_level)
        blur_op = blur_op.to(device)
        
        physics_list.append(blur_op)
    
    # Stack physics operators into a single operator
    stacked_physics = stack(*physics_list)
    
    # Generate measurements (returns a TensorList)
    measurements = stacked_physics(clean_image)
    
    return stacked_physics, measurements, clean_image


def main():
    """Run distributed PnP reconstruction."""
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    num_iterations = 10
    lr = 1
    denoiser_sigma = 0.05
    img_size = (512, 512)
    patch_size = 128
    receptive_field_size = 32
    
    # ============================================================================
    # DISTRIBUTED CONTEXT
    # ============================================================================
    
    # Initialize distributed context (handles single and multi-process automatically)
    with DistributedContext(seed=42) as ctx:
        
        if ctx.rank == 0:
            print("=" * 70)
            print("🚀 Distributed PnP Reconstruction")
            print("=" * 70)
            print(f"\n📊 Running on {ctx.world_size} process(es)")
            print(f"   Device: {ctx.device}")
        
        # ============================================================================
        # STEP 1: Create stacked physics operators and measurements
        # ============================================================================
        
        stacked_physics, measurements, clean_image = create_physics_and_measurements(
            ctx.device, img_size=img_size
        )
        
        if ctx.rank == 0:
            print(f"\n✅ Created stacked physics with {len(stacked_physics)} operators")
            print(f"   Image shape: {clean_image.shape}")
            print(f"   Measurements type: {type(measurements).__name__}")
        
        # ============================================================================
        # STEP 2: Load denoiser model and create PnP prior
        # ============================================================================
        
        # PnP prior with denoiser
        denoiser = DRUNet(pretrained="download").to(ctx.device)
        pnp_prior = PnP(denoiser=denoiser)
        
        # ============================================================================
        # STEP 3: Configure distributed components
        # ============================================================================
        
        # Factory configuration: stacked physics, measurements, and data fidelity
        # The framework automatically extracts individual operators from StackedPhysics
        factory_config = FactoryConfig(
            physics=stacked_physics,
            measurements=measurements,
            data_fidelity=L2(),
        )
        
        # Tiling configuration: how to split the image for distributed processing
        tiling_config = TilingConfig(
            patch_size=patch_size,
            receptive_field_size=receptive_field_size,
        )
        
        if ctx.rank == 0:
            print(f"\n🔧 Configured distributed components")
            print(f"   Patch size: {patch_size}x{patch_size}")
            print(f"   Receptive field radius: {receptive_field_size}")
        
        # ============================================================================
        # STEP 4: Build distributed bundle
        # ============================================================================
        
        B, C, H, W = clean_image.shape
        
        distributed_bundle = make_distrib_bundle(
            ctx,
            factory_config=factory_config,
            signal_shape=(B, C, H, W),
            prior=pnp_prior,
            tiling=tiling_config,
        )
        
        if ctx.rank == 0:
            print(f"\n🏗️ Built distributed bundle")
            print(f"   Local physics operators: {len(distributed_bundle.physics.local_idx)}")
            print(f"   Local measurements: {len(distributed_bundle.measurements.local)}")
        
        # ============================================================================
        # STEP 5: Run distributed PnP algorithm
        # ============================================================================
        
        if ctx.rank == 0:
            print(f"\n🔄 Running PnP reconstruction ({num_iterations} iterations)...")
        
        # Extract components
        distributed_signal = distributed_bundle.signal
        distributed_data_fidelity = distributed_bundle.data_fidelity
        distributed_prior = distributed_bundle.prior
        
        # Initialize with zeros
        distributed_signal.update_(torch.zeros_like(clean_image))
        
        # Track PSNR (only on rank 0)
        psnr_metric = PSNR()
        psnr_history = []
        
        # PnP iterations
        with torch.no_grad():
            for it in range(num_iterations):
                # Data fidelity gradient (distributed across processes)
                grad = distributed_data_fidelity.grad(distributed_signal)
                
                # Gradient descent
                new_data = distributed_signal.data - lr * grad
                distributed_signal.update_(new_data)
                
                # Denoising (distributed prior)
                if distributed_prior is not None:
                    denoised = distributed_prior.prox(
                        distributed_signal, sigma_denoiser=denoiser_sigma
                    )
                    distributed_signal.update_(denoised)
                
                # Compute PSNR on rank 0
                if ctx.rank == 0:
                    psnr_val = psnr_metric(distributed_signal.data, clean_image).item()
                    psnr_history.append(psnr_val)
                    
                    if it == 0 or (it + 1) % 2 == 0:
                        print(f"   Iteration {it+1}/{num_iterations}, PSNR: {psnr_val:.2f} dB")
        
        # ============================================================================
        # STEP 6: Visualize results (only on rank 0)
        # ============================================================================
        
        if ctx.rank == 0:
            print(f"\n✅ Reconstruction completed!")
            print(f"   Final PSNR: {psnr_history[-1]:.2f} dB")
            
            # Plot results
            reconstruction = distributed_signal.data
            
            plot(
                [clean_image, measurements[0], reconstruction],
                titles=["Ground Truth", "Measurement", "Reconstruction"],
                save_fn="distributed_pnp_result.png",
                figsize=(12, 4),
            )
            
            print(f"\n📊 Results saved to distributed_pnp_result.png")
            print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
