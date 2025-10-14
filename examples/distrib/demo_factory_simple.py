"""
Distributed Framework Factory API Demo
======================================

This example demonstrates the simplified factory API for DeepInverse's distributed framework.
The factory API reduces boilerplate code by using configuration objects and builder functions.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distrib/demo_factory_simple.py

    # Multi-process with torchrun
    torchrun --nproc_per_node=2 examples/distrib/demo_factory_simple.py

**Key Concepts:**

- **FactoryConfig**: Simplified configuration for physics, measurements, and data fidelity
- **TilingConfig**: Configuration for spatial tiling strategies
- **make_distrib_bundle**: Builder function that creates all distributed components
- **DistributedBundle**: Container for all distributed objects
"""

import torch
import deepinv as dinv

# Import physics and loss components
from deepinv.physics import GaussianNoise, LinearPhysics
from deepinv.physics.blur import Blur, gaussian_blur
from deepinv.utils.demo import load_example
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.loss.metric import PSNR

# Import the new factory API
from deepinv.distrib import (
    DistributedContext,
    FactoryConfig,
    TilingConfig,
    make_distrib_bundle,
)


def create_simple_physics_and_measurements(device: torch.device, img_size: tuple = (256, 256)):
    """
    Create simple physics operators and measurements using example images.
    
    :param torch.device device: Device to create operators on
    :param tuple img_size: Size of the image (H, W)
    
    :returns: Tuple of (physics_list, measurements_list, clean_image)
    :rtype: Tuple[List, List[torch.Tensor], torch.Tensor]
    """
    # Load example image (similar to demo_blur_tour.py)
    dtype = torch.float32
    clean_image = load_example(
        "CBSD_0010.png", grayscale=False, device=device, dtype=dtype, img_size=img_size
    )
    
    print(f"Loaded example image with shape: {clean_image.shape}")
    
    # Create different Gaussian blur kernels (simple and clean)
    kernels = [
        gaussian_blur(sigma=1.0, device=str(device)),           # Small blur
        gaussian_blur(sigma=2.0, device=str(device)),           # Medium blur  
        gaussian_blur(sigma=(1.5, 3.0), angle=30, device=str(device))  # Anisotropic blur
    ]
    
    # Create physics operators and measurements
    physics_list = []
    measurements_list = []
    noise_levels = [0.03, 0.05, 0.04]  # Different noise levels for each operator
    
    for i, kernel in enumerate(kernels):
        # Create blur operator with circular padding
        blur_op = Blur(filter=kernel, padding="circular", device=str(device))
        
        # Set the noise model directly on the blur operator
        blur_op.noise_model = GaussianNoise(sigma=noise_levels[i])

        # Move to device
        blur_op = blur_op.to(device)
        
        physics_list.append(blur_op)
        
        # Generate measurement using forward (which applies blur + noise automatically)
        measurement = blur_op(clean_image)
        measurements_list.append(measurement)
        
        print(f"Created physics operator {i+1}: blur kernel {kernel.shape}, noise σ={noise_levels[i]:.3f}")
    
    return physics_list, measurements_list, clean_image


def run_distributed_pnp_factory_demo():
    """
    Demonstrate the distributed PnP algorithm using the Factory API.
    """
    
    print("=" * 70)
    print("🚀 Distributed Framework Factory API Demo")
    print("=" * 70)
    
    # Configuration
    num_iterations = 10
    lr = 0.05
    denoiser_sigma = 0.02
    patch_size = 128
    receptive_field_radius = 32
    
    # Initialize Distributed Context
    with DistributedContext(sharding="round_robin", seed=42) as ctx:

        print(f"\n📊 Process Information:")
        print(f"  • World size: {ctx.world_size} process(es)")
        print(f"  • Current rank: {ctx.rank}")
        print(f"  • Device: {ctx.device}")
        print(f"  • Distributed: {ctx.is_dist}")

        # Create physics operators and measurements
        physics_list, measurements_list, clean_image = create_simple_physics_and_measurements(
            ctx.device, img_size=(128, 128)
        )
        B, C, H, W = clean_image.shape

        if ctx.rank == 0:
            print(f"\n🔬 Created {len(physics_list)} physics operators")
            print(f"  • Image shape: {clean_image.shape}")

        # ==========================================
        # Factory API Configuration and Usage
        # ==========================================

        if ctx.rank == 0:
            print(f"\n🔧 Setting up Factory API configuration...")

        # Create factory configuration
        factory_config = FactoryConfig(
            physics=physics_list,
            measurements=measurements_list,
            data_fidelity=L2(),
        )

        # Create tiling configuration for the prior
        tiling_config = TilingConfig(
            patch_size=patch_size,
            receptive_field_radius=receptive_field_radius,
            overlap=True,
            strategy="smart_tiling",
        )

        # Create PnP prior
        from deepinv.models import DRUNet
        drunet = DRUNet(pretrained="download").to(ctx.device)
        pnp_prior = PnP(denoiser=drunet)

        if ctx.rank == 0:
            print(f"  ✅ Factory configuration created")
            print(f"  ✅ Tiling configuration: {patch_size}x{patch_size} patches")

        # ==========================================
        # Single Builder Call - This is the key benefit!
        # ==========================================

        if ctx.rank == 0:
            print(f"\n🏗️ Building all distributed components with single call...")

        # This single call creates ALL distributed components!
        distributed_bundle = make_distrib_bundle(
            ctx,
            factory_config=factory_config,
            signal_shape=(B, C, H, W),
            prior=pnp_prior,
            tiling=tiling_config,
        )
        
        # Extract components from bundle
        distributed_physics = distributed_bundle.physics
        distributed_measurements = distributed_bundle.measurements
        distributed_data_fidelity = distributed_bundle.data_fidelity
        distributed_signal = distributed_bundle.signal
        distributed_prior = distributed_bundle.prior
        
        if ctx.rank == 0:
            print(f"  ✅ Distributed physics: {len(distributed_physics.local_idx)} local operators")
            print(f"  ✅ Distributed measurements: {len(distributed_measurements.local)} local measurements")
            print(f"  ✅ Distributed signal: shape {distributed_signal.shape}")
            print(f"  ✅ Distributed data fidelity: ready")
            print(f"  ✅ Distributed prior: ready")
        
        # ==========================================
        # Run Distributed PnP Algorithm
        # ==========================================
        
        if ctx.rank == 0:
            print(f"\n🔄 Running distributed PnP algorithm ({num_iterations} iterations)...")
        
        psnr_metric = PSNR()
        psnr_history = []
        
        # Initialize signal with zeros
        distributed_signal.update_(torch.zeros_like(clean_image))
        
        with torch.no_grad():
            for it in range(num_iterations):
                # Data fidelity gradient step (distributed across processes)
                grad = distributed_data_fidelity.grad(distributed_signal)
                
                # Gradient step
                new_data = distributed_signal.data - lr * grad
                distributed_signal.update_(new_data)
                
                # Denoising step (if prior is available)
                if distributed_prior is not None:
                    denoised = distributed_prior.prox(
                        distributed_signal, sigma_denoiser=denoiser_sigma
                    )
                    distributed_signal.update_(denoised)
                
                # Compute PSNR (only on rank 0)
                if ctx.rank == 0:
                    psnr_val = psnr_metric(
                        distributed_signal.data, clean_image
                    ).item()
                    psnr_history.append(psnr_val)
                    
                    if it == 0 or (it + 1) % 2 == 0:
                        print(f"  Iteration {it+1}/{num_iterations}, PSNR: {psnr_val:.2f} dB")
        
        # ==========================================
        # Results Summary  
        # ==========================================
        
        if ctx.rank == 0:
            print(f"\n📊 Results Summary:")
            print(f"  • Final PSNR: {psnr_history[-1]:.2f} dB")
            print(f"  • Reconstruction completed successfully!")
            
            print(f"\n🎯 Factory API Benefits Demonstrated:")
            print(f"  • ✅ Single builder call instead of manual factory functions")
            print(f"  • ✅ Configuration-driven approach")  
            print(f"  • ✅ Type-safe configuration objects")
            print(f"  • ✅ Reduced boilerplate code")
            print(f"  • ✅ Easy to modify and reuse configurations")
            
            print(f"\n✅ Factory API demo completed successfully!")
            
            # Show comparison with manual approach
            print(f"\n📝 Comparison with Manual Approach:")
            print(f"  Manual: ~20+ lines of factory function definitions")
            print(f"  Factory API: ~5 lines of configuration + 1 builder call")
            print(f"  Code reduction: ~75% fewer lines!")


if __name__ == "__main__":
    run_distributed_pnp_factory_demo()
    
    print(f"\n" + "="*70)
    print(f"🎉 Factory API Demo Complete!")
    print(f"="*70)
    print(f"""
Key Takeaways:

1. **FactoryConfig**: Define physics, measurements, and data fidelity
2. **TilingConfig**: Configure spatial processing strategies  
3. **make_distrib_bundle**: Single call to create all distributed objects
4. **DistributedBundle**: Convenient container for all components

Next steps:
- Try your own physics operators and priors
- Experiment with different tiling configurations
- Scale to multiple processes with torchrun
""")