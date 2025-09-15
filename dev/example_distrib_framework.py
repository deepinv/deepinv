import torch
from deepinv.physics import GaussianNoise
from deepinv.physics.blur import Blur
from deepinv.physics.inpainting import Inpainting
from deepinv.optim.data_fidelity import L2


from deepinv.physics.forward import StackedLinearPhysics


from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedDataFidelity,
    DistributedMeasurements,
    DistributedSignal,
)



def create_test_problem():
    """Create a test reconstruction problem."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (1, 1, 64, 64)
    
    # Create ground truth signal
    true_signal = torch.zeros(img_size, device=device)
    # Add some geometric shapes
    true_signal[0, 0, 20:44, 20:44] = 1.0  # Large square
    true_signal[0, 0, 25:39, 25:39] = 0.3  # Inner square
    true_signal[0, 0, 30:34, 30:34] = 0.8  # Innermost square
    
    # Create physics operators
    physics_list = []
    measurements_list = []
    
    # Physics 1: Gaussian blur
    blur_kernel = torch.ones((1, 1, 5, 5), device=device) / 25.0
    physics1 = Blur(filter=blur_kernel, device=device)
    physics1.noise_model = GaussianNoise(sigma=0.02)
    physics_list.append(physics1)
    measurements_list.append(physics1(true_signal))
    
    # Physics 2: Inpainting with structured missing data
    torch.manual_seed(42)
    # Create mask with spatial dimensions only (C, H, W)
    mask_shape = img_size[1:]  # Remove batch dimension: (1, 64, 64)
    mask = torch.ones(mask_shape, device=device, dtype=torch.bool)
    # Create stripes of missing data
    mask[0, ::4, :] = False  # Every 4th row
    mask[0, :, ::6] = False  # Every 6th column
    physics2 = Inpainting(img_size=mask_shape, mask=mask, device=device)
    physics2.noise_model = GaussianNoise(sigma=0.01)
    physics_list.append(physics2)
    measurements_list.append(physics2(true_signal))
    
    # Physics 3: Different blur (edge detection)
    edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                              device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    physics3 = Blur(filter=edge_kernel, device=device)
    physics3.noise_model = GaussianNoise(sigma=0.05)
    physics_list.append(physics3)
    measurements_list.append(physics3(true_signal))
    
    return true_signal, physics_list, measurements_list

true_signal, physics_list, measurements_list = create_test_problem()

def factory_physics(idx, device, shared):
    return physics_list[idx].to(device)

def factory_data_fidelity(idx, device, shared):
    # return the DF object for index i (e.g., L2, Poisson…)
    return L2().to(device)

def read_measurement(idx, device, shared):
    # load or generate measurement for index i directly on device
    return measurements_list[idx].to(device)

num_forward_models = len(measurements_list)  # total number of forward models / measurements
lr = 0.001           # learning rate (reduced from 0.1)
T = 10            # number of iterations

with DistributedContext(sharding="round_robin") as ctx:
    # Build physics (local shards only)
    physics = DistributedLinearPhysics(
        ctx, num_ops=num_forward_models, factory=factory_physics,
    )

    # Build shared measurements (local shards only)
    measurements = DistributedMeasurements(
        ctx, num_items=num_forward_models, factory=read_measurement
    )

    # Build replicated signal - extract shape from true_signal
    B, C, H, W = true_signal.shape
    signal = DistributedSignal(ctx, shape=(B, C, H, W))
    # Initialize with noisy version of true signal for testing
    signal.update_(true_signal.clone() + 0.1 * torch.randn_like(true_signal))

    # Data fidelity
    df = DistributedDataFidelity(
        ctx, physics, measurements, data_fidelity_factory=factory_data_fidelity, reduction="sum"
    )

    # Simple gradient descent
    for it in range(T):
        loss = df.fn(signal)
        g = df.grad(signal)
        signal.data = signal.data - lr * g
        if ctx.rank == 0:
            print(f"Iteration {it}, Loss: {loss.item():.6f}, Grad norm: {g.norm().item():.6f}")
    
    if ctx.rank == 0:
        print("Gradient descent completed!")
