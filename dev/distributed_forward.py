#!/usr/bin/env python3
"""
Distributed Radio Interferometry Inverse Problem Solver

Usage:
    torchrun --nproc_per_node=<NUM_GPUS> distributed_forward.py

This script demonstrates efficient distributed computation for radio interferometry
inverse problems where multiple physics operators share the same reconstruction parameter x.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import deepinv as dinv
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path

from deepinv.physics.radio import RadioInterferometry


class RadioInterferometryDataset(Dataset):
    """Dataset that holds multiple RadioInterferometry operators and their corresponding measurements."""

    def __init__(self, physics_list, y_list):
        """
        Args:
            physics_list: List of RadioInterferometry operators
            y_list: List of corresponding measurements
        """
        assert len(physics_list) == len(
            y_list
        ), "Number of operators must match number of measurements"
        self.physics_list = physics_list
        self.y_list = y_list

    def __len__(self):
        return len(self.physics_list)

    def __getitem__(self, idx):
        return self.physics_list[idx], self.y_list[idx]


class DistributedRadioModel(nn.Module):
    """
    Distributed model for radio interferometry inverse problems.

    - x: Shared reconstruction parameter (synchronized across GPUs)
    - Local physics operators and measurements (not synchronized)
    """

    def __init__(self, img_shape, device):
        super().__init__()
        # Shared parameter x that will be synchronized by DDP
        # Add batch and channel dimensions: (1, 1, H, W)
        full_shape = (1, 1) + img_shape
        self.x = nn.Parameter(
            torch.zeros(full_shape, device=device, dtype=torch.float32)
        )

        # Local storage for physics operators and measurements (not parameters)
        self.local_physics = []
        self.local_measurements = []

    def add_local_data(self, physics_list, y_list):
        """Add physics operators and measurements to this GPU (not synchronized)."""
        self.local_physics.extend(physics_list)
        self.local_measurements.extend(y_list)

    def forward(self):
        """
        Compute local loss: sum_i ||A_i(x) - y_i||^2
        DDP will automatically sum gradients across all GPUs.
        """
        if len(self.local_physics) == 0:
            return torch.tensor(0.0, device=self.x.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.x.device)

        for physics, y in zip(self.local_physics, self.local_measurements):
            # Forward pass: A(x)
            Ax = physics.A(self.x)

            # Compute residual and loss
            residual = Ax - y
            loss = 0.5 * torch.sum(torch.abs(residual) ** 2)
            total_loss = total_loss + loss

        # Average over local operators for numerical stability
        return total_loss / len(self.local_physics)


def init_distributed():
    """Initialize distributed training environment."""
    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            # Choose backend based on availability
            if torch.cuda.is_available() and torch.distributed.is_nccl_available():
                backend = "nccl"
            else:
                backend = "gloo"  # Fallback for CPU or systems without NCCL

            print(f"Initializing distributed training with backend: {backend}")
            dist.init_process_group(backend=backend)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        # Set device based on availability
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        # Single GPU/CPU mode
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    return rank, world_size, local_rank, device


def create_physics_operators(uv_coordinates_list, img_shape, weights_list, device):
    """
    Create multiple RadioInterferometry operators with different sampling patterns.

    Args:
        uv_coordinates_list: List of UV coordinate arrays
        img_shape: Image shape for reconstruction
        weights_list: List of weighting arrays
        device: Device to create operators on

    Returns:
        List of RadioInterferometry operators
    """
    physics_list = []

    for uv_coords, weights in zip(uv_coordinates_list, weights_list):
        uv_tensor = torch.from_numpy(uv_coords).to(device)
        weights_tensor = torch.from_numpy(weights).to(device)

        physics = RadioInterferometry(
            img_size=img_shape,
            samples_loc=uv_tensor.permute(1, 0),
            real_projection=True,
            device=device,
        )

        # Set weights
        physics.setWeight(weights_tensor)

        physics_list.append(physics)

    return physics_list


def load_real_data_unified(data_dir, device, rank, world_size, num_operators=None):
    """
    Load real radio interferometry data from unified files and split into multiple operators.

    This function loads the complete dataset from the demo files and then creates multiple
    physics operators by splitting the UV coordinates and measurements.

    Expected file structure:
    data_dir/
        ├── uv_coordinates.npy  # Full UV coordinates (2, N) or (N, 2)
        ├── 3c353_gdth.npy      # Ground truth image
        └── briggs_weight.npy   # Briggs weighting

    Args:
        data_dir: Directory containing data files
        device: Device to load data to
        rank: Current process rank
        world_size: Total number of processes
        num_operators: Number of physics operators to create (default: world_size * 4)

    Returns:
        physics_list, y_list, x_true
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    # Required files
    uv_file = data_path / "uv_coordinates.npy"
    gt_file = data_path / "3c353_gdth.npy"
    weights_file = data_path / "briggs_weight.npy"

    if not uv_file.exists():
        raise FileNotFoundError(f"UV coordinates file not found: {uv_file}")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    if rank == 0:
        print(f"Loading real radio interferometry data from {data_dir}")

    # Load ground truth image
    image_gdth = np.load(gt_file, allow_pickle=True)
    x_true = torch.from_numpy(image_gdth).unsqueeze(0).unsqueeze(0).to(device)
    img_shape = x_true.shape[-2:]

    # Load UV coordinates
    uv = np.load(uv_file, allow_pickle=True)
    uv = torch.from_numpy(uv).to(device).float()  # Ensure float32
    # UV coordinates need to be transposed to (2, N) for RadioInterferometry
    if uv.shape[1] == 2:
        uv = uv.transpose(0, 1)  # Convert from (N, 2) to (2, N)

    # Load Briggs weights
    briggs_weight = np.load(weights_file, allow_pickle=True)
    briggs_weight = (
        torch.from_numpy(briggs_weight).to(device).float().squeeze()
    )  # Remove singleton dimensions

    # Noise parameters from the demo
    tau = 0.5976 * 2e-3

    # Create the full physics operator to generate measurements
    if rank == 0:
        print(f"Creating full physics operator with {uv.shape[1]} measurements")

    full_physics = RadioInterferometry(
        img_size=img_shape,
        samples_loc=uv,
        real_projection=True,
        device=device,
    )

    # Generate measurements with noise (following the demo)
    torch.manual_seed(42)  # For reproducibility
    y_full = full_physics.A(x_true)
    noise = (torch.randn_like(y_full) + 1j * torch.randn_like(y_full)) / np.sqrt(2)
    y_full = y_full + tau * noise.to(y_full.dtype)  # Ensure same dtype

    # Apply weighting (following the demo)
    y_full *= briggs_weight / tau
    weights_full = (briggs_weight / tau).to(y_full.dtype)  # Ensure same dtype

    # Determine number of operators to create
    if num_operators is None:
        num_operators = world_size

    total_measurements = uv.shape[1]  # UV coords are (2, N), so N is uv.shape[1]
    measurements_per_operator = total_measurements // num_operators

    if rank == 0:
        print(
            f"Splitting {total_measurements} measurements into {num_operators} operators"
        )
        print(f"Approximately {measurements_per_operator} measurements per operator")

    physics_list = []
    y_list = []

    # Split the data into multiple operators
    for i in range(num_operators):
        start_idx = i * measurements_per_operator
        if i == num_operators - 1:
            # Last operator gets remaining measurements
            end_idx = total_measurements
        else:
            end_idx = (i + 1) * measurements_per_operator

        # Extract subset of UV coordinates and measurements
        uv_subset = uv[:, start_idx:end_idx]  # UV coords are (2, N)
        y_subset = y_full[
            :, :, start_idx:end_idx
        ]  # Keep batch+channel dimension, subset measurements
        weights_subset = weights_full[start_idx:end_idx]

        # Create physics operator for this subset
        physics = RadioInterferometry(
            img_size=img_shape,
            samples_loc=uv_subset,
            real_projection=True,
            device=device,
        )

        # Set weights for this subset
        physics.setWeight(weights_subset)

        physics_list.append(physics)
        y_list.append(y_subset)

        if rank == 0 and i < 3:  # Print info for first few operators
            print(f"  Operator {i+1}: {uv_subset.shape[1]} measurements")

    if rank == 0:
        print(f"Created {len(physics_list)} physics operators")
        print(f"Ground truth image shape: {x_true.shape}")

    return physics_list, y_list, x_true


def load_real_data(data_dir, device, rank):
    """
    Load real radio interferometry data from files.

    Expected file structure:
    data_dir/
        ├── uv_coordinates_0.npy, uv_coordinates_1.npy, ...
        ├── measurements_0.npy, measurements_1.npy, ...
        ├── weights_0.npy, weights_1.npy, ...
        └── ground_truth.npy (optional)

    Args:
        data_dir: Directory containing data files
        device: Device to load data to
        rank: Current process rank

    Returns:
        physics_list, y_list, x_true (if available)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    # Find all UV coordinate files
    uv_files = sorted(data_path.glob("uv_coordinates_*.npy"))
    measurement_files = sorted(data_path.glob("measurements_*.npy"))
    weight_files = sorted(data_path.glob("weights_*.npy"))

    if len(uv_files) == 0:
        raise FileNotFoundError(f"No UV coordinate files found in {data_dir}")

    if len(uv_files) != len(measurement_files):
        raise ValueError(
            "Number of UV coordinate files must match number of measurement files"
        )

    # Load ground truth if available
    gt_file = data_path / "ground_truth.npy"
    if gt_file.exists():
        x_true = torch.from_numpy(np.load(gt_file)).to(device)
    else:
        x_true = None

    physics_list = []
    y_list = []

    for i, (uv_file, meas_file) in enumerate(zip(uv_files, measurement_files)):
        if rank == 0:
            print(f"Loading operator {i+1}/{len(uv_files)}: {uv_file.name}")

        # Load UV coordinates
        uv_coords = np.load(uv_file)
        uv_tensor = torch.from_numpy(uv_coords).to(device)

        # Load measurements
        measurements = np.load(meas_file)
        y = torch.from_numpy(measurements).to(device)

        # Load weights if available
        if i < len(weight_files):
            weights = np.load(weight_files[i])
            weights_tensor = torch.from_numpy(weights).to(device)
        else:
            weights_tensor = torch.ones(len(measurements), device=device)

        # Get image shape from first measurement or use default
        if x_true is not None:
            img_shape = x_true.shape[-2:]
        else:
            img_shape = (256, 256)  # Default, should be specified in practice

        # Create physics operator
        physics = RadioInterferometry(
            img_size=img_shape,
            samples_loc=(
                uv_tensor.permute(1, 0) if uv_tensor.shape[0] == 2 else uv_tensor.T
            ),
            real_projection=True,
            device=device,
        )

        # Set weights
        physics.setWeight(weights_tensor)

        physics_list.append(physics)
        y_list.append(y)

    return physics_list, y_list, x_true


def generate_synthetic_data(img_shape, num_operators, device, rank, world_size):
    """
    Generate synthetic radio interferometry data for testing.
    In practice, you would load real UV coordinates and measurements.
    """
    torch.manual_seed(42 + rank)  # Different seed per rank for diversity

    uv_coordinates_list = []
    weights_list = []
    physics_list = []
    y_list = []

    # Create a ground truth image (only for generating synthetic data)
    if rank == 0:
        # Simple test image with point sources
        x_true = torch.zeros(
            (1, 1) + img_shape, device=device
        )  # Add batch and channel dims
        x_true[0, 0, img_shape[0] // 4, img_shape[1] // 4] = 1.0
        x_true[0, 0, 3 * img_shape[0] // 4, 3 * img_shape[1] // 4] = 0.8
        x_true[0, 0, img_shape[0] // 2, img_shape[1] // 2] = 0.5
    else:
        x_true = torch.zeros((1, 1) + img_shape, device=device)

    # Broadcast ground truth to all ranks for consistent data generation (only in distributed mode)
    if world_size > 1:
        dist.broadcast(x_true, src=0)

    for i in range(num_operators):
        # Generate random UV coordinates
        num_visibilities = 1000 + i * 100  # Varying number of measurements
        uv_coords = np.random.uniform(
            -np.pi, np.pi, (2, num_visibilities)
        )  # Shape: (2, N)
        weights = np.ones(num_visibilities)

        uv_coordinates_list.append(uv_coords)
        weights_list.append(weights)

        # Create physics operator
        uv_tensor = torch.from_numpy(uv_coords).to(device).float()  # Ensure float32
        physics = RadioInterferometry(
            img_size=img_shape,
            samples_loc=uv_tensor,  # Already in correct shape (2, N)
            real_projection=True,
            device=device,
        )

        # Generate measurement with noise
        with torch.no_grad():
            y_clean = physics.A(x_true)
            noise = (
                torch.randn_like(y_clean) + 1j * torch.randn_like(y_clean)
            ) / np.sqrt(2)
            y = y_clean + 0.01 * noise.to(y_clean.dtype)  # Ensure same dtype

        physics_list.append(physics)
        y_list.append(y)

    return physics_list, y_list, x_true


def distribute_data(physics_list, y_list, rank, world_size):
    """Distribute physics operators and measurements across GPUs."""
    total_operators = len(physics_list)

    # Simple round-robin distribution
    local_physics = []
    local_y = []

    for i in range(rank, total_operators, world_size):
        local_physics.append(physics_list[i])
        local_y.append(y_list[i])

    return local_physics, local_y


def main():
    # Initialize distributed environment
    rank, world_size, local_rank, device = init_distributed()

    # Problem configuration
    img_shape = (256, 256)  # Image size
    num_operators = 4  # Total number of physics operators (for synthetic data)
    use_real_data = True  # Set to True to load real data
    use_unified_data = (
        True  # Set to True to use unified data loading (load_real_data_unified)
    )
    data_dir = "./"  # Directory containing real data files

    if rank == 0:
        print(f"Running distributed radio interferometry with {world_size} GPUs")
        if use_real_data:
            if use_unified_data:
                print(f"Loading unified real data from: {data_dir}")
                print(
                    "  Expected files: uv_coordinates.npy, 3c353_gdth.npy, briggs_weight.npy"
                )
            else:
                print(f"Loading real data from: {data_dir}")
        else:
            print(f"Generating synthetic data with {num_operators} operators")

    # Load or generate data
    if use_real_data:
        try:
            if use_unified_data:
                physics_list, y_list, x_true = load_real_data_unified(
                    data_dir, device, rank, world_size, num_operators
                )
                # Update img_shape from loaded data
                img_shape = x_true.shape[-2:]
            else:
                physics_list, y_list, x_true = load_real_data(data_dir, device, rank)
                # Update img_shape from loaded data
                if x_true is not None:
                    img_shape = x_true.shape[-2:]
        except (FileNotFoundError, ValueError) as e:
            if rank == 0:
                print(f"Error loading real data: {e}")
                print("Falling back to synthetic data generation...")
            use_real_data = False

    if not use_real_data:
        # Generate synthetic data
        physics_list, y_list, x_true = generate_synthetic_data(
            img_shape, num_operators, device, rank, world_size
        )

    if rank == 0:
        print(f"Image shape: {img_shape}")
        print(f"Total physics operators: {len(physics_list)}")

    # Distribute data across GPUs
    local_physics, local_y = distribute_data(physics_list, y_list, rank, world_size)

    if rank == 0:
        print(f"Rank {rank}: {len(local_physics)} local operators")

    # Create distributed model
    model = DistributedRadioModel(img_shape, device)
    model.add_local_data(local_physics, local_y)

    # Wrap with DDP only in distributed mode
    if world_size > 1:
        if torch.cuda.is_available():
            # GPU mode: specify device_ids and output_device
            ddp_model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,  # Don't sync physics operators/measurements
                find_unused_parameters=False,
                static_graph=True,
            )
        else:
            # CPU mode: don't specify device_ids
            ddp_model = DDP(
                model,
                broadcast_buffers=False,  # Don't sync physics operators/measurements
                find_unused_parameters=False,
                static_graph=True,
            )
    else:
        # Single GPU/CPU mode - no DDP needed
        ddp_model = model

    # Initialize x with a simple guess (e.g., zeros)
    with torch.no_grad():
        # Skip backprojection initialization for now due to weight size issues
        # Just initialize with zeros
        if world_size > 1:
            ddp_model.module.x.data.zero_()
        else:
            ddp_model.x.data.zero_()

    # Ensure x is synchronized across all ranks (only in distributed mode)
    if world_size > 1:
        dist.all_reduce(ddp_model.module.x.data, op=dist.ReduceOp.SUM)
        ddp_model.module.x.data /= world_size

    # Optimizer (only optimizes x, the shared parameter)
    if world_size > 1:
        optimizer = torch.optim.SGD(
            [ddp_model.module.x], lr=1e-3
        )  # Lower LR for real data
    else:
        optimizer = torch.optim.SGD([ddp_model.x], lr=1e-3)  # Lower LR for real data

    # Training loop
    num_iterations = 50  # Quick test

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass: compute local loss
        loss = ddp_model()

        # Backward pass: DDP automatically reduces gradients
        loss.backward()

        # Gradient clipping for numerical stability
        if world_size > 1:
            torch.nn.utils.clip_grad_norm_(ddp_model.module.x, max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(ddp_model.x, max_norm=1.0)

        # Optimization step
        optimizer.step()

        # Logging
        if rank == 0 and iteration % 10 == 0:
            with torch.no_grad():
                # Compute total loss across all GPUs for monitoring
                total_loss = loss.item() * world_size  # Approximate

                # Compute PSNR if ground truth is available
                if x_true is not None:
                    if world_size > 1:
                        x_current = ddp_model.module.x.detach()
                    else:
                        x_current = ddp_model.x.detach()
                    mse = torch.mean((x_current - x_true) ** 2)
                    psnr = (
                        20 * torch.log10(1.0 / torch.sqrt(mse))
                        if mse > 0
                        else float("inf")
                    )
                    print(
                        f"Iteration {iteration:3d} | Loss: {total_loss:.6f} | PSNR: {psnr:.2f} dB"
                    )
                else:
                    print(f"Iteration {iteration:3d} | Loss: {total_loss:.6f}")

    # Save final result
    if rank == 0:
        if world_size > 1:
            x_final = ddp_model.module.x.detach().cpu()
        else:
            x_final = ddp_model.x.detach().cpu()
        x_true_cpu = x_true.cpu()

        # Compute final metrics
        mse = torch.mean((x_final - x_true_cpu) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else float("inf")

        print(f"Final Results:")
        print(f"Final PSNR: {psnr:.2f} dB")
        print(f"Final MSE: {mse:.6f}")

        # Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        torch.save(x_final, output_dir / "x_reconstructed.pt")
        torch.save(x_true_cpu, output_dir / "x_ground_truth.pt")

        print(f"Results saved to {output_dir}/")

    # Cleanup
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
