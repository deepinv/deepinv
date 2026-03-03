r"""
Distributed Training of Unrolled PGD with PnP Prior
----------------------------------------------------

This example demonstrates training an unrolled PGD (Proximal Gradient Descent) algorithm with
Plug-and-Play (PnP) prior in a fully distributed DeepInverse setup:

- distributed physics operators,
- distributed data-fidelity gradient,
- distributed PnP denoiser with tiling,
- trainable algorithmic parameters (stepsize) and denoiser weights
- proper distributed metric reduction across all ranks using weighted averaging.

We use Urban100HR as image dataset, a pretrained DRUNet as denoiser in the PnP prior, and run only
a few epochs for the sake of the demo.

**Usage:**

.. code-block:: bash

    # Single process
    python examples/distributed/demo_unrolled_PGD_distributed.py

.. code-block:: bash

    # Multi-process (2 ranks)
    python -m torch.distributed.run --nproc_per_node=2 examples/distributed/demo_unrolled_PGD_distributed.py

"""

# %%
# Import Libraries
# ----------------
#
# Import necessary modules for distributed training, dataset handling, and optimization.

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import deepinv as dinv
from deepinv.datasets import generate_dataset, HDF5Dataset
from deepinv.models import DRUNet
from deepinv.loss.metric import PSNR
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim import PGD
from deepinv.distributed import DistributedContext, distribute
from deepinv.physics import Denoising, GaussianNoise, stack
from deepinv.utils import get_data_home
from deepinv.utils.plotting import plot, plot_curves
from deepinv.utils.tensorlist import TensorList

# %%
# .. raw:: html
#
#    <details>
#    <summary style="cursor: pointer; color: #0066cc; font-weight: bold;">Click to expand: Helper functions for data handling</summary>
#
# We define several helper functions:
#
# 1. ``collate_deepinv_batch``: Handles batching of TensorList measurements from stacked physics
# 2. ``_move_measurement_to_device``: Moves various measurement types to GPU/CPU


def collate_deepinv_batch(batch):
    """Custom collate function to handle TensorList objects from deepinv.

    This function properly handles batching of ground truths and measurements,
    including special treatment for TensorList objects from stacked physics operators.
    """
    if len(batch) == 1:
        ground_truth, measurement = batch[0]
        # Add batch dimension if needed
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.unsqueeze(0)
        if isinstance(measurement, TensorList):
            measurement = TensorList(
                [m.unsqueeze(0) if m.ndim == 3 else m for m in measurement]
            )
        elif isinstance(measurement, (list, tuple)):
            measurement = [m.unsqueeze(0) if m.ndim == 3 else m for m in measurement]
        elif measurement.ndim == 3:
            measurement = measurement.unsqueeze(0)
        return ground_truth, measurement
    else:
        # For batch_size > 1, stack ground truths and handle measurements
        ground_truths = []
        measurements = []
        for gt, meas in batch:
            ground_truths.append(gt)
            measurements.append(meas)

        # Stack ground truths
        ground_truth_batch = torch.stack(ground_truths, dim=0)

        # Handle measurements - if TensorList, stack each operator's measurements
        if isinstance(measurements[0], TensorList):
            # Stack measurements for each operator separately
            num_operators = len(measurements[0])
            stacked_measurements = []
            for op_idx in range(num_operators):
                op_measurements = [meas[op_idx] for meas in measurements]
                stacked_measurements.append(torch.stack(op_measurements, dim=0))
            measurement_batch = TensorList(stacked_measurements)
        elif isinstance(measurements[0], (list, tuple)):
            # Stack list/tuple measurements
            num_operators = len(measurements[0])
            stacked_measurements = []
            for op_idx in range(num_operators):
                op_measurements = [meas[op_idx] for meas in measurements]
                stacked_measurements.append(torch.stack(op_measurements, dim=0))
            measurement_batch = stacked_measurements
        else:
            # Single tensor measurements
            measurement_batch = torch.stack(measurements, dim=0)
        return ground_truth_batch, measurement_batch


def _move_measurement_to_device(measurement, device):
    """Move measurement to device, handling various tensor types."""
    if hasattr(measurement, "to"):
        return measurement.to(device)
    elif isinstance(measurement, TensorList):
        return TensorList([m.to(device) for m in measurement])
    elif isinstance(measurement, list):
        return [m.to(device) for m in measurement]
    return measurement


# %%
#  Dataset Preparation Function
# ------------------------------
#
# This function encapsulates the full distributed data pipeline: it builds stacked physics operators with
# varying noise levels, downloads and splits Urban100 into train/validation sets, precomputes measurements (
# rank 0 only), and sets up the DataLoaders.


def prepare_dataset(
    noise_levels: tuple[float, ...],
    crop_size: int,
    train_images: int,
    val_images: int,
    batch_size: int,
    num_workers: int,
    ctx: DistributedContext,
    seed: int = 0,
    dataset_name: str = "urban100",
):
    """Prepare training and validation datasets with pre-computed measurements.

    :param tuple noise_levels: Tuple of noise levels for stacked physics operators
    :param int crop_size: Size to crop images
    :param int train_images: Number of training images
    :param int val_images: Number of validation images
    :param int batch_size: Batch size for DataLoaders
    :param int num_workers: Number of worker processes for data loading
    :param DistributedContext ctx: Distributed context
    :param int seed: Random seed for reproducibility
    :param str dataset_name: Simple name for dataset files

    :return: Tuple of (stacked_physics, train_loader, val_loader)
    """
    # Create stacked physics operators with different noise levels
    physics_list = []
    for i, sigma in enumerate(noise_levels):
        rng = torch.Generator(device=ctx.device).manual_seed(seed + i)
        physics_list.append(Denoising(noise_model=GaussianNoise(sigma=sigma, rng=rng)))
    stacked_physics = stack(*physics_list)

    # Setup data transforms and load base dataset
    data_root = get_data_home() / "Urban100"
    transform = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )

    os.makedirs(data_root, exist_ok=True)
    base_dataset = dinv.datasets.Urban100HR(
        root=str(data_root), download=True, transform=transform
    )

    # Split into train and validation
    max_images = min(len(base_dataset), train_images + val_images)
    train_base = Subset(base_dataset, list(range(train_images)))
    val_base = Subset(base_dataset, list(range(train_images, max_images)))

    # Only rank 0 generates datasets to avoid conflicts
    if ctx.rank == 0:
        generate_dataset(
            train_dataset=train_base,
            physics=stacked_physics,
            save_dir=str(data_root),
            dataset_filename=f"{dataset_name}_train",
            device=ctx.device,
            train_datapoints=train_images,
            num_workers=num_workers,
        )
        generate_dataset(
            train_dataset=val_base,
            physics=stacked_physics,
            save_dir=str(data_root),
            dataset_filename=f"{dataset_name}_val",
            device=ctx.device,
            train_datapoints=len(val_base),
            num_workers=num_workers,
        )

    # Synchronize all ranks before loading datasets
    if ctx.use_dist:
        dist.barrier()

    # Load datasets (generate_dataset appends rank suffix)
    train_path = str(data_root / f"{dataset_name}_train0.h5")
    val_path = str(data_root / f"{dataset_name}_val0.h5")

    train_dataset = HDF5Dataset(path=train_path, train=True)
    val_dataset = HDF5Dataset(path=val_path, train=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_deepinv_batch,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_deepinv_batch,
        num_workers=num_workers,
    )

    return stacked_physics, train_loader, val_loader


# %%
# We then define a ``evaluate_psnr`` function to evaluate PSNR on the validation set.


@torch.no_grad()
def evaluate_psnr(
    model,
    denoiser: torch.nn.Module,
    loader: DataLoader,
    physics,
    metric: PSNR,
    ctx: DistributedContext,
) -> float:
    """Evaluate PSNR on validation set."""
    model.eval()
    denoiser.eval()
    psnr_sum = 0.0
    count = 0
    for ground_truth, measurement in loader:
        # Move data to device
        ground_truth = ground_truth.to(ctx.device)
        measurement = _move_measurement_to_device(measurement, ctx.device)

        # Generate reconstruction
        x_hat = model(measurement, physics)

        # Accumulate weighted PSNR (weight = batch size)
        b = ground_truth.shape[0]
        psnr_val = metric(x_hat, ground_truth)
        if psnr_val.ndim > 0:  # Handle batch of PSNRs
            psnr_sum += psnr_val.sum().item()
        else:  # Handle single PSNR value
            psnr_sum += psnr_val.item() * b
        count += b

    return psnr_sum / count if count > 0 else 0.0


# %%
# Configuration
# -------------
#
# Set up training hyperparameters and distributed configuration.
#
# **Key parameters:**
#
# - ``n_unroll``: Number of PGD iterations to unfold (number of layers)
# - ``learning_rate``: Small (1e-5) to handle training 32M+ denoiser parameters alongside algorithmic parameters
# - ``batch_size``: Samples per batch
# - ``patch_size`` and ``overlap``: For distributed tiling of images beyond GPU memory

n_unroll = 4  # number of unfolded PGD iterations
epochs = 2 if torch.cuda.is_available() else 1
crop_size = 128 if torch.cuda.is_available() else 64
batch_size = 1
train_images = 24 if torch.cuda.is_available() else 8
val_images = 8 if torch.cuda.is_available() else 4
learning_rate = 1e-5  # Small learning rate for training 32M+ parameters
sigma_denoiser = 0.05
patch_size = crop_size // 2
overlap = max(4, patch_size // 8)
num_workers = 4 if torch.cuda.is_available() else 0
seed = 0


# %%
# Distributed setup and training
# -------------------------------

with DistributedContext(seed=0) as ctx:
    if ctx.rank == 0:
        print("=" * 78)
        print("Distributed Unrolled PGD Training with PnP Prior")
        print("=" * 78)
        print(f"Processes: {ctx.world_size}")
        print(f"Device: {ctx.device}")
        print(f"Unrolled iterations: {n_unroll}")

    # Step 1: Prepare Dataset
    stacked_physics, train_loader, val_loader = prepare_dataset(
        noise_levels=(0.06, 0.08),
        crop_size=crop_size,
        train_images=train_images,
        val_images=val_images,
        batch_size=batch_size,
        num_workers=num_workers,
        ctx=ctx,
        seed=seed,
        dataset_name="urban100_pgd",
    )

    # Step 2: Distributed physics and distributed data-fidelity
    distributed_physics = distribute(
        stacked_physics,
        ctx,
        type_object="linear_physics",
        reduction="mean",
    )
    distributed_data_fidelity = distribute(L2(), ctx)

    # Step 3: Shared pretrained denoiser + distributed wrapper
    denoiser = DRUNet(pretrained="download").to(ctx.device)
    distributed_denoiser = distribute(
        denoiser,
        ctx,
        type_object="denoiser",
        patch_size=patch_size,
        overlap=overlap,
        max_batch_size=1,
    )

    # Step 4: PGD unfolded trainable model
    # We will train both the stepsize parameters and the denoiser weights end-to-end.
    stepsize = [0.642] * n_unroll  # stepsize of the algorithm
    trainable_params = ["stepsize"]  # define which parameters are trainable

    # Create PnP prior with distributed denoiser
    prior = PnP(denoiser=distributed_denoiser)

    # Create PGD model with unfolding
    model = PGD(
        stepsize=stepsize,
        sigma_denoiser=sigma_denoiser,
        trainable_params=trainable_params,
        data_fidelity=distributed_data_fidelity,
        max_iter=n_unroll,
        prior=prior,
        unfold=True,
    )

    # Step 5: Setup Training
    # Train all parameters: algorithmic parameters (stepsize) + denoiser weights
    # Since the distributed wrapper doesn't expose parameters, we add denoiser.parameters() explicitly
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(denoiser.parameters()), lr=learning_rate
    )
    psnr_metric = PSNR(reduction="mean")

    # Keep one validation image for qualitative comparison
    demo_x, demo_y = next(iter(val_loader))
    demo_x = demo_x.to(ctx.device)
    demo_y = _move_measurement_to_device(demo_y, ctx.device)
    with torch.no_grad():
        demo_rec_before = model(demo_y, distributed_physics).detach()

    # Evaluate initial PSNR
    init_val_psnr = evaluate_psnr(
        model, denoiser, val_loader, distributed_physics, psnr_metric, ctx
    )

    if ctx.rank == 0:
        print(f"Initial validation PSNR: {init_val_psnr:.2f} dB")
        print("Starting training...")

    # Step 6: Training with Trainer
    trainer = dinv.Trainer(
        model=model,
        physics=distributed_physics,
        epochs=epochs,
        device=ctx.device,
        losses=[dinv.loss.SupLoss(metric=dinv.metric.MSE())],
        metrics=psnr_metric,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        grad_clip=1.0,
        compare_no_learning=False,
        save_path=f"ckpts/distributed_pgd_rank{ctx.rank}",
        verbose=(ctx.rank == 0),
        show_progress_bar=(ctx.rank == 0),
        check_grad=True,
    )
    trainer.train()

    train_psnr_history = trainer.train_metrics_history["PSNR"]
    val_psnr_history = [init_val_psnr] + trainer.eval_metrics_history["PSNR"]
    neg_val_psnr_history = [-v for v in val_psnr_history]

    # Step 7: Visualize (rank 0)

    with torch.no_grad():
        demo_rec_after = model(demo_y, distributed_physics).detach()
    if ctx.rank == 0:
        final_steps = [f"{s.item():.4f}" for s in model.params_algo["stepsize"]]
        print(f"Final trainable step sizes: {final_steps}")
        print(
            "Note: we track `-PSNR` as a decreasing objective (equivalent to increasing PSNR)."
        )

        plot(
            [demo_x, demo_y[0], demo_rec_before, demo_rec_after],
            titles=[
                "Ground truth",
                "One noisy measurement",
                f"Before training ({init_val_psnr:.2f} dB)",
                f"After training ({val_psnr_history[-1]:.2f} dB)",
            ],
            figsize=(16, 4),
            save_fn="distributed_pgd_result.png",
        )

        plot_curves(
            {
                "train_psnr": [train_psnr_history],
                "val_psnr": [val_psnr_history],
                "neg_val_psnr": [neg_val_psnr_history],
            }
        )

        print("Saved: distributed_pgd_result.png")
        print("=" * 78)
