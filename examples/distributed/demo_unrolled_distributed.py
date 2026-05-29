r"""
Distributed Training of an Unfolded DRS Algorithm on Urban100
-------------------------------------------------------------

In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU.
The `deepinv.distributed` framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.

This example shows how to combine:

- the distributed framework (image/model parallelism over large images),
- unfolded optimization with :class:`deepinv.optim.DRS`,
- standard training with :class:`deepinv.Trainer`.

Each GPU (rank) processes different parts/operators of the same image. This is not
standard data-parallel training over different images.

Usage

.. code-block:: bash

    # Single process
    python examples/distributed/demo_unrolled_distributed.py

.. code-block:: bash

    # Multi-process (2 ranks)
    python -m torch.distributed.run --nproc_per_node=2 examples/distributed/demo_unrolled_distributed.py
"""

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import deepinv as dinv
from deepinv.datasets import HDF5Dataset, generate_dataset
from deepinv.distributed import DistributedContext, distribute
from deepinv.loss.metric import PSNR
from deepinv.models import DRUNet
from deepinv.optim import DRS
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.physics import Denoising, GaussianNoise, stack
from deepinv.utils import get_data_home
from deepinv.utils.plotting import plot, plot_curves
from deepinv.utils.tensorlist import TensorList

# %%
# Helper functions
# -----------------------------------------------------------------------------
# These functions assist with data collation, device management, and dataset preparation.


def collate_batch(batch):
    """Collate clean/measured pairs while preserving TensorList measurements."""
    if len(batch) == 1:
        x, y = batch[0]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if isinstance(y, TensorList):
            y = TensorList([m.unsqueeze(0) if m.ndim == 3 else m for m in y])
        elif isinstance(y, (list, tuple)):
            y = [m.unsqueeze(0) if m.ndim == 3 else m for m in y]
        elif y.ndim == 3:
            y = y.unsqueeze(0)
        return x, y

    xs = [x for x, _ in batch]
    ys = [y for _, y in batch]
    x_batch = torch.stack(xs, dim=0)

    if isinstance(ys[0], TensorList):
        n_ops = len(ys[0])
        return x_batch, TensorList(
            [torch.stack([yy[i] for yy in ys], dim=0) for i in range(n_ops)]
        )
    if isinstance(ys[0], (list, tuple)):
        n_ops = len(ys[0])
        return x_batch, [torch.stack([yy[i] for yy in ys], dim=0) for i in range(n_ops)]
    return x_batch, torch.stack(ys, dim=0)


def prepare_dataset(
    ctx: DistributedContext,
    *,
    seed: int,
    crop_size: int,
    train_images: int,
    val_images: int,
    batch_size: int,
    num_workers: int,
    dataset_name: str,
):
    """Create/load Urban100 measurements for training and validation.

    Important: all ranks iterate over the same batches since distribution is over
    image content/operators, not over different images.
    """
    noise_levels = (0.06, 0.08)
    physics_list = []
    for i, sigma in enumerate(noise_levels):
        torch.manual_seed(seed + i)
        physics_list.append(Denoising(noise_model=GaussianNoise(sigma=sigma)))
    stacked_physics = stack(*physics_list)

    data_root = get_data_home() / "Urban100"
    os.makedirs(data_root, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )
    base_dataset = dinv.datasets.Urban100HR(
        root=str(data_root), download=True, transform=transform
    )

    max_images = min(len(base_dataset), train_images + val_images)
    train_base = Subset(base_dataset, list(range(train_images)))
    val_base = Subset(base_dataset, list(range(train_images, max_images)))

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

    train_ds = HDF5Dataset(
        path=str(data_root / f"{dataset_name}_train0.h5"), train=True
    )
    val_ds = HDF5Dataset(path=str(data_root / f"{dataset_name}_val0.h5"), train=True)
    train_generator = torch.Generator().manual_seed(seed + 123)
    val_generator = torch.Generator().manual_seed(seed + 456)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        generator=val_generator,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )

    return stacked_physics, train_loader, val_loader


# %%
# Configuration
# -----------------------------------------------------------------------------
# Settings for training and distributed processing.
# patch_size and overlap control the size of the image patches that each rank processes, and how much they overlap with each other.

seed = 0
n_unroll = 3  # Number of unrolled iterations (DRS steps).
crop_size = 128 if torch.cuda.is_available() else 64

# Training and dataloader settings
epochs = 2 if torch.cuda.is_available() else 1
batch_size = 1
train_images = 16 if torch.cuda.is_available() else 6
val_images = 6 if torch.cuda.is_available() else 4
learning_rate = 2e-4
num_workers = 4 if torch.cuda.is_available() else 0

# Distributed processing settings
patch_size = crop_size // 2
overlap = max(8, patch_size // 8)

torch.manual_seed(seed)


# %%
# Build distributed physics/model and train with deepinv.Trainer
# -----------------------------------------------------------------------------
# The distributed framework allows to distribute unfolded network with a few simple steps:
# - Initialize the distributed context
# - Prepare the physics, model, trainer and dataloaders
# - call :func:`deepinv.distributed.distribute` to distribute the physics and model across ranks
# - Train with :class:`deepinv.Trainer` as usual.
# The framework takes care of synchronizing the forward/backward passes across ranks, and communicating the necessary information between them.


# Keep identical random streams across ranks: this framework splits each image
# across devices, so all ranks should consume the same minibatches.
with DistributedContext(seed=seed, seed_offset=False) as ctx:
    if ctx.rank == 0:
        print("=" * 78)
        print("Distributed Unfolded DRS Demo (Urban100)")
        print("=" * 78)
        print(f"Processes: {ctx.world_size}")
        print(f"Device: {ctx.device}")

    stacked_physics, train_loader, val_loader = prepare_dataset(
        ctx,
        seed=seed,
        crop_size=crop_size,
        train_images=train_images,
        val_images=val_images,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_name="urban100_drs_unfolded",
    )

    # Distribute the stacked physics across ranks.
    distributed_physics = distribute(
        stacked_physics,
        ctx,
    )

    # Build an unfolded DRS model and distribute trainable components.
    denoiser = DRUNet(pretrained="download").to(ctx.device)
    prior = PnP(denoiser=denoiser)
    model = DRS(
        stepsize=[0.9] * n_unroll,
        sigma_denoiser=[0.04] * n_unroll,
        beta=[1.0] * n_unroll,
        trainable_params=["stepsize", "sigma_denoiser", "beta"],
        data_fidelity=L2(),
        prior=prior,
        max_iter=n_unroll,
        unfold=True,
    )
    model = distribute(
        model,
        ctx,
        patch_size=patch_size,
        overlap=overlap,
        max_batch_size=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    psnr_metric = PSNR(reduction="mean")

    # Reconstruction before training.
    demo_x, demo_y = next(iter(val_loader))
    demo_x = demo_x.to(ctx.device)
    demo_y = demo_y.to(ctx.device)
    with torch.no_grad():
        demo_rec_before = model(demo_y, distributed_physics)

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
        save_path=f"ckpts/distributed_unfolded_drs_rank{ctx.rank}",
        verbose=(ctx.rank == 0),
        show_progress_bar=(ctx.rank == 0),
        freq_update_progress_bar=5,
        check_grad=True,
        non_blocking_transfers=False,
    )
    trainer.train()

    with torch.no_grad():
        demo_rec_after = model(demo_y, distributed_physics)

    # %%
    # Display training summary and qualitative result (rank 0 only)
    # -------------------------------------------------------------------------

    if ctx.rank == 0:
        train_history = trainer.train_metrics_history.get("PSNR", [])
        val_history = trainer.eval_metrics_history.get("PSNR", [])

        final_steps = [f"{p.item():.4f}" for p in model.params_algo["stepsize"]]
        print(f"Final trainable stepsizes: {final_steps}")
        if val_history:
            print(f"Final val PSNR: {val_history[-1]:.2f} dB")

        plot(
            [demo_x, demo_y[0], demo_rec_before, demo_rec_after],
            titles=[
                "Ground truth",
                "One noisy measurement",
                "Before training",
                "After training",
            ],
            save_fn="distributed_unrolled_result.png",
        )
        if train_history and val_history:
            plot_curves({"train_psnr": [train_history], "val_psnr": [val_history]})

        print("Saved: distributed_unrolled_result.png")
        print("=" * 78)
