# torchrun --nproc_per_node=$NGPUS main.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

# ----- Option A: explicit matrices (dense example) -----
# Each A_i: [m_i, n] ; each y_i: [m_i]
# If you have operators as callables, see Option B below.

class LocalLeastSquares(nn.Module):
    """
    Holds a shared parameter x and a *local* pool of (A_i, y_i) that live on this GPU.
    Only x is a parameter -> only x is synchronized/reduced by DDP.
    A_i, y_i are registered as buffers and will NOT be broadcast because we set broadcast_buffers=False in DDP.
    """
    def __init__(self, n, A_list, y_list, device):
        super().__init__()
        self.x = nn.Parameter(torch.zeros(n, device=device))  # shared across all ranks via DDP
        # Keep local operators/data as buffers (no grad, persistent on device, reused every step)

        # Use buffers so DDP doesn't treat them as params; they won’t be synced.
        # We keep simple Python lists and register each tensor as a buffer.
        self.A_list = nn.ModuleList()  # container to keep them attached to module
        for A, y in zip(A_list, y_list):
            mod = nn.Module()
            mod.register_buffer("A", A.to(device, non_blocking=True))
            mod.register_buffer("y", y.to(device, non_blocking=True))
            self.A_list.append(mod)

    def forward(self):
        # 1/2 * sum_i ||A_i x - y_i||^2   -> grad wrt x is sum_i A_i^T(A_i x - y_i)
        loss = torch.zeros((), device=self.x.device)
        x = self.x
        for mod in self.A_list:
            r = mod.A @ x - mod.y
            loss = loss + 0.5 * (r @ r)
        return loss / len(self.A_list)  # optional scaling

def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    return rank, world_size, device

def shard_operators_globally(A_all, y_all, rank, world_size):
    # Simple static sharding: each rank gets a contiguous slice
    N = len(A_all)
    per = (N + world_size - 1) // world_size
    i0 = rank * per
    i1 = min((rank + 1) * per, N)
    return A_all[i0:i1], y_all[i0:i1]

def main():
    rank, world_size, device = init_distributed()

    # ---- Prepare your operators/data once (e.g., loaded from disk on rank 0) ----
    # In practice, read from files; here we mock them on every rank for brevity,
    # then each rank keeps only its shard on *its* GPU.
    torch.manual_seed(0)
    n = 1024
    num_ops = 2000
    A_all = [torch.randn(256, n, dtype=torch.float32) for _ in range(num_ops)]
    y_all = [torch.randn(256, dtype=torch.float32) for _ in range(num_ops)]

    # Optional: build these only on rank 0 and broadcast metadata;
    # for huge datasets you'd load from shared storage per rank to avoid extra copies.
    A_local, y_local = shard_operators_globally(A_all, y_all, rank, world_size)

    # ---- Build model and DDP ----
    model = LocalLeastSquares(n, A_local, y_local, device).to(device)
    # Very important: broadcast_buffers=False -> we don't sync A,y (kept local, one copy per GPU)
    ddp = DDP(
        model,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,  # if your graph is static; optional but speeds up
    )

    opt = torch.optim.SGD([ddp.module.x], lr=1e-2)  # only x is a parameter

    # ---- Training loop (same x across all operators; only x is broadcast/reduced) ----
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        loss = ddp.forward()     # each rank computes sum over *its* A_i,y_i
        loss.backward()          # DDP all-reduces grad(x) -> global sum over all ranks
        opt.step()               # update x (identical on every rank)
        if rank == 0 and step % 10 == 0:
            print(f"step {step:03d} loss {loss.item():.4f}")

    # After training, ddp.module.x is the optimized x on all ranks
    if rank == 0:
        x_est = ddp.module.x.detach().cpu()
        torch.save(x_est, "x_est.pt")
        print("Saved x_est.pt")

if __name__ == "__main__":
    main()
