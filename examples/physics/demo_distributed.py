import torch
import torch.distributed as dist

from deepinv.utils.distrib import DistributedContext
from deepinv.physics.forward import LinearPhysics, StackedLinearPhysics, DistributedLinearPhysics


class GaussianLinear(LinearPhysics):
    def __init__(self, A_mat: torch.Tensor):
        super().__init__(A=self._A, A_adjoint=self._A_adj)
        self.register_buffer("A_mat", A_mat)
    def _A(self, x):     return x @ self.A_mat.t()
    def _A_adj(self, y): return y @ self.A_mat


def main():
    batch, dim_in, dim_out, n_ops = 2, 8, 5, 4

    with DistributedContext(seed=123, deterministic=True, verbose=True) as ctx:
        # Rank-0 builds shared state (CPU); DistributedLinearPhysics will broadcast it
        shared = None
        if ctx.is_rank0():
            mats = torch.randn(n_ops, dim_out, dim_in, device="cpu")
            shared = {"mats": mats}

        # Factory builds only the local operator i on the provided device
        def factory(i: int, device: torch.device, shared_dict):
            A_mat = shared_dict["mats"][i].to(device, non_blocking=True)
            return GaussianLinear(A_mat)

        print(f"Rank {ctx.info.rank} building distributed physics")

        # Build distributed operator (no device setup needed here; ctx handled it)
        dp = DistributedLinearPhysics(factory=factory, num_ops=n_ops, shared=shared, reduction="sum")

        print(f"Rank {ctx.info.rank} done building distributed physics")

        # Input x: user provides once on rank-0; class broadcasts automatically
        if ctx.is_rank0():
            x0 = torch.randn(batch, dim_in, device="cpu")
        else:
            x0 = torch.empty(0)  # placeholder; will be ignored

        print(f"Rank {ctx.info.rank} broadcasting input")

        # Forward on distributed
        y_dist = dp.A(x0)

        print(f"Rank {ctx.info.rank} done for forward")

        # Adjoint on distributed (all ranks get the same result with all_reduce)
        xadj_dist = dp.A_adjoint(y_dist)

        print(f"Rank {ctx.info.rank} done for adjoint")

        # Reference on rank-0 for parity check
        if ctx.is_rank0():
            physics_ref = [GaussianLinear(shared["mats"][i]) for i in range(n_ops)]
            stacked = StackedLinearPhysics(physics_ref, reduction="sum")
            y_ref = stacked.A(x0)
            xadj_ref = stacked.A_adjoint(y_ref)

            # Compare
            max_fwd = max(float((y_ref[i] - y_dist[i].cpu()).abs().max()) for i in range(n_ops))
            max_adj = float((xadj_ref - xadj_dist.cpu()).abs().max())
            print(f"max |fwd diff| = {max_fwd:.3e},  max |adj diff| = {max_adj:.3e}")
            assert max_fwd < 1e-6 and max_adj < 1e-6, "Mismatch!"


if __name__ == "__main__":
    main()
