from __future__ import annotations

import os
import copy
from typing import Callable, Sequence
import warnings

import torch
import torch.distributed as dist

from deepinv.physics import Physics, LinearPhysics
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.utils.tensorlist import TensorList

from deepinv.distributed.strategies import DistributedSignalStrategy, create_strategy
from deepinv.distributed.distributed_utils import map_reduce_gather


# =========================
# Distributed Context
# =========================
class DistributedContext:
    r"""
    Context manager for distributed computing.

    Handles:
      - Initialization/destruction of the process group (if `RANK` / `WORLD_SIZE` environment variables exist)
      - Backend choice: NCCL when one-GPU-per-process per node, else Gloo.
      - Device selection based on `LOCAL_RANK` and visible GPUs
      - Sharding helpers and tiny communication helpers

    :param str | None backend: backend to use for distributed communication. If `None` (default), automatically selects NCCL for GPU or Gloo for CPU.
    :param bool cleanup: whether to clean up the process group on exit. Default is `True`.
    :param int | None seed: random seed for reproducible results. If provided, behavior depends on `seed_offset`. Default is `None`.
    :param bool seed_offset: whether to add rank offset to seed (each rank gets `seed + rank`). Default is `True`.
        When `True`: each process uses a unique seed for diverse random sequences.
        When `False`: all processes share the same seed for synchronized randomness.
    :param bool deterministic: whether to use deterministic cuDNN operations. Default is `False`.
    :param str | None device_mode: device selection mode. Options are `'cpu'`, `'gpu'`, or `None` for automatic. Default is `None`.

    """

    def __init__(
        self,
        backend: str | None = None,
        cleanup: bool = True,
        seed: int | None = None,
        seed_offset: bool = True,
        deterministic: bool = False,
        device_mode: str | None = None,
    ):
        r"""
        Initialize the distributed context manager.
        """
        self.backend = backend
        self.cleanup = cleanup
        self.seed = seed
        self.seed_offset = seed_offset
        self.deterministic = deterministic
        self.device_mode = device_mode  # "cpu", "gpu", or None (auto)

        # set in __enter__
        self.created_dist = False
        self.use_dist = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.local_world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __enter__(self):
        # Detect whether we should initialize a process group
        env_has_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
        should_init_pg = (not dist.is_initialized()) and env_has_dist

        # Count GPUs *visible to this process* (respects CUDA_VISIBLE_DEVICES)
        visible_gpus = torch.cuda.device_count()
        cuda_ok = torch.cuda.is_available() and (visible_gpus > 0)

        # Get number of processes and ranks
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if should_init_pg:
            backend = self.backend
            if backend is None:
                # Backend decision considering device_mode:
                #   - If device_mode is "cpu", always use Gloo
                #   - If device_mode is "gpu", always use NCCL (will fail if no GPU)
                #   - If auto (None), decide based on available resources:
                #     * If each node has at least as many *visible* GPUs as processes per node -> NCCL
                #     * Otherwise -> Gloo (e.g., GPU oversubscription or CPU)
                if self.device_mode == "cpu":
                    backend = "gloo"
                elif self.device_mode == "gpu":
                    if not dist.is_nccl_available():
                        raise RuntimeError(
                            "GPU mode requested but NCCL backend not available"
                        )
                    backend = "nccl"
                else:
                    # Auto mode
                    if (
                        cuda_ok
                        and dist.is_nccl_available()
                        and (self.local_world_size <= visible_gpus)
                    ):
                        backend = "nccl"
                    else:
                        backend = "gloo"

            dist.init_process_group(backend=backend)
            self.created_dist = True

        # Refresh flags from the actual PG (in case we didn't init)
        self.use_dist = dist.is_initialized()
        if self.use_dist:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        # ---- Device selection ----
        if self.device_mode == "cpu":
            # Force CPU mode
            self.device = torch.device("cpu")
        elif self.device_mode == "gpu":
            # Force GPU mode (require CUDA)
            if not cuda_ok:
                raise RuntimeError(
                    "GPU mode requested but CUDA not available or no visible GPUs"
                )
            if visible_gpus == 1:
                # GPU isolation is handled externally - use the only visible GPU
                dev_index = 0
            else:
                # Multiple GPUs visible - map local_rank to device index
                dev_index = self.local_rank % visible_gpus
            self.device = torch.device(f"cuda:{dev_index}")
            torch.cuda.set_device(self.device)
        else:
            # Auto mode
            if cuda_ok:
                # When CUDA_VISIBLE_DEVICES is set externally (e.g., by SLURM/submitit),
                # each process sees only its assigned GPU(s). In this case, if there's
                # only 1 visible GPU per process, just use cuda:0 for all processes.
                # Otherwise, map local_rank onto visible devices.
                if visible_gpus == 1:
                    # GPU isolation is handled externally - use the only visible GPU
                    dev_index = 0
                else:
                    # Multiple GPUs visible - map local_rank to device index
                    dev_index = self.local_rank % visible_gpus

                self.device = torch.device(f"cuda:{dev_index}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

        self._post_init_setup()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Only destroy process group if:
        # 1. cleanup=True (caller wants cleanup)
        # 2. We initialized it (created_dist=True)
        # 3. It's still initialized
        if self.cleanup and self.created_dist and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    # ----------------------
    # Post-init knobs
    # ----------------------
    def _post_init_setup(self):
        # Seeding (rank offset for reproducible-but-unique RNG per process)
        if self.seed is not None:
            s = self.seed + (self.rank if (self.use_dist and self.seed_offset) else 0)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

        # Deterministic cuDNN
        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # ----------------------
    # Sharding
    # ----------------------
    def local_indices(self, num_items: int) -> list[int]:
        r"""
        Get local indices for this rank based on round robin sharding.

        :param int num_items: total number of items to shard.
        :return: list of indices assigned to this rank.
        """
        indices = [i for i in range(num_items) if (i % self.world_size) == self.rank]

        # Warning for efficiency, but allow empty indices (don't raise error)
        if self.use_dist and len(indices) == 0 and self.rank == 0:
            warnings.warn(
                f"Some ranks have no work items to process "
                f"(num_items={num_items}, world_size={self.world_size}). "
                f"Consider reducing world_size or increasing the workload for better efficiency.",
                UserWarning,
            )

        return indices

    # ----------------------
    # Collectives
    # ----------------------
    def __getattr__(self, name):
        if hasattr(dist, name):

            def wrapper(*args, **kwargs):
                if self.use_dist:
                    return getattr(dist, name)(*args, **kwargs)
                return None

            return wrapper
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# =========================
# Distributed Physics
# =========================
class DistributedStackedPhysics(Physics):
    r"""
    Holds only local physics operators. Exposes fast local and compatible global APIs.

    This class distributes a *collection* of physics operators across multiple processes,
    where each process owns a subset of the operators.

    .. note::

        It is intended to parallelize models naturally expressed as a stack/list of operators
        (e.g., :class:`deepinv.physics.StackedPhysics` or an explicit Python list of
        :class:`deepinv.physics.Physics` objects) and is **not** meant to split a
        single monolithic physics operator across ranks.

    If your forward model is a single operator that can be decomposed into multiple
    sub-operators, it is up to you to perform that decomposition (e.g., build a
    :class:`deepinv.physics.StackedPhysics`) and then pass that collection to
    :class:`DistributedStackedPhysics` via the ``factory`` argument.

    :param DistributedContext ctx: distributed context manager.
    :param int num_operators: total number of physics operators.
    :param Callable factory: factory function that creates physics operators. Should have signature `factory(index, device, factory_kwargs) -> Physics`.
    :param dict | None factory_kwargs: shared data dictionary passed to factory function. Default is `None`.
    :param torch.dtype | None dtype: data type for operations. Default is `None`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)

        Default is `'concatenated'`.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_operators: int,
        factory: Callable[[int, torch.device, dict | None], Physics],
        *,
        factory_kwargs: dict | None = None,
        dtype: torch.dtype | None = None,
        gather_strategy: str = "concatenated",
        **kwargs,
    ):
        r"""
        Initialize distributed physics operators.
        """
        super().__init__(**kwargs)
        self.ctx = ctx
        self.dtype = dtype or torch.float32
        self.num_operators = num_operators
        self.local_indexes: list[int] = ctx.local_indices(num_operators)

        # Validate and set gather strategy
        valid_strategies = ("naive", "concatenated", "broadcast")
        if gather_strategy not in valid_strategies:
            raise ValueError(
                f"gather_strategy must be one of {valid_strategies}, got '{gather_strategy}'"
            )
        self.gather_strategy = gather_strategy

        # Broadcast shared object (lightweight) once (root=0) if present
        self.factory_kwargs = factory_kwargs
        if ctx.use_dist and factory_kwargs is not None:
            obj = [factory_kwargs if ctx.rank == 0 else None]
            self.ctx.broadcast_object_list(obj, src=0)
            self.factory_kwargs = obj[0]

        # Build local physics
        self.local_physics: list[Physics] = []
        for i in self.local_indexes:
            p_i = factory(i, ctx.device, self.factory_kwargs)
            self.local_physics.append(p_i)

    # -------- Factorized map-reduce logic --------
    def _map_reduce_gather(
        self,
        x: torch.Tensor | list[torch.Tensor],
        local_op: Callable,
        gather: bool = True,
        reduce_op: str | None = "sum",
        **kwargs,
    ) -> TensorList | list[torch.Tensor] | torch.Tensor:
        """
        Map-reduce pattern for distributed operations.

        Delegates to generic implementation in distributed_utils.
        """
        return map_reduce_gather(
            ctx=self.ctx,
            local_items=self.local_physics,
            x=x,
            local_op=local_op,
            local_indices=self.local_indexes,
            num_operators=self.num_operators,
            gather_strategy=self.gather_strategy,
            dtype=self.dtype,
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )

    def A(
        self,
        x: torch.Tensor,
        gather: bool = True,
        reduce_op: str | None = None,
        **kwargs,
    ) -> TensorList | list[torch.Tensor]:
        r"""
        Apply forward operator to all distributed physics operators with automatic gathering.

        Applies the forward operator :math:`A(x)` by computing local measurements and gathering
        results from all ranks using the configured gather strategy.

        :param torch.Tensor x: input signal.
        :param bool gather: whether to gather results across ranks. If `False`, returns local measurements. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. Default is `None`.
        :param kwargs: optional parameters for the forward operator.
        :return: complete list of measurements from all operators (or local list if `reduce=False`).
        """

        return self._map_reduce_gather(
            x,
            lambda p, x, **kw: p.A(x, **kw),
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )

    def forward(self, x, gather: bool = True, reduce_op: str | None = None, **kwargs):
        r"""
        Apply full forward model with sensor and noise models to the input signal and gather results.

        .. math::

            y = N(A(x))

        :param torch.Tensor x: input signal.
        :param bool gather: whether to gather results across ranks. If `False`, returns local measurements. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. Default is `None`.
        :param kwargs: optional parameters for the forward model.
        :return: complete list of noisy measurements from all operators.
        """

        return self._map_reduce_gather(
            x,
            lambda p, x, **kw: p.forward(x, **kw),
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )


class DistributedStackedLinearPhysics(DistributedStackedPhysics, LinearPhysics):
    r"""
    Distributed linear physics operators.

    This class extends :class:`DistributedStackedPhysics` for linear operators. It provides distributed
    operations that automatically handle communication and reductions.

    .. note::

        This class is intended to distribute a *collection* of linear operators (e.g.,
        :class:`deepinv.physics.StackedLinearPhysics` or an explicit Python list of
        :class:`deepinv.physics.LinearPhysics` objects) across ranks. It is **not** a
        mechanism to shard a single linear operator internally.

    If you have one linear physics operator that can naturally be split into multiple
    operators, you must do that split yourself (build a stacked/list representation) and
    provide those operators through the `factory`.

    All linear operations (`A_adjoint`, `A_vjp`, etc.) support a `reduce_op` parameter:

        - If `reduce_op='sum'` (default): The method computes the global result by performing a single all-reduce across all ranks.
        - If `reduce_op=None`: The method computes only the local contribution from operators owned by this rank, without any inter-rank communication. This is useful for deferring reductions in custom algorithms. Beware, using `reduce_op=None` may lead to errors or incorrect results if the full global operation is required (e.g., for correct gradients in training) and should be used with caution.

    :param DistributedContext ctx: distributed context manager.
    :param int num_operators: total number of physics operators to distribute.
    :param Callable factory: factory function that creates linear physics operators.
        Should have signature `factory(index: int, device: torch.device, factory_kwargs: dict | None) -> LinearPhysics`.
    :param dict | None factory_kwargs: shared data dictionary passed to factory function for all operators. Default is `None`.
    :param torch.dtype | None dtype: data type for operations. Default is `None`.
    :param str gather_strategy: strategy for gathering distributed results in forward operations.
        Options are `'naive'`, `'concatenated'`, or `'broadcast'`. Default is `'concatenated'`.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        num_operators: int,
        factory,
        *,
        factory_kwargs: dict | None = None,
        dtype: torch.dtype | None = None,
        gather_strategy: str = "concatenated",
        **kwargs,
    ):
        r"""
        Initialize distributed linear physics operators.
        """
        super().__init__(
            ctx=ctx,
            num_operators=num_operators,
            factory=factory,
            factory_kwargs=factory_kwargs,
            dtype=dtype,
            gather_strategy=gather_strategy,
            A=lambda x, **kw: x,
            A_adjoint=lambda y, **kw: y,
            **kwargs,
        )

        for p in self.local_physics:
            if not isinstance(p, LinearPhysics):
                raise ValueError("factory must return LinearPhysics instances.")

    def A_adjoint(
        self,
        y: TensorList | list[torch.Tensor],
        gather: bool = True,
        reduce_op: str | None = "sum",
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute global adjoint operation with automatic reduction.

        Extracts local measurements, computes local adjoint contributions, and sum reduces
        across all ranks to obtain the complete :math:`A^T y = \sum_{i=1}^n A_i^T y_i` where :math:`A` is the
        stacked operator :math:`A = [A_1, A_2, \ldots, A_n]`, :math:`A_i` and :math:`y_i` are the individual linear operators and measurements respectively.

        :param TensorList | list[torch.Tensor] y: full list of measurements from all operators.
        :param bool gather: whether to gather results across ranks. If False, returns local contribution. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. If `None`, no reduction (neither local or global) is performed, reduction must be performed manually to produce complete adjoint result :math:`A^T y`. Default is `"sum"`.
        :param kwargs: optional parameters for the adjoint operation.
        :return: complete adjoint result :math:`A^T y` (or local contribution if gather=False).
        """

        # Extract local measurements
        if len(y) == self.num_operators:
            y_local = [y[i] for i in self.local_indexes]
        elif len(y) == len(self.local_indexes):
            y_local = y
        else:
            raise ValueError(
                f"Input y has length {len(y)}, expected {self.num_operators} (global) or {len(self.local_indexes)} (local)."
            )

        # Use _map_reduce_gather with per-operator inputs and sum_results=True
        # This gathers all A_i^T(y_i) and sums them automatically
        return self._map_reduce_gather(
            y_local,
            lambda p, y_i, **kw: p.A_adjoint(y_i, **kw),
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )

    def A_vjp(
        self,
        x: torch.Tensor,
        v: TensorList | list[torch.Tensor],
        gather: bool = True,
        reduce_op: str | None = "sum",
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute global vector-Jacobian product with automatic reduction.

        Extracts local cotangent vectors, computes local VJP contributions, and reduces
        across all ranks to obtain the complete VJP.

        :param torch.Tensor x: input tensor.
        :param TensorList | list[torch.Tensor] v: full list of cotangent vectors from all operators.
        :param bool gather: whether to gather results across ranks. If False, returns local contribution. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. If `None`, no reduction (neither local or global) is performed, reduction must be performed manually to produce complete VJP result. Default is `"sum"`.
        :param kwargs: optional parameters for the VJP operation.
        :return: complete VJP result (or local contribution if gather=False).
        """

        if len(v) == self.num_operators:
            v_local = [v[i] for i in self.local_indexes]
        elif len(v) == len(self.local_indexes):
            v_local = v
        else:
            raise ValueError(
                f"Input v has length {len(v)}, expected {self.num_operators} (global) or {len(self.local_indexes)} (local)."
            )

        return self._map_reduce_gather(
            v_local,
            lambda p, v_i, **kw: p.A_vjp(x, v_i, **kw),
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )

    def A_adjoint_A(
        self,
        x: torch.Tensor,
        gather: bool = True,
        reduce_op: str | None = "sum",
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute global :math:`A^T A` operation with automatic reduction.

        Computes the complete normal operator :math:`A^T A x = \sum_i A_i^T A_i x` by
        combining local contributions from all ranks.

        :param torch.Tensor x: input tensor.
        :param bool gather: whether to gather results across ranks. If False, returns local contribution. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. If `None`, no reduction (neither local or global) is performed, reduction must be performed manually to produce complete  result :math:`A^T A x`. Default is `"sum"`.
        :param kwargs: optional parameters for the operation.
        :return: complete :math:`A^T A x` result (or local contribution if gather=False).
        """

        return self._map_reduce_gather(
            x,
            lambda p, x, **kw: p.A_adjoint_A(x, **kw),
            gather=gather,
            reduce_op=reduce_op,
            **kwargs,
        )

    def A_A_adjoint(
        self,
        y: TensorList | list[torch.Tensor],
        gather: bool = True,
        **kwargs,
    ) -> TensorList | list[torch.Tensor]:
        r"""
        Compute global :math:`A A^T` operation with automatic reduction.

        For stacked operators, this computes :math:`A A^T y` where :math:`A^T y = \sum_i A_i^T y_i`
        and then applies the forward operator to get :math:`[A_1(A^T y), A_2(A^T y), \ldots, A_n(A^T y)]`.

        .. note::

            Unlike other operations, the adjoint step `A^T y` is always computed globally (with full
            reduction across ranks) even when `gather=False`. This is because computing the correct
            `A_A_adjoint` requires the full adjoint `sum_i A_i^T y_i`. The `gather` parameter only
            controls whether the final forward operation `A(...)` is gathered across ranks.

        :param TensorList | list[torch.Tensor] y: full list of measurements from all operators.
        :param bool gather: whether to gather final results across ranks. If `False`, returns only local
            operators' contributions (but still uses the global adjoint). Default is `True`.
        :param kwargs: optional parameters for the operation.
        :return: TensorList with entries :math:`A_i A^T y` for all operators (or local list if `gather=False`).
        """

        # First compute A^T y globally (always with reduction to get the full adjoint)
        # This is necessary because A_A_adjoint(y) = A(A^T y) and A^T y = sum_i A_i^T y_i
        x_adjoint = self.A_adjoint(y, gather=True, **kwargs)
        # Then compute A(A^T y) which returns a TensorList (or list if reduce=False)
        return self.A(x_adjoint, gather=gather, reduce_op=None, **kwargs)

    def A_dagger(
        self,
        y: TensorList | list[torch.Tensor],
        solver: str = "CG",
        max_iter: int | None = None,
        tol: float | None = None,
        verbose: bool = False,
        *,
        local_only: bool = True,
        gather: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Distributed pseudoinverse computation. This method provides two strategies:

            1. **Local approximation** (`local_only=True`, default): Each rank computes the pseudoinverse
            of its local operators independently, then averages the results with a single reduction.
            This is efficient (minimal communication) but **provides only an approximation**.
            In other words, for stacked operators this computes

            .. math::

                A^\dagger y = \frac{1}{n} \sum_i A_i^\dagger y_i

            2. **Global computation** (`local_only=False`): Uses the full least squares solver
            with distributed :meth:`A_adjoint_A` and :meth:`A_A_adjoint` operations.
            This computes the exact pseudoinverse but requires communication at every iteration.

        :param TensorList | list[torch.Tensor] y: measurements to invert.
        :param str solver: least squares solver to use (only for `local_only=False`).
            Choose between `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'`. Default is `'CG'`.
        :param int | None max_iter: maximum number of iterations for least squares solver. Default is `None`.
        :param float | None tol: relative tolerance for least squares solver. Default is `None`.
        :param bool verbose: print information (only on rank 0). Default is `False`.
        :param bool local_only: If `True` (default), compute local daggers and sum-reduce (efficient).
            If `False`, compute exact global pseudoinverse with full communication (expensive). Default is `True`.
        :param bool gather: whether to gather results across ranks (only applies if local_only=True). Default is `True`.
        :param kwargs: optional parameters for the forward operator.

        :return: pseudoinverse solution. If `local_only=True`, returns approximation.
            If `local_only=False`, returns exact least squares solution.
        """
        if local_only:
            # Efficient local computation with single sum reduction
            if isinstance(y, TensorList):
                y_local = [y[i] for i in self.local_indexes]
            elif len(y) == self.num_operators:
                y_local = [y[i] for i in self.local_indexes]
            elif len(y) == len(self.local_indexes):
                y_local = y
            else:
                raise ValueError(
                    f"Input y has length {len(y)}, expected {self.num_operators} (global) or {len(self.local_indexes)} (local)."
                )

            return self._map_reduce_gather(
                y_local,
                lambda p, y_i, **kw: p.A_dagger(y_i, **kw),
                gather=gather,
                reduce_op="mean",
                **kwargs,
            )
        else:
            # Global computation: call parent class A_dagger which uses least squares
            # This will use our distributed A, A_adjoint, A_adjoint_A and A_A_adjoint
            # XXX: this could use a dedicated algorithm to faster computations
            verbose_flag = verbose and self.ctx.rank == 0
            return super().A_dagger(
                y,
                solver=solver,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose_flag,
                **kwargs,
            )

    def compute_sqnorm(
        self,
        x0: torch.Tensor,
        *,
        max_iter: int = 50,
        tol: float = 1e-3,
        verbose: bool = True,
        local_only: bool = True,
        gather: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the squared spectral :math:`\ell_2` norm of the distributed operator.

        This method provides two strategies:

            1. **Local approximation** (`local_only=True`, default): Each rank computes the norm
            of its local operators independently, then a single max-reduction provides an upper bound.
            This is efficient (minimal communication) and valid for conservative estimates.
            For stacked operators :math:`A = [A_1; A_2; \ldots; A_n]`, we have
            :math:`\|A\|^2 \leq \sum_i \|A_i\|^2`, and we use :math:`\max_i \|A_i\|^2` as
            a conservative upper bound.

            2. **Global computation** (`local_only=False`): Uses the full distributed :meth:`A_adjoint_A`
            with communication at every power iteration. This computes the exact norm but is
            communication-intensive.

        :param torch.Tensor x0: an unbatched tensor sharing its shape, dtype and device with the initial iterate.
        :param int max_iter: maximum number of iterations for power method. Default is `50`.
        :param float tol: relative variation criterion for convergence. Default is `1e-3`.
        :param bool verbose: print information (only on rank 0). Default is `True`.
        :param bool local_only: If `True` (default), compute local norms and max-reduce (efficient).
            If `False`, compute exact global norm with full communication (expensive). Default is `True`.
        :param bool gather: whether to gather results across ranks (only applies if local_only=True). Default is `True`.
        :param kwargs: optional parameters for the forward operator.

        :return: Squared spectral norm. If `local_only=True`, returns upper bound.
            If `local_only=False`, returns exact value.
        """

        if local_only:
            # Efficient local computation with single max reduction
            local_sqnorm = self._map_reduce_gather(
                x0,
                lambda p, x, **kw: p.compute_sqnorm(
                    x, max_iter=max_iter, tol=tol, verbose=False, **kw
                ),
                gather=gather,
                reduce_op="sum",
                **kwargs,
            )

            if verbose and self.ctx.rank == 0 and gather:
                print(
                    f"Computed local norm upper bound: ||A||_2^2 ≤ {local_sqnorm.item():.2f}"
                )

            return local_sqnorm

        else:
            # Global computation: call parent class compute_sqnorm which uses power method
            # This will use our distributed A_adjoint_A
            verbose_flag = verbose and self.ctx.rank == 0
            return super().compute_sqnorm(
                x0, max_iter=max_iter, tol=tol, verbose=verbose_flag, **kwargs
            )


class DistributedProcessing:
    r"""
    Distributed signal processing using pluggable tiling and reduction strategies.

    This class enables distributed processing of large signals (images, volumes, etc.) by:

        1. Splitting the signal into patches using a chosen strategy
        2. Distributing patches across multiple processes/GPUs
        3. Processing each patch independently using a provided processor function
        4. Combining processed patches back into the full signal with proper overlap handling

    The processor can be any callable that operates on tensors (e.g., denoisers, priors,
    neural networks, etc.). The class handles all distributed coordination automatically.

    |sep|

    **Example use cases:**

        - Distributed denoising of large images/volumes
        - Applying neural network priors across multiple GPUs
        - Processing signals too large to fit on a single device

    :param DistributedContext ctx: distributed context manager.
    :param Callable[[torch.Tensor], torch.Tensor] processor: processing function to apply to signal patches.
        Should accept a batched tensor of shape ``(N, C, ...)`` and return a tensor of the same shape.
        Examples: denoiser, neural network, prior gradient function, etc.
    :param str | DistributedSignalStrategy | None strategy: signal processing strategy for patch extraction
        and reduction. Either a strategy name (``'basic'``, ``'overlap_tiling'``) or a custom strategy instance.
        Default is ``'overlap_tiling'`` which handles overlapping patches with smooth blending.
    :param dict | None strategy_kwargs: additional keyword arguments passed to the strategy constructor
        when using string strategy names. Examples: ``patch_size``, ``overlap``, ``tiling_dims``. Default is `None`.
    :param int | None max_batch_size: maximum number of patches to process in a single batch.
        If ``None``, all local patches are batched together. Set to ``1`` for sequential processing
        (useful for memory-constrained scenarios). Higher values increase throughput but require more memory. Default is `None`.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        processor: Callable[[torch.Tensor], torch.Tensor],
        *,
        strategy: str | DistributedSignalStrategy | None = None,
        strategy_kwargs: dict | None = None,
        max_batch_size: int | None = None,
        **kwargs,
    ):
        r"""
        Initialize distributed signal processor.
        """
        self.ctx = ctx
        self.processor = processor
        self.max_batch_size = max_batch_size
        self.strategy = strategy if strategy is not None else "overlap_tiling"
        self.strategy_kwargs = strategy_kwargs or {}
        self.current_shape: torch.Size | None = None

        if hasattr(processor, "to"):
            self.processor.to(ctx.device)

    def __call__(
        self, x: torch.Tensor, *args, gather: bool = True, **kwargs
    ) -> torch.Tensor:
        r"""
        Apply distributed processing to input signal.

        :param torch.Tensor x: input signal tensor to process, typically of shape ``(B, C, H, W)`` for 2D
            or ``(B, C, D, H, W)`` for 3D signals.
        :param args: additional positional arguments passed to the processor.
        :param bool gather: whether to gather results across ranks. If False, returns local contribution. Default is `True`.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: processed signal with the same shape as input.
        """

        if self.current_shape != x.shape:
            self._init_shape_and_strategy(x.shape)
            self.current_shape = x.shape
        return self._apply_op(x, *args, gather=gather, **kwargs)

    # ---- internals --------------------------------------------------------

    def _init_shape_and_strategy(
        self,
        img_size: Sequence[int],
    ):
        r"""
        Initialize or update the signal shape and processing strategy.

        This method is called automatically on the first forward pass to set up the
        tiling strategy based on the input signal dimensions. It creates the strategy,
        determines the number of patches, and assigns patches to ranks.

        :param Sequence[int] img_size: full shape of the input signal tensor (e.g., ``(B, C, H, W)``).
        """

        self.img_size = torch.Size(img_size)
        tiling_dims = self.strategy_kwargs.pop("tiling_dims", None)

        # Create or set the strategy
        self._strategy = (
            create_strategy(
                self.strategy, img_size, tiling_dims=tiling_dims, **self.strategy_kwargs
            )
            if isinstance(self.strategy, str)
            else self.strategy
        )

        if self._strategy is None:
            raise RuntimeError(
                "Distributed strategy is None - failed to create or import strategy"
            )

        self.num_patches = self._strategy.get_num_patches()
        if self.num_patches == 0:
            raise ValueError("Distributed strategy produced zero patches.")

        # determine local patch indices for this rank
        self.local_indices: list[int] = list(self.ctx.local_indices(self.num_patches))

        # Check for insufficient work distribution
        if self.ctx.use_dist and self.ctx.rank == 0:
            ranks_with_work = sum(
                1
                for rank in range(self.ctx.world_size)
                if len(self.ctx.local_indices(self.num_patches)) > 0
            )
            if ranks_with_work < self.ctx.world_size:
                warnings.warn(
                    f"Only {ranks_with_work}/{self.ctx.world_size} ranks have patches to process. "
                    f"Some ranks will have no work. "
                    f"Current: {self.num_patches} patches for {self.ctx.world_size} ranks.",
                    UserWarning,
                )

    def _apply_op(
        self, x: torch.Tensor, *args, gather: bool = True, **kwargs
    ) -> torch.Tensor:
        r"""
        Apply processor using the distributed strategy (internal method).

        This method orchestrates the complete distributed processing pipeline:

            1. Extracts local patches from the input signal using the strategy
            2. Applies batching as defined by the strategy and max_batch_size
            3. Applies the processor function to each batch of patches
            4. Unpacks batched results back to individual patches
            5. Reduces patches back to the output signal using the strategy's blending
            6. All-reduces the final result across ranks to combine overlapping regions

        :param torch.Tensor x: input signal to process.
        :param bool gather: whether to gather results across ranks. If False, returns local contribution.
        :param args: additional positional arguments passed to the processor.
        :param kwargs: additional keyword arguments passed to the processor.
        :return: processed signal with the same shape as input.
        """
        # Handle empty case early
        if not self.local_indices:
            out_local = torch.zeros(
                self.img_size, device=self.ctx.device, dtype=x.dtype
            )
            if gather:
                self.ctx.all_reduce(out_local, op=dist.ReduceOp.SUM)
            return out_local

        # 1. Extract local patches using strategy
        local_pairs = self._strategy.get_local_patches(x, self.local_indices)
        patches = [patch for _, patch in local_pairs]

        # 2. Apply batching strategy with max_batch_size
        batched_patches = self._strategy.apply_batching(
            patches, max_batch_size=self.max_batch_size
        )

        # 3. Apply processor to each batch
        processed_batches = []
        for batch in batched_patches:
            result = self.processor(batch, *args, **kwargs)
            processed_batches.append(result)

        # 4. Unpack results back to individual patches
        processed_patches = self._strategy.unpack_batched_results(
            processed_batches, len(patches)
        )

        # 5. Pair with global indices
        if len(processed_patches) != len(self.local_indices):
            raise RuntimeError(
                f"Mismatch between processed patches ({len(processed_patches)}) "
                f"and local indices ({len(self.local_indices)})"
            )

        processed_pairs = list(zip(self.local_indices, processed_patches))

        # 6. Initialize output tensor and apply reduction strategy
        out_local = torch.zeros(self.img_size, device=self.ctx.device, dtype=x.dtype)
        self._strategy.reduce_patches(out_local, processed_pairs)

        # 7. All-reduce to combine results from all ranks
        if gather:
            self.ctx.all_reduce(out_local, op=dist.ReduceOp.SUM)

        return out_local


# =========================
# Distributed Data Fidelity
# =========================
class DistributedDataFidelity:
    r"""
    Distributed data fidelity term for use with distributed physics operators.

    This class wraps a standard DataFidelity object and makes it compatible with
    DistributedStackedLinearPhysics by implementing efficient distributed computation patterns.
    It computes data fidelity terms and gradients using local operations followed by
    a single reduction, avoiding redundant communication.

    The key operations are:

        - ``fn(x, y, physics)``: Computes the data fidelity :math:`\sum_i d(A_i(x), y_i)`
        - ``grad(x, y, physics)``: Computes the gradient :math:`\sum_i A_i^T \nabla d(A_i(x), y_i)`

    Both operations use an efficient pattern:

        1. Compute local forward operations (A_local)
        2. Apply distance function and compute gradients locally
        3. Perform a single reduction across ranks

    :param DistributedContext ctx: distributed context manager.
    :param DataFidelity | Callable data_fidelity: either a DataFidelity instance
        or a factory function that creates DataFidelity instances for each operator.
        The factory should have signature
        ``factory(index: int, device: torch.device, factory_kwargs: dict | None) -> DataFidelity``.
    :param int | None num_operators: number of operators (required if data_fidelity is a factory). Default is `None`.
    :param dict | None factory_kwargs: shared data dictionary passed to factory function for all operators. Default is `None`.
    :param str reduction: reduction mode matching the distributed physics. Options are ``'sum'`` or ``'mean'``.
        Default is ``'sum'``.
    """

    def __init__(
        self,
        ctx: DistributedContext,
        data_fidelity: (
            DataFidelity | Callable[[int, torch.device, dict | None], DataFidelity]
        ),
        num_operators: int | None = None,
        *,
        factory_kwargs: dict | None = None,
        reduction: str = "sum",
    ):
        r"""
        Initialize distributed data fidelity.

        :param DistributedContext ctx: distributed context manager.
        :param DataFidelity | Callable data_fidelity: data fidelity term or factory.
        :param int | None num_operators: number of operators (required if data_fidelity is a factory). Default is `None`.
        :param dict | None factory_kwargs: shared data dictionary passed to factory function. Default is `None`.
        :param str reduction: reduction mode for distributed operations. Options are ``'sum'`` and ``'mean'``. Default is ``'sum'``.
        """
        self.ctx = ctx
        self.reduction_mode = reduction
        self.local_data_fidelities = []
        self.single_fidelity = None

        if isinstance(data_fidelity, DataFidelity):
            self.single_fidelity = copy.deepcopy(data_fidelity)
            if hasattr(self.single_fidelity, "to"):
                self.single_fidelity.to(ctx.device)
        elif callable(data_fidelity):
            if num_operators is None:
                raise ValueError("num_operators must be provided when using a factory.")
            # Create local data fidelity instances using factory
            local_indexes = list(ctx.local_indices(num_operators))
            for i in local_indexes:
                df = data_fidelity(i, ctx.device, factory_kwargs)
                self.local_data_fidelities.append(df)
        else:
            raise ValueError(
                "data_fidelity must be a DataFidelity instance or a factory callable."
            )

    def _get_fidelity(self, i: int) -> DataFidelity:
        if self.single_fidelity is not None:
            return self.single_fidelity
        return self.local_data_fidelities[i]

    def _check_is_distributed_physics(self, physics: DistributedStackedLinearPhysics):
        if not isinstance(physics, DistributedStackedLinearPhysics):
            raise ValueError(
                "physics must be a DistributedStackedLinearPhysics instance to be used with DistributedDataFidelity."
            )

    def _apply_op(
        self,
        local_op: Callable,
        x: torch.Tensor,
        y: list[torch.Tensor],
        physics: DistributedStackedLinearPhysics,
        gather: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply a local operation across distributed physics with map-reduce-gather pattern.

        :param Callable local_op: local operation to apply. Should have signature `local_op(index: int, data: Any, **kwargs) -> torch.Tensor`.
        :param torch.Tensor x: input signal.
        :param list[torch.Tensor] y: measurements (TensorList or list of tensors).
        :param DistributedStackedLinearPhysics physics: distributed physics operator.
        :param bool gather: whether to gather results across ranks. Default is `True`.
        :param args: additional positional arguments passed to the local operation.
        :param kwargs: additional keyword arguments passed to the local operation.
        :return: result of applying the local operation and reducing across ranks.
        """

        self._check_is_distributed_physics(physics)

        # Get local measurements
        y_local = [y[i] for i in physics.local_indexes]

        # Compute A(x) locally
        Ax_local = physics.A(x, gather=False, **kwargs)

        # Zip Ax and y for mapping
        if len(Ax_local) != len(y_local):
            raise ValueError("Ax and y local sizes do not match.")

        zipped_data = list(zip(Ax_local, y_local))
        # Pseudo items to iterate on
        local_items = list(range(len(physics.local_indexes)))

        if len(Ax_local) == 0:
            local_items = []
            zipped_data = []

        return map_reduce_gather(
            ctx=self.ctx,
            local_items=local_items,
            x=zipped_data,
            local_op=local_op,
            local_indices=physics.local_indexes,
            num_operators=physics.num_operators,
            gather=gather,
            reduce_op=self.reduction_mode,
            **kwargs,
        )

    def fn(
        self,
        x: torch.Tensor,
        y: list[torch.Tensor],
        physics: DistributedStackedLinearPhysics,
        gather: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the distributed data fidelity term.

        For distributed physics with operators :math:`\{A_i\}` and measurements :math:`\{y_i\}`,
        computes:

        .. math::

            f(x) = \sum_i d(A_i(x), y_i)

        This is computed efficiently by:

            1. Each rank computes :math:`A_i(x)` for its local operators
            2. Each rank computes :math:`\sum_{i \in \text{local}} d(A_i(x), y_i)`
            3. Results are reduced across all ranks

        :param torch.Tensor x: input signal at which to evaluate the data fidelity.
        :param list[torch.Tensor] y: measurements (TensorList or list of tensors).
        :param DistributedStackedLinearPhysics physics: distributed physics operator.
        :param bool gather: whether to gather (reduce) results across ranks. Default is `True`.
        :param args: additional positional arguments passed to the distance function.
        :param kwargs: additional keyword arguments passed to the distance function.
        :return: scalar data fidelity value.
        """

        def _local_fidelity_op(idx, data, **kw):
            Ax_i, y_i = data
            return self._get_fidelity(idx).d.fn(Ax_i, y_i, *args, **kw)

        return self._apply_op(
            local_op=_local_fidelity_op,
            x=x,
            y=y,
            physics=physics,
            gather=gather,
            **kwargs,
        )

    def grad(
        self,
        x: torch.Tensor,
        y: list[torch.Tensor],
        physics: DistributedStackedLinearPhysics,
        gather: bool = True,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the gradient of the distributed data fidelity term.

        For distributed physics with operators :math:`\{A_i\}` and measurements :math:`\{y_i\}`,
        computes:

        .. math::

            \nabla_x f(x) = \sum_i A_i^T \nabla d(A_i(x), y_i)

        This is computed efficiently by:

            1. Each rank computes :math:`A_i(x)` for its local operators
            2. Each rank computes :math:`\nabla d(A_i(x), y_i)` for its local operators
            3. Each rank computes :math:`\sum_{i \in \text{local}} A_i^T \nabla d(A_i(x), y_i)` using A_vjp_local
            4. Results are reduced across all ranks

        :param torch.Tensor x: input signal at which to compute the gradient.
        :param list[torch.Tensor] y: measurements (TensorList or list of tensors).
        :param DistributedStackedLinearPhysics physics: distributed physics operator.
        :param bool gather: whether to gather (reduce) results across ranks. Default is `True`.
        :param args: additional positional arguments passed to the distance function gradient.
        :param kwargs: additional keyword arguments passed to the distance function gradient.
        :return: gradient with same shape as x.
        """

        def _local_grad_op(idx, data, **kw):
            Ax_i, y_i = data
            grad_d = self._get_fidelity(idx).d.grad(Ax_i, y_i, *args, **kw)
            return physics.local_physics[idx].A_vjp(x, grad_d, **kw)

        return self._apply_op(
            local_op=_local_grad_op, x=x, y=y, physics=physics, gather=gather, **kwargs
        )
