from __future__ import annotations

from typing import Callable

import torch

from deepinv.physics import Physics, LinearPhysics
from deepinv.utils.tensorlist import TensorList

from deepinv.distributed.framework.distributed_utils import (
    DistributedGradientSync,
    map_reduce_gather,
)
from deepinv.distributed.framework.distributed_context import DistributedContext


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
        force_input_grad_sync: bool = False,
        return_graph_anchor: bool = False,
        **kwargs,
    ) -> (
        TensorList
        | list[torch.Tensor]
        | torch.Tensor
        | tuple[
            TensorList | list[torch.Tensor] | torch.Tensor,
            torch.Tensor | list[torch.Tensor],
        ]
    ):
        """
        Map-reduce pattern for distributed operations.

        Delegates to generic implementation in distributed_utils.
        """
        # Pure local mode (no reduction/gather) should not force cross-rank gradient
        # synchronization: outputs are local contributions and some ranks may hold no
        # items, which would otherwise create mismatched backward collectives.
        local_only_mode = (reduce_op is None) and (not gather)

        # If input is a tensor that requires grad, wrap it with DistributedGradientSync
        # to ensure gradients are properly synchronized across ranks during backward pass.
        if (
            isinstance(x, torch.Tensor)
            and x.requires_grad
            and self.ctx.use_dist
            and (force_input_grad_sync or not local_only_mode)
        ):
            x = DistributedGradientSync.apply(x, self.ctx)

        out = map_reduce_gather(
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
        if return_graph_anchor:
            return out, x
        return out

    def A(
        self,
        x: torch.Tensor,
        gather: bool = True,
        reduce_op: str | None = None,
        force_input_grad_sync: bool = False,
        return_graph_anchor: bool = False,
        **kwargs,
    ) -> (
        TensorList
        | list[torch.Tensor]
        | tuple[TensorList | list[torch.Tensor], torch.Tensor]
    ):
        r"""
        Apply forward operator to all distributed physics operators with automatic gathering.

        Applies the forward operator :math:`A(x)` by computing local measurements and gathering
        results from all ranks using the configured gather strategy.

        :param torch.Tensor x: input signal.
        :param bool gather: whether to gather results across ranks. If `False`, returns local measurements. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. Default is `None`.
        :param bool force_input_grad_sync: force synchronization of input gradients even
            in pure local mode (``gather=False`` and ``reduce_op=None``). Default is `False`.
        :param bool return_graph_anchor: if `True`, also return the tensor used as graph
            anchor inside this call. Intended for advanced internal usage. Default is `False`.
        :param kwargs: optional parameters for the forward operator.
        :return: complete list of measurements from all operators (or local list if `reduce=False`).
        """

        return self._map_reduce_gather(
            x,
            lambda p, x, **kw: p.A(x, **kw),
            gather=gather,
            reduce_op=reduce_op,
            force_input_grad_sync=force_input_grad_sync,
            return_graph_anchor=return_graph_anchor,
            **kwargs,
        )

    def forward(
        self,
        x,
        gather: bool = True,
        reduce_op: str | None = None,
        force_input_grad_sync: bool = False,
        return_graph_anchor: bool = False,
        **kwargs,
    ):
        r"""
        Apply full forward model with sensor and noise models to the input signal and gather results.

        .. math::

            y = N(A(x))

        :param torch.Tensor x: input signal.
        :param bool gather: whether to gather results across ranks. If `False`, returns local measurements. Default is `True`.
        :param str | None reduce_op: reduction operation to apply across ranks. Default is `None`.
        :param bool force_input_grad_sync: force synchronization of input gradients even
            in pure local mode (``gather=False`` and ``reduce_op=None``). Default is `False`.
        :param bool return_graph_anchor: if `True`, also return the tensor used as graph
            anchor inside this call. Intended for advanced internal usage. Default is `False`.
        :param kwargs: optional parameters for the forward model.
        :return: complete list of noisy measurements from all operators.
        """

        return self._map_reduce_gather(
            x,
            lambda p, x, **kw: p.forward(x, **kw),
            gather=gather,
            reduce_op=reduce_op,
            force_input_grad_sync=force_input_grad_sync,
            return_graph_anchor=return_graph_anchor,
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
