from __future__ import annotations

import copy
from typing import Callable

import torch

from deepinv.optim.data_fidelity import DataFidelity

from deepinv.distributed.framework.distributed_utils import map_reduce_gather
from deepinv.distributed.framework.distributed_context import DistributedContext
from deepinv.distributed.framework.distributed_physics import (
    DistributedStackedLinearPhysics,
)


class DistributedDataFidelity(torch.nn.Module):
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
        super().__init__()
        self.ctx = ctx
        self.reduction_mode = reduction
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
            local_data_fidelities = []
            for i in local_indexes:
                df = data_fidelity(i, ctx.device, factory_kwargs)
                local_data_fidelities.append(df)
            # Register as ModuleList for proper parameter management
            self.local_data_fidelities = torch.nn.ModuleList(local_data_fidelities)
        else:
            raise ValueError(
                "data_fidelity must be a DataFidelity instance or a factory callable."
            )

    def _get_fidelity(self, i: int) -> DataFidelity:
        if self.single_fidelity is not None:
            return self.single_fidelity
        if hasattr(self, "local_data_fidelities"):
            return self.local_data_fidelities[i]
        raise ValueError("No data fidelity available.")

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

        # Compute local forward measurements through the distributed physics API.
        # We request the internal graph-anchor tensor so empty ranks can still
        # participate in backward sync through the downstream reduction anchor.
        Ax_local, graph_anchor = physics.A(
            x,
            gather=False,
            force_input_grad_sync=True,
            return_graph_anchor=True,
            **kwargs,
        )

        if len(y) != physics.num_operators:
            raise ValueError(
                f"Input y has length {len(y)}, expected {physics.num_operators} "
                "(global measurements)."
            )
        y_local = [y[i] for i in physics.local_indexes]

        # Zip Ax and y for mapping
        if len(Ax_local) != len(y_local):
            raise ValueError("Ax and y local sizes do not match.")

        zipped_data = list(zip(Ax_local, y_local, strict=True))
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
            graph_anchor=graph_anchor,
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

    def prox(
        self,
        x: torch.Tensor,
        y: list[torch.Tensor],
        physics: DistributedStackedLinearPhysics,
        *args,
        gamma=1.0,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute proximal step for distributed data-fidelity.

        Currently supported when a single shared DataFidelity object is used for all
        operators (the standard unfolded setup). In that case, the prox is delegated
        to the wrapped DataFidelity with the distributed physics object.
        """
        self._check_is_distributed_physics(physics)
        if self.single_fidelity is None:
            raise NotImplementedError(
                "prox is only supported when DistributedDataFidelity wraps a single shared DataFidelity instance."
            )
        return self.single_fidelity.prox(
            x, y, physics=physics, *args, gamma=gamma, **kwargs
        )
