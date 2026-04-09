from __future__ import annotations

from typing import Callable, Sequence

import torch

from deepinv.physics import Physics, LinearPhysics
from deepinv.physics.forward import StackedPhysics, StackedLinearPhysics
from deepinv.optim.data_fidelity import DataFidelity, StackedPhysicsDataFidelity
from deepinv.optim.prior import Prior
from deepinv.optim.optimizers import BaseOptim
from deepinv.models.base import Denoiser

from deepinv.distributed.distrib_framework import (
    DistributedContext,
    DistributedStackedPhysics,
    DistributedStackedLinearPhysics,
    DistributedProcessing,
    DistributedDataFidelity,
    DistributedReplicatedParameters,
)

from deepinv.distributed.strategies import DistributedSignalStrategy


def _distribute_physics(
    physics: (
        StackedPhysics
        | list[Physics]
        | Callable[[int, torch.device, dict | None], Physics]
    ),
    ctx: DistributedContext,
    *,
    num_operators: int | None = None,
    type_object: str | None = "physics",
    dtype: torch.dtype | None = torch.float32,
    gather_strategy: str = "concatenated",
    **kwargs,
) -> DistributedStackedPhysics | DistributedStackedLinearPhysics:
    r"""
    Distribute a Physics object across multiple devices.

    :param StackedPhysics | list[Physics] | Callable physics: Physics object to distribute.
        Can be a StackedPhysics, list of Physics objects, or a factory function.
    :param DistributedContext ctx: distributed context manager.
    :param int | None num_operators: number of physics operators when using a factory for physics, otherwise inferred. Default is `None`.
    :param str | None type_object: type of physics object to distribute. Options are `'physics'` or `'linear_physics'`. Default is `'physics'`.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for DistributedStackedPhysics.

    :return: Distributed version of the input Physics object.
    """
    # Physics factory
    if isinstance(physics, (StackedPhysics, StackedLinearPhysics)):
        # Extract physics_list from StackedPhysics
        physics_list_extracted = physics.physics_list
        num_operators = len(physics_list_extracted)

        def physics_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
            return physics_list_extracted[idx].to(device)

    elif callable(physics):
        physics_factory = physics
        if num_operators is None or not isinstance(num_operators, int):
            raise ValueError(
                "When using a factory for physics, you must provide num_operators."
            )
    else:
        physics_list_extracted = physics
        num_operators = len(physics_list_extracted)

        def physics_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
            return physics_list_extracted[idx].to(device)

    if type_object == "linear_physics":
        return DistributedStackedLinearPhysics(
            ctx,
            num_operators=num_operators,
            factory=physics_factory,
            dtype=dtype,
            gather_strategy=gather_strategy,
            **kwargs,
        )
    else:
        return DistributedStackedPhysics(
            ctx,
            num_operators=num_operators,
            factory=physics_factory,
            dtype=dtype,
            gather_strategy=gather_strategy,
            **kwargs,
        )


def _distribute_processor(
    processor: Prior | Denoiser,
    ctx: DistributedContext,
    *,
    dtype: torch.dtype | None = torch.float32,
    tiling_strategy: str | DistributedSignalStrategy | None = None,
    patch_size: int = 256,
    overlap: int = 64,
    tiling_dims: int | tuple[int, ...] | None = None,
    max_batch_size: int | None = None,
    checkpoint_batches: str = "auto",
    checkpoint_use_reentrant: bool = False,
    checkpoint_preserve_rng_state: bool = True,
    gather_strategy: str = "concatenated",
    **kwargs,
) -> DistributedProcessing:
    r"""
    Distribute a DeepInverse prior or denoiser across multiple devices.

    :param Prior | Denoiser processor: DeepInverse prior or denoiser to distribute.
    :param DistributedContext ctx: distributed context manager.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str | DistributedSignalStrategy | None tiling_strategy: strategy for tiling the signal. Options are `'basic'`, `'overlap_tiling'`, or a custom strategy instance. Default is `None` (which defaults to `'overlap_tiling'`).
    :param int patch_size: size of patches for tiling strategies. Default is `256`.
    :param int overlap: receptive field size for overlap in tiling strategies. Default is `64`.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile over.
        Can be one of the following:
            - If ``None`` (default), tiles the last N-2 dimensions of your input tensor, i.e. for a tensor of shape (B, C, H, W), tiles over (H, W).
            - If an int ``N``, only tiles over the specified dimension.
            - If a tuple, specifies exact dimensions to tile.
    :param int | None max_batch_size: maximum number of patches to process in a single batch. If `None`, all patches are batched together. Set to `1` for sequential processing. Default is `None`.
    :param str checkpoint_batches: activation checkpointing mode for patch-batches during backward.
        Supported values are ``'auto'``, ``'always'`` and ``'never'``.
        Default is ``'auto'``.
    :param bool checkpoint_use_reentrant: reentrant mode forwarded to
        :func:`torch.utils.checkpoint.checkpoint`. Default is ``False``.
    :param bool checkpoint_preserve_rng_state: whether to preserve RNG state during
        checkpoint recomputation. Default is ``True``.
    :param str gather_strategy: strategy for gathering distributed results (currently unused for processors, kept for API consistency). Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for DistributedProcessing.

    :return: Distributed version of the input processor.
    """

    return DistributedProcessing(
        ctx=ctx,
        processor=processor,
        dtype=dtype,
        strategy=tiling_strategy,
        strategy_kwargs={
            "patch_size": patch_size,
            "overlap": overlap,
            "tiling_dims": tiling_dims,
        },
        max_batch_size=max_batch_size,
        checkpoint_batches=checkpoint_batches,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        checkpoint_preserve_rng_state=checkpoint_preserve_rng_state,
    )


def _distribute_data_fidelity(
    data_fidelity: (
        DataFidelity
        | StackedPhysicsDataFidelity
        | list[DataFidelity]
        | Callable[[int, torch.device, dict | None], DataFidelity]
    ),
    ctx: DistributedContext,
    num_operators: int | None = None,
    **kwargs,
) -> DistributedDataFidelity:
    r"""
    Distribute a DataFidelity object across multiple devices.

    :param DataFidelity | StackedPhysicsDataFidelity | list[DataFidelity] | Callable data_fidelity: DataFidelity object to distribute.
        Can be a DataFidelity, StackedPhysicsDataFidelity, list of DataFidelity objects, or a factory function.
    :param DistributedContext ctx: distributed context manager.
    :param int | None num_operators: number of data fidelity operators when using a factory for data_fidelity, otherwise inferred. Default is `None`.
    :param kwargs: additional keyword arguments for DistributedDataFidelity.

    :return: Distributed version of the input DataFidelity object.
    """
    # DataFidelity factory

    if isinstance(data_fidelity, DataFidelity):
        return DistributedDataFidelity(
            ctx,
            data_fidelity=data_fidelity,
            num_operators=num_operators,
            **kwargs,
        )

    elif isinstance(data_fidelity, StackedPhysicsDataFidelity):
        data_fidelity_list_extracted = data_fidelity.data_fidelity_list
        num_operators = len(data_fidelity_list_extracted)

        def data_fidelity_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
            return data_fidelity_list_extracted[idx].to(device)

    elif callable(data_fidelity):
        data_fidelity_factory = data_fidelity
        if num_operators is None or not isinstance(num_operators, int):
            raise ValueError(
                "When using a factory for data_fidelity, you must provide num_operators."
            )
    else:
        data_fidelity_list_extracted = data_fidelity
        num_operators = len(data_fidelity_list_extracted)

        def data_fidelity_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
            return data_fidelity_list_extracted[idx].to(device)

    return DistributedDataFidelity(
        ctx,
        data_fidelity=data_fidelity_factory,
        num_operators=num_operators,
        **kwargs,
    )


def _distribute_replicated_parameters(
    params: torch.nn.Parameter | Sequence[torch.nn.Parameter],
    ctx: DistributedContext,
    *,
    average: bool = True,
):
    r"""
    Attach distributed gradient synchronization to replicated parameters.
    """
    if isinstance(params, torch.nn.Parameter):
        if params.device != ctx.device:
            params = torch.nn.Parameter(
                params.data.to(ctx.device), requires_grad=params.requires_grad
            )
        sync = DistributedReplicatedParameters(ctx, [params], average=average)
        setattr(params, "_deepinv_dist_sync", sync)
        return params

    params_list = list(params)
    for i, p in enumerate(params_list):
        if not isinstance(p, torch.nn.Parameter):
            raise TypeError(f"Expected torch.nn.Parameter, got {type(p)} at index {i}.")
        if p.device != ctx.device:
            params_list[i] = torch.nn.Parameter(
                p.data.to(ctx.device), requires_grad=p.requires_grad
            )

    sync = DistributedReplicatedParameters(ctx, params_list, average=average)
    for p in params_list:
        setattr(p, "_deepinv_dist_sync", sync)
    return params_list


def _distribute_base_optim(
    model: BaseOptim,
    ctx: DistributedContext,
    *,
    average: bool = True,
    tiling_strategy: str | DistributedSignalStrategy | None = None,
    patch_size: int = 256,
    overlap: int = 64,
    tiling_dims: int | tuple | None = None,
    max_batch_size: int | None = None,
    checkpoint_batches: str = "auto",
    checkpoint_use_reentrant: bool = False,
    checkpoint_preserve_rng_state: bool = True,
    dtype: torch.dtype | None = torch.float32,
    gather_strategy: str = "concatenated",
) -> BaseOptim:
    r"""
    In-place distribute a :class:`deepinv.optim.BaseOptim` unfolded model.

    Uses the structured interface of :class:`~deepinv.optim.BaseOptim` directly —
    rather than an opaque module walk — to distribute each component precisely:

    - ``model.data_fidelity`` (``nn.ModuleList``): each
      :class:`~deepinv.optim.DataFidelity` entry is replaced with a
      :class:`DistributedDataFidelity`.
    - ``model.prior`` (``nn.ModuleList``): for every :class:`~deepinv.optim.Prior`
      that holds a ``.denoiser`` attribute (e.g. :class:`~deepinv.optim.PnP`,
      :class:`~deepinv.optim.RED`), the denoiser is replaced with a
      :class:`DistributedProcessing` (overlap-tiled, gradient-synced).
    - ``model.params_algo`` (``nn.ParameterDict``): all ``nn.ParameterList`` values
      (trainable step sizes, regularisation weights, etc.) receive distributed
      gradient-sync hooks via :class:`DistributedReplicatedParameters`.

    :class:`deepinv.physics.Physics` operators are **not** touched: they live outside
    the model and must be distributed separately, then passed to ``model.forward``.

    This function is called automatically by :func:`distribute` when a
    :class:`~deepinv.optim.BaseOptim` model with ``unfold=True`` is passed.
    Users typically only need::

        distribute(model, ctx, patch_size=64, overlap=8)

    :param BaseOptim model: unfolded model (``model.unfold`` must be ``True``),
        modified **in-place**.
    :param DistributedContext ctx: distributed context.
    :param bool average: average gradients across ranks. Default ``True``.
    :param str | DistributedSignalStrategy | None tiling_strategy: tiling strategy
        forwarded to every :class:`DistributedProcessing` created. Default ``None``
        (falls back to ``'overlap_tiling'``).
    :param int patch_size: patch size forwarded to :class:`DistributedProcessing`.
        Default ``256``.
    :param int overlap: overlap forwarded to :class:`DistributedProcessing`.
        Default ``64``.
    :param int | tuple | None tiling_dims: tiling dimensions. Default ``None``.
    :param int | None max_batch_size: maximum patches per batch. Default ``None``.
    :param str checkpoint_batches: activation checkpointing mode for
        patch-batches during backward. Supported values are ``'auto'``,
        ``'always'`` and ``'never'``. Default ``'auto'``.
    :param bool checkpoint_use_reentrant: reentrant mode for activation
        checkpointing. Default ``False``.
    :param bool checkpoint_preserve_rng_state: preserve RNG state during
        activation checkpoint recomputation. Default ``True``.
    :param torch.dtype | None dtype: dtype override. Default ``torch.float32``.
    :param str gather_strategy: gather strategy (forwarded, currently unused for
        processors).
    :return: the same ``model`` object, modified in-place.
    :raises TypeError: if ``model.unfold`` is ``False``.
    """
    if not model.unfold:
        raise TypeError(
            "`_distribute_base_optim` requires a model built with `unfold=True`. "
            "For non-unfolded BaseOptim, distribute physics and denoisers separately."
        )

    model = model.to(ctx.device)

    processor_kwargs = dict(
        dtype=dtype,
        tiling_strategy=tiling_strategy,
        patch_size=patch_size,
        overlap=overlap,
        tiling_dims=tiling_dims,
        max_batch_size=max_batch_size,
        checkpoint_batches=checkpoint_batches,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        checkpoint_preserve_rng_state=checkpoint_preserve_rng_state,
    )

    # 1. Distribute each DataFidelity in the list.
    # Replace the whole ModuleList to avoid nn.ModuleList.__setitem__ type conflicts.
    if model.data_fidelity is not None:
        distributed_dfs = [
            (
                _distribute_data_fidelity(df, ctx)
                if not isinstance(df, DistributedDataFidelity)
                else df
            )
            for df in model.data_fidelity
        ]
        model.data_fidelity = torch.nn.ModuleList(distributed_dfs)

    # 2. Distribute the denoiser inside each Prior (PnP, RED, ScorePrior, ...)
    # DistributedProcessing is now an nn.Module, so normal attribute assignment works.
    if model.prior is not None:
        for prior_module in model.prior:
            if (
                hasattr(prior_module, "denoiser")
                and isinstance(prior_module.denoiser, Denoiser)
                and not isinstance(prior_module.denoiser, DistributedProcessing)
            ):
                prior_module.denoiser = _distribute_processor(
                    prior_module.denoiser, ctx, **processor_kwargs
                )

    # 3. Sync all trainable params_algo entries (step sizes, lambda, g_param, ...)
    # When unfold=True, BaseOptim wraps trainable entries as nn.ParameterList inside
    # an nn.ParameterDict.  Non-trainable entries remain plain Python lists.
    params: list[torch.nn.Parameter] = []
    if isinstance(getattr(model, "params_algo", None), torch.nn.ParameterDict):
        for v in model.params_algo.values():
            if isinstance(v, torch.nn.ParameterList):
                params.extend(list(v))
    if params:
        sync = DistributedReplicatedParameters(ctx, params, average=average)
        setattr(model, "_deepinv_dist_sync", sync)

    return model


def distribute(
    object: (
        StackedPhysics
        | list[Physics]
        | Callable[[int, torch.device, dict | None], Physics]
        | Denoiser
        | Callable[[int, torch.device, dict | None], Denoiser]
        | DataFidelity
        | list[DataFidelity]
        | StackedPhysicsDataFidelity
        | Callable[[int, torch.device, dict | None], DataFidelity]
        | torch.nn.parameter.Parameter
        | Sequence[torch.nn.parameter.Parameter]
        | torch.nn.Module
    ),
    ctx: DistributedContext,
    *,
    num_operators: int | None = None,
    type_object: str | None = "auto",
    dtype: torch.dtype | None = torch.float32,
    gather_strategy: str = "concatenated",
    tiling_strategy: str | DistributedSignalStrategy | None = "overlap_tiling",
    tiling_dims: int | tuple[int, ...] | None = None,
    patch_size: int = 256,
    overlap: int = 64,
    max_batch_size: int | None = None,
    checkpoint_batches: str = "auto",
    checkpoint_use_reentrant: bool = False,
    checkpoint_preserve_rng_state: bool = True,
    **kwargs,
) -> (
    DistributedStackedPhysics
    | DistributedStackedLinearPhysics
    | DistributedProcessing
    | DistributedDataFidelity
    | torch.nn.parameter.Parameter
    | list[torch.nn.parameter.Parameter]
    | BaseOptim
):
    r"""
    Distribute a DeepInverse object across multiple devices.

    This function takes a DeepInverse object and distributes it using the provided DistributedContext.

    The list of supported DeepInverse objects includes:

        - Physics operators: a list of :class:`deepinv.physics.Physics`, :class:`deepinv.physics.StackedPhysics` or :class:`deepinv.physics.StackedLinearPhysics`.
        - Data fidelity terms: a list of :class:`deepinv.optim.DataFidelity` or :class:`deepinv.optim.StackedPhysicsDataFidelity`.
        - Priors/Denoisers: :class:`deepinv.models.Denoiser` or :class:`deepinv.optim.Prior` objects.

    :param StackedPhysics | list[Physics] | Callable | Denoiser | DataFidelity | StackedPhysicsDataFidelity | list[DataFidelity] | torch.nn.Module | torch.nn.parameter.Parameter | Sequence[torch.nn.parameter.Parameter] object:
        DeepInverse object to distribute.
    :param DistributedContext ctx: distributed context manager.
    :param int | None num_operators: number of physics operators when using a factory for physics, otherwise inferred. Default is `None`.
    :param str | None type_object: type of object to distribute. Options are `'physics'`, `'linear_physics'`,
        `'data_fidelity'`, `'denoiser'`, `'module'`, `'parameters'`, or `'auto'` for automatic detection.
        Default is `'auto'`.
        ``'module'`` is restricted to :class:`~deepinv.optim.BaseOptim` models built
        with ``unfold=True`` (including the legacy ``BaseUnfold`` subclass).
        Generic ``torch.nn.Module`` instances are intentionally not supported by this
        API, to avoid ambiguous partial auto-distribution.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results.

        Options are:
            - `'naive'`: Simple object serialization (best for small tensors)
            - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
            - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)

        Default is `'concatenated'`.

    :param str | DistributedSignalStrategy | None tiling_strategy: strategy for tiling the signal (for Denoiser).
        Options are `'basic'`, `'overlap_tiling'`, or a custom strategy instance. Default is `'overlap_tiling'`.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile over (for Denoiser).

        Can be one of the following:
            - If ``None`` (default), tiles the last N-2 dimensions of your input tensor.
            - If an int ``N``, only tiles over the specified dimension.
            - If a tuple, specifies exact dimensions to tile.

        Examples:
            - For ``(B, C, H, W)`` image: ``tiling_dims=(2, 3)`` tiles over H and W.
            - For ``(B, C, D, H, W)`` volume: ``tiling_dims=(2, 3, 4)`` tiles over D, H, W.
            - For ``(B, C, H, W)`` image: ``tiling_dims=2`` tiles only over H dimension.
            - For ``(B, C, D, H, W)`` volume: ``tiling_dims=None`` tiles over D, H, W dimensions.

    :param int patch_size: size of patches for tiling strategies (for Denoiser).
        Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is `256`.
    :param int overlap: receptive field size for overlap in tiling strategies (for Denoiser).
        Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is `64`.
    :param int | None max_batch_size: maximum number of patches to process in a single batch (for Denoiser). If `None`, all patches are batched together. Set to `1` for sequential processing. Default is `None`.
    :param str checkpoint_batches: activation checkpointing mode for
        patch-batches during backward (for Denoiser).
        Supported values are ``'auto'``, ``'always'`` and ``'never'``.
        Default is ``'auto'``.
    :param bool checkpoint_use_reentrant: reentrant mode for activation
        checkpointing in denoiser processing. Default is ``False``.
    :param bool checkpoint_preserve_rng_state: preserve RNG state during
        checkpoint recomputation in denoiser processing. Default is ``True``.
    :param kwargs: additional keyword arguments for specific distributed classes.

    :return: Distributed version of the input object.

    |sep|

    :Examples:

        Distribute a Physics object:

        >>> from deepinv.physics import Blur, StackedLinearPhysics
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     physics = StackedLinearPhysics([Blur(kernel_size=5), Blur(kernel_size=9)])
        ...     dphysics = distribute(physics, ctx)

        Distribute a DataFidelity object:

        >>> from deepinv.optim.data_fidelity import L2
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     data_fidelity = L2()
        ...     ddata_fidelity = distribute(data_fidelity, ctx)

        Distribute a Prior object:

        >>> from deepinv.models import DnCNN
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     denoiser = DnCNN()
        ...     ddenoiser = distribute(denoiser, ctx)

        Distribute a full unfolded PGD model in one call:

        >>> from deepinv.models import DnCNN
        >>> from deepinv.optim import PGD
        >>> from deepinv.optim.data_fidelity import L2
        >>> from deepinv.optim.prior import PnP
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     model = PGD(
        ...         data_fidelity=L2(),
        ...         prior=PnP(DnCNN(in_channels=1, out_channels=1)),
        ...         stepsize=[0.9, 0.8],
        ...         max_iter=2,
        ...         unfold=True,
        ...     )
        ...     distribute(model, ctx, patch_size=64, overlap=8)
    """
    # Check object type and distribute accordingly
    if type_object == "auto":
        if isinstance(object, (StackedPhysics, StackedLinearPhysics)) or (
            isinstance(object, list)
            and len(object) > 0
            and isinstance(object[0], Physics)
        ):
            type_object = (
                "linear_physics"
                if isinstance(object, (StackedLinearPhysics, list))
                and (
                    not isinstance(object, list) or isinstance(object[0], LinearPhysics)
                )
                else "physics"
            )
        elif isinstance(object, (DataFidelity, StackedPhysicsDataFidelity)) or (
            isinstance(object, list)
            and len(object) > 0
            and isinstance(object[0], DataFidelity)
        ):
            type_object = "data_fidelity"
        elif isinstance(object, Denoiser):
            type_object = "denoiser"
        elif isinstance(object, torch.nn.Parameter):
            type_object = "parameters"
        elif isinstance(object, BaseOptim):
            type_object = "module"
        elif isinstance(object, torch.nn.Module):
            raise ValueError(
                "Cannot auto-detect generic torch.nn.Module for distribute(). "
                "Only BaseOptim models with unfold=True are supported via "
                "type_object='module'."
            )
        elif (
            isinstance(object, (list, tuple))
            and len(object) > 0
            and isinstance(object[0], torch.nn.Parameter)
        ):
            type_object = "parameters"
        elif callable(object):
            raise ValueError(
                "For callable objects, you must specify type_object parameter"
            )
        else:
            raise ValueError(f"Cannot auto-detect type for object: {type(object)}")

    if type_object in ["physics", "linear_physics"]:
        return _distribute_physics(
            object,
            ctx,
            dtype=dtype,
            num_operators=num_operators,
            type_object=type_object,
            gather_strategy=gather_strategy,
            **kwargs,
        )
    elif type_object == "denoiser":
        return _distribute_processor(
            object,
            ctx,
            dtype=dtype,
            gather_strategy=gather_strategy,
            tiling_strategy=tiling_strategy,
            patch_size=patch_size,
            overlap=overlap,
            tiling_dims=tiling_dims,
            max_batch_size=max_batch_size,
            checkpoint_batches=checkpoint_batches,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
            checkpoint_preserve_rng_state=checkpoint_preserve_rng_state,
            **kwargs,
        )
    elif type_object == "data_fidelity":
        return _distribute_data_fidelity(
            object,
            ctx,
            num_operators=num_operators,
            **kwargs,
        )
    elif type_object == "parameters":
        return _distribute_replicated_parameters(object, ctx, **kwargs)
    elif type_object == "module":
        if not isinstance(object, BaseOptim):
            raise TypeError(
                "type_object='module' only supports deepinv BaseOptim models with "
                "unfold=True. Generic torch.nn.Module is not supported."
            )
        if not object.unfold:
            raise TypeError(
                "distribute() received a BaseOptim model with unfold=False. Only "
                "unfolded (trainable) models can be distributed via this path. "
                "Distribute physics and denoisers separately instead."
            )
        average = kwargs.pop("average", True)
        if kwargs:
            raise TypeError(
                "Unsupported keyword arguments for type_object='module': "
                f"{sorted(kwargs.keys())}"
            )
        return _distribute_base_optim(
            object,
            ctx,
            average=average,
            dtype=dtype,
            gather_strategy=gather_strategy,
            tiling_strategy=tiling_strategy,
            patch_size=patch_size,
            overlap=overlap,
            tiling_dims=tiling_dims,
            max_batch_size=max_batch_size,
            checkpoint_batches=checkpoint_batches,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
            checkpoint_preserve_rng_state=checkpoint_preserve_rng_state,
        )
    else:
        raise ValueError(f"Unsupported type_object: {type_object}")
