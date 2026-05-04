.. _distributed-training:

Distributed Training
====================

The distributed framework can be used during training. This is useful when a
single inverse problem is too large for one GPU, for example because reconstruction
model processes large images or volumes.

The API is the same as for :ref:`distributed reconstruction <distributed>`:

1. :class:`deepinv.distributed.DistributedContext` manages the processes and devices
2. :func:`deepinv.distributed.distribute` converts deepinv objects to distributed versions

.. note::

    This is different from data-parallel training. In data parallelism,
    each GPU sees different images. Here, each rank works on different parts of
    the same image or volume. Therefore, all ranks should usually iterate
    over the same minibatches.

.. warning::

    This module is in beta and may undergo significant changes in future releases.
    Some features are experimental and only supported for specific use cases.
    Please report any issues you encounter on our `GitHub repository <https://github.com/deepinv/deepinv>`_.


Quick Start
-----------

The typical workflow is to distribute the physics and then distribute the
trainable unfolded model:

.. code-block:: python

    import torch
    import deepinv as dinv
    from deepinv.distributed import DistributedContext, distribute
    from deepinv.optim import DRS
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.prior import PnP

    with DistributedContext(seed=0, seed_offset=False) as ctx:
        physics = distribute(stacked_physics, ctx)

        model = DRS(
            data_fidelity=L2(),
            prior=PnP(denoiser),
            max_iter=5,
            unfold=True,
            trainable_params=["stepsize", "sigma_denoiser"],
        )
        model = distribute(model, ctx, patch_size=256, overlap=64)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = dinv.Trainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            device=ctx.device,
            verbose=(ctx.rank == 0),
            show_progress_bar=(ctx.rank == 0),
        )
        trainer.train()

The rest of your training code can stay close to a standard
:class:`deepinv.Trainer` workflow. The distributed objects handle the
communication needed by the forward pass and by backpropagation.

.. note::

    See :ref:`sphx_glr_auto_examples_distributed_demo_unrolled_distributed.py`
    for a complete example training an unfolded DRS model.


When to Use It
--------------

Distributed training is useful when you want to train a model but one process
cannot hold the full computation or would be too slow. Common scenarios include:

**Many physics operators**: if your measurements come from several operators
:math:`A_i`, the framework can split them across ranks. Each rank applies its
local operators and the results are combined automatically.

**Large images or volumes**: if your denoiser or prior is too expensive to run
on the full signal, the framework can split the signal into overlapping tiles,
process the tiles on different ranks, and blend them back together.

**Unfolded algorithms**: unfolded models such as :class:`deepinv.optim.DRS` can
be distributed in one call when they are created with ``unfold=True``. The
framework distributes their data-fidelity terms, tiled denoisers, and trainable
algorithm parameters.


Simple Training Pattern
-----------------------

**Step 1: Use a synchronized context**

.. code-block:: python

    with DistributedContext(seed=seed, seed_offset=False) as ctx:
        ...

Using ``seed_offset=False`` keeps random streams aligned across ranks. This is
important because ranks should usually consume the same data in the same order.

**Step 2: Distribute the physics**

.. code-block:: python

    physics = distribute(stacked_physics, ctx)

For a stacked physics :math:`A = [A_1, \ldots, A_N]`, each rank owns a subset of
the operators. The global forward, adjoint, and data-fidelity computations are
assembled from these local contributions.

**Step 3: Distribute the unfolded model**

.. code-block:: python

    model = distribute(
        model,
        ctx,
        patch_size=256,
        overlap=64,
        max_batch_size=1,
    )

For unfolded models, :func:`deepinv.distributed.distribute` replaces the
denoiser inside compatible priors by a tiled distributed processor and adds
gradient synchronization for trainable parameters such as stepsizes.

**Step 4: Train as usual**

.. code-block:: python

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        device=ctx.device,
        verbose=(ctx.rank == 0),
        show_progress_bar=(ctx.rank == 0),
    )
    trainer.train()

All ranks must enter the training loop. For printing, plotting, and progress
bars, it is usually best to do the visible work only on rank 0.


Data Loading
------------

Because the framework distributes one inverse problem across ranks, the
dataloader should usually return the same batch on every rank.

In practice:

- Do not use :class:`torch.utils.data.distributed.DistributedSampler` for this use case.
- Use the same dataset, batch size, and shuffle seed on all ranks.
- Move both images and measurements to ``ctx.device``.
- If measurements are stored as a list or :class:`deepinv.utils.TensorList`, move each tensor to the device.

This is different from data parallelism, where each process receives
different samples and gradients are synchronized only between model replicas.


How Backward Works
------------------

You normally do not need to write custom backward code. The distributed objects
are built so that PyTorch autograd can follow the full computation.

**Through distributed physics**: each rank applies its local operators
:math:`A_i`. During backward, the gradient with respect to the shared input is
computed from local contributions and synchronized across ranks. For linear
physics, adjoints and vector-Jacobian products are reduced so that the model sees
the gradient of the full stacked problem, not only the local operators.

**Through data fidelity terms**: losses such as :class:`deepinv.optim.data_fidelity.L2`
are evaluated locally on each rank's measurements, then reduced. Their gradients
are propagated back through the corresponding local physics operators and
combined automatically.

**Through tiled denoisers and priors**: the input is split into overlapping
patches. Each rank processes a group of patches with the same denoiser weights.
The processed patches are blended back into the full image, and backward sends
gradients through the same patch operations. Gradients of replicated trainable
weights are synchronized so that all ranks keep the same model parameters after
the optimizer step.


Checkpointing
-------------

Distributed tiled processing can use activation checkpointing to reduce memory
during training. Instead of storing every intermediate activation for every
patch batch, PyTorch recomputes some patch forwards during backward.

The default setting is usually enough:

.. code-block:: python

    model = distribute(model, ctx, checkpoint_batches="auto")

The available modes are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Description
   * - ``"auto"``
     - Checkpoint patch batches only when gradients are enabled and there are multiple local batches.
   * - ``"always"``
     - Checkpoint patch batches whenever gradients are enabled.
   * - ``"never"``
     - Disable checkpointing.

Set ``max_batch_size=1`` to process local patches sequentially when memory is
tight. This is slower, but often allows training on larger images or volumes.


Running Multi-Process
---------------------

Use ``torchrun`` to launch one process per GPU:

.. code-block:: bash

    torchrun --nproc_per_node=4 my_training_script.py

The same script also works in single-process mode:

.. code-block:: bash

    python my_training_script.py

:class:`deepinv.distributed.DistributedContext` detects whether distributed
environment variables are present and selects the device for each rank.


Troubleshooting
---------------

**Training hangs**

- Make sure every rank enters the same training loop.
- Avoid branching around forward or backward calls unless every rank follows the same branch.

**Gradients differ across ranks**

- Use ``seed_offset=False`` when all ranks should consume the same random operations.
- Make sure the same minibatch is loaded on every rank.
- Do not use a data-parallel sampler unless you intentionally changed the training strategy.

**Out of memory errors**

- Reduce ``patch_size`` and ``overlap`` to reduce the patch memory footprint.
- Set ``max_batch_size=1`` to process fewer patches at once.
- Use ``checkpoint_batches="always"`` for the distributed denoiser or unfolded model.


See Also
--------

- **Complete example**: :ref:`sphx_glr_auto_examples_distributed_demo_unrolled_distributed.py`
- **Distributed reconstruction guide**: :ref:`distributed`
- **Trainer guide**: :ref:`trainer`
- **Multi-GPU training guide**: :ref:`multigpu`
- **API Reference**: :doc:`/api/deepinv.distributed`
