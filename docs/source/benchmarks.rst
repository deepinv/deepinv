.. _benchmarks:

Benchmarks
=================

This section provides benchmark results for various datasets and physics models.

.. note::

    Benchmarks are defined in the https://github.com/deepinv/benchmarks repository.
    To contribute a new benchmark or add your solver to an existing benchmark, please refer to this repository.


List of benchmarks
^^^^^^^^^^^^^^^^^^^

.. list-table::
    :class: sortable-table
    :header-rows: 1

    * - Benchmark
      - Dataset
      - Physics
      - Noise Model
%%BENCHMARK_ROWS%%

.. toctree::
   :maxdepth: 2
   :hidden:

%%PHYSICS_TOCTREE%%

Testing your method on benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To evaluate your own reconstruction methods on these benchmarks, install `deepinv_bench`:

.. code-block:: bash

    pip install git+https://github.com/deepinv/benchmarks.git#egg=deepinv_bench


If you have already installed benchmarks, you can update it with:

.. code-block:: bash

    pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/benchmarks.git#egg=deepinv_bench


and then run on python:

.. code-block:: python

    from deepinv_bench import run_benchmark
    import deepinv as dinv
    my_solver = ... # replace with your reconstruction method
    results = run_benchmark(my_solver, "benchmark_name")

where  `benchmark_name` is the name of the benchmark and `my_solver` is your reconstruction method which receives `(y, physics, **kwargs)` where

- `y` is a :class:`torch.Tensor` containing the measurements,
- `physics` is the :class:`forward operator <deepinv.physics.Physics>`

and outputs a :class:`torch.Tensor` containing the reconstructed image.
