.. |plusminus| unicode:: U+00B1 .. plus-minus sign

.. _%%LABEL%%:

%%TITLE%%
%%UNDERLINE%%

- *Dataset*: :sclass:`deepinv.datasets.%%DATASET%%`

- *Physics*: :sclass:`deepinv.physics.%%PHYSICS%%`

- *Noise model*: :sclass:`deepinv.physics.%%NOISE%%`

%%DATASET_PARAMS%%

Run this benchmark with

.. code-block:: python

   from deepinv_bench import run_benchmark
   my_solver = lambda y, physics: ...  # your solver here
   results = run_benchmark(my_solver, "%%LABEL%%")

.. list-table::
   :class: sortable-table
   :header-rows: 1

%%TABLE_ROWS%%
