deepinv.unfolded
===================

This module provides networks architectures based on unfolding optimization algorithms.
Please refer to :ref:`user guide <unfolded>` for more details.

Unfolded
--------
.. userguide:: unfolded

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.unfolded.unfolded_builder


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.unfolded.BaseUnfold


Deep Equilibrium
----------------
.. userguide:: deep-equilibrium

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

    deepinv.unfolded.DEQ_builder

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.unfolded.BaseDEQ


Custom Unfolded Blocks
----------------------
.. userguide:: custom-unfolded-blocks

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.PDNet_PrimalBlock
   deepinv.models.PDNet_DualBlock