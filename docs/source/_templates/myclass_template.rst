{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :no-undoc-members:
   :special-members: __mul__, __add__, __div__, __neg__, __sub__, __truediv__

.. _sphx_glr_backref_{{module}}.{{objname}}:

.. minigallery:: {{module}}.{{objname}}
    :add-heading: Examples using ``{{objname}}``:

{% if objname in benchmark_mapping %}
Find in benchmarks
------------------

- :ref:`{{ benchmark_mapping[objname] }}`
{% endif %}