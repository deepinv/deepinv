.. _physics_defining:

Defining new operators
--------------------------------

Defining a new forward operator is relatively simple. You need to create a new class that inherits from the right
physics class, that is :meth:`deepinv.physics.Physics` for non-linear operators,
:meth:`deepinv.physics.LinearPhysics` for linear operators and :meth:`deepinv.physics.DecomposablePhysics`
for linear operators with a closed-form singular value decomposition. The only requirement is to define
a :class:`deepinv.physics.Physics.A` method that computes the forward operator. See the
example :ref:`sphx_glr_auto_examples_basics_demo_physics.py` for more details.

You can also inherit from mixin classes to provide useful methods for your physics:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.physics.TimeMixin

Defining a new linear operator requires the definition of :class:`deepinv.physics.LinearPhysics.A_adjoint`,
you can define the adjoint automatically using autograd with

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.physics.adjoint_function

Note however that coding a closed form adjoint is generally more efficient.

