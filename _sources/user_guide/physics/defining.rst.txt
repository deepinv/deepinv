.. _physics_defining:

Defining New Operators
----------------------

Defining a new forward operator is relatively simple. You need to create a new class that inherits from the right
physics class, that is :class:`deepinv.physics.Physics` for non-linear operators,
:class:`deepinv.physics.LinearPhysics` for linear operators and :class:`deepinv.physics.DecomposablePhysics`
for linear operators with a closed-form singular value decomposition. The only requirement is to define
a :class:`deepinv.physics.Physics.A` method that computes the forward operator.

**See the example** :ref:`sphx_glr_auto_examples_basics_demo_physics.py` **for more details**.


.. tip::

    Defining a new linear operator requires the definition of :class:`deepinv.physics.LinearPhysics.A_adjoint`,
    you can define the adjoint automatically using autograd with :class:`deepinv.physics.adjoint_function`.
    Note however that coding a closed form adjoint is generally more efficient.

.. tip::

    You can also inherit from mixin classes such as :class:`deepinv.physics.TimeMixin` and :class:`deepinv.physics.MRIMixin` to provide useful methods for your physics.

.. tip::

    You can also define a new operator by :ref:`combining existing operators <physics_combining>`.