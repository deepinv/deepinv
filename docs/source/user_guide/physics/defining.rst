.. _physics_defining:

Defining New Operators
----------------------

Defining a new forward operator is relatively simple. You need to create a new class that inherits from the right
physics class, that is :meth:`deepinv.physics.Physics` for non-linear operators,
:meth:`deepinv.physics.LinearPhysics` for linear operators and :meth:`deepinv.physics.DecomposablePhysics`
for linear operators with a closed-form singular value decomposition. The only requirement is to define
a :class:`deepinv.physics.Physics.A` method that computes the forward operator.

**See the example** :ref:`sphx_glr_auto_examples_basics_demo_physics.py` **for more details**.


.. tip::

    Defining a new linear operator requires the definition of :class:`deepinv.physics.LinearPhysics.A_adjoint`,
    you can define the adjoint automatically using autograd with :class:`deepinv.physics.adjoint_function`.
    Note however that coding a closed form adjoint is generally more efficient.

.. _mixin:

**TODO**: move this to a new section of "How to contribute", not just specific to physics.

You can also inherit from mixin classes to provide specific useful methods,
for your physics, models, losses etc. This is encouraged to maintain a consistent API,
and to prevent rewriting code.

.. list-table:: Mixins
   :header-rows: 1

   * - **Mixin**
     - **Description**

   * - :class:`deepinv.utils.mixin.TimeMixin`
     - Easily add temporal capabilities for dynamic physics, models etc.

   * - :class:`deepinv.utils.mixin.RandomMixin`
     - Enforce reproducibility in classes that require randomness.

   * - :class:`deepinv.utils.mixin.MRIMixin`
     - Provides helper methods for FFT and mask checking.

