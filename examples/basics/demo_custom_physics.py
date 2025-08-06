r"""
Bring your own physics
====================================================================================================

This examples shows you how to use DeepInverse with your own physics.

While DeepInverse offers a :ref:`large number of forward operators <physics>`,
you can also bring your own forward operator for your specific imaging problem.

DeepInverse's modular :ref:`physics framework <physics_intro>` makes this easy by letting you inherit useful methods from the most appropriate
physics base class:

* :class:`deepinv.physics.Physics` for non-linear operators;
* :class:`deepinv.physics.LinearPhysics` for linear operators;
* :class:`deepinv.physics.DecomposablePhysics` for linear operators with a closed-form singular value decomposition.

.. seealso::

    Often your physics can be modelled without much work. You could:

    * Use an existing physics but with custom `params`, e.g. blur with a custom kernel, or MRI with a custom sampling pattern. See :ref:`parameter dependent operators <parameter-dependent-operators>`.
    * Inherit from an existing physics but override or wrap a particular method e.g. :class:`deepinv.physics.MRI` :math:`\rightarrow` :class:`deepinv.physics.DynamicMRI`
    * Define a new operator by :ref:`combining existing operators <physics_combining>`.

In this example we will demonstrate creating a simple forward operator from scratch that converts RGB images to grayscale images.
We also show you how to exploit the singular value decomposition of the operator to speed up the evaluation of
the pseudo-inverse and proximal operators.
"""

import deepinv as dinv
import torch

# %%
# Creating a custom forward operator.
# ----------------------------------------------------------------------------------------
# Defining a new linear operator only requires a forward function :math:`\forw{\cdot}` and its adjoint operation :math:`A^\top(\cdot)`,
# inheriting the remaining structure of the :class:`deepinv.physics.LinearPhysics` class.
#
# Once the operator is defined, we can use any of the functions in the :class:`deepinv.physics.Physics` class and
# :class:`deepinv.physics.LinearPhysics` class, such as computing the norm of the operator, testing the adjointness,
# computing the proximal operator, etc.
#
# .. tip::
#     You can also define the adjoint automatically using autograd with :class:`deepinv.physics.adjoint_function`.
#     Note however that defining a closed form adjoint is generally more efficient.
#
# .. note::
#     To make the new physics compatible with other torch functionalities, all physics parameters (i.e. attributes of type :class:`torch.Tensor`) should be registered as `module buffers <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer>`_ by using `self.register_buffer(param_name, param_tensor)`. This ensures methods like `.to(), .cuda()` work properly, allowing one to train a model using Distributed Data Parallel.
#
# .. tip::
#     Inherit from :ref:`mixin <mixin>` classes to provide specialised methods for your physics.


class Decolorize(dinv.physics.LinearPhysics):
    r"""
    Converts RGB images to grayscale.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        coefficients = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
        self.register_buffer("coefficients", coefficients)

    def A(self, x: torch.Tensor, theta: torch.Tensor = None) -> torch.Tensor:
        """Forward operator.

        :param torch.Tensor x: input image with 3 colour (RGB) channels, i.e. [*,3,*,*]
        :param torch.Tensor theta: an optional parameter that is not used here
        :return: torch.Tensor grayscale measurements
        """
        y = x * self.coefficients[None, :, None, None]
        return torch.sum(y, dim=1, keepdim=True)

    def A_adjoint(self, y: torch.Tensor, theta: torch.Tensor = None) -> torch.Tensor:
        """Adjoint operator.

        :param torch.Tensor y: input grayscale measurements
        :param torch.Tensor theta: an optional parameter that is not used here
        :return: torch.Tensor adjoint reconstruction
        """
        return y * self.coefficients[None, :, None, None]


# %%
# Test physics
# --------------------------------------------------------------------
#
# We test our physics on a toy image with 3 color channels.
# We add a Gaussian noise model to the linear physics.
#
# We simulate measurements using the forward operator :math:`y=\forw{x}+\epsilon`.
# We then leverage the linear physics base class to automatically compute the linear pseudo-inverse :math:`A^\dagger y`.
#


x = torch.zeros(1, 3, 96, 128)
x[:, 0, :32, :] = 1
x[:, 1, 32:64, :] = 1
x[:, 2, 64:, :] = 1

physics = Decolorize(noise_model=dinv.physics.GaussianNoise(sigma=0.1))

y = physics(x)

dinv.utils.plot({"x": x, "y": y, "Linear pseudo-inverse": physics.A_dagger(y)})


# %%
# Verifying our linear operator
# --------------------------------------------
#
# If the operator is indeed linear, you should verify that the transpose is well-defined using
# :func:`deepinv.physics.LinearPhysics.adjointness_test`.
#
# It is often useful for reconstruction algorithms that the physics has unit norm, which you can verify using :func:`deepinv.physics.LinearPhysics.compute_norm`.
# We see that this physics fails this.

if physics.adjointness_test(x) < 1e-5:
    print("The linear operator has a well defined transpose")

print(f"The linear operator has norm={physics.compute_norm(x):.2f}")

# %%
# Creating a decomposable forward operator.
# --------------------------------------------
# If the forward operator has a closed form singular value decomposition (SVD),
# you should instead implement the operator using the :class:`deepinv.physics.DecomposablePhysics` base class.
#
# The operator :math:`A` in this example has a known closed-form SVD:
#
# .. math::
#   A = U\text{diag}(s)V^{\top}
#
# where :math:`\text{diag}(s)` is the `mask` of singular values.


class DecolorizeSVD(dinv.physics.DecomposablePhysics):
    r"""
    Converts RGB images to grayscale.

    We use unnormalised coefficients and a singular value of 0.447.
    """

    def __init__(self, **kwargs):
        super().__init__(mask=0.447, **kwargs)
        coefficients = torch.tensor([0.6687, 1.3132, 0.2550], dtype=torch.float32)
        self.register_buffer("coefficients", coefficients)

    def V_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.coefficients[None, :, None, None]
        return torch.sum(y, dim=1, keepdim=True)

    def V(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.coefficients[None, :, None, None]


physics2 = DecolorizeSVD(noise_model=dinv.physics.GaussianNoise(sigma=0.1))

y2 = physics2(x)

dinv.utils.plot({"x": x, "y": y2, "Linear pseudo-inverse": physics2.A_dagger(y2)})

if physics.adjointness_test(x) < 1e-5:
    print("The decomposable operator has a well defined transpose")

print(f"The decomposable operator has norm={physics.compute_norm(x):.2f}")

# %%
# Benefits of using a decomposable forward operator.
# ----------------------------------------------------------------------------------------
#
# The main benefit of using a decomposable forward operator is that it provides closed form solutions for the
# proximal operator and the linear pseudo-inverse. Moreover, some algorithms, such as :class:`deepinv.sampling.DDRM`
# require the forward operator to be decomposable.

import time

start = time.time()
for i in range(10):
    xlin = physics.A_dagger(y)
    xprox = physics.prox_l2(x, y, 0.1)

end = time.time()
print(f"Elapsed time for LinearPhysics: {end - start:.2f} seconds")

start = time.time()
for i in range(10):
    xlin2 = physics2.A_dagger(y)
    xprox2 = physics2.prox_l2(x, y2, 0.1)

end = time.time()
print(f"Elapsed time for DecomposablePhysics: {end - start:.2e} seconds")

# %%
