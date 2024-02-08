"""
Stacking and concatenating forward operators.
====================================================================================================

In this example, we show how to stack and concatenate forward operators to create new operators.
In particular, we create a pan-sharpening operator by stacking a downsampling and a color-to-grayscale
operators.

"""

import deepinv as dinv
import torch

# %%
# Stacking forward operators.
# ------------------------------------
# We can define a new forward operator by stacking or concatenating existing operators. Mathematically, this is
# equivalent to obtaining
#
# .. math::
#           \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} x
#
# Here we show how to stack two operators, one that downsamples a color image and another that converts the color image
# grayscale. This is equivalent to the :class:`deepinv.physics.Pansharpen` operator.

img_size = (3, 64, 64)
factor = 2
filter = "gaussian"
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

physics1 = dinv.physics.Downsampling(
    img_size=img_size, factor=factor, filter=filter, device=device
)
physics2 = dinv.physics.Decolorize()
physics_stacked = physics2 + physics1

# %%
# Generate toy image
# --------------------------------------------------------------------
#
# This example uses a toy image with 3 color channels.
#
# The measurements of a stacked operator are :class:`deepinv.utils.TensorList` objects, which are lists of tensors
# that can be added, multiplied, etc. to other :class:`deepinv.utils.TensorList` objects. It is also possible to
# generate random or zero-filled :class:`deepinv.utils.TensorList` objects in one line of code (similarly to standard
# :class:`torch.Tensor`).
#

x = torch.zeros((1,) + img_size, device=device)
x[:, 0, 16:48, 16:48] = 0.7

y = physics_stacked(x)
xlin = physics_stacked.A_dagger(y)  # compute the linear pseudo-inverse

dinv.utils.plot(
    [x, y[0], y[1], xlin],
    titles=["image", "high-res grayscale", "low-res color", "linear rec."],
)


# %%
# Verifying the stacked operator
# --------------------------------------------
#
# If the operator is linear, it is recommended to verify that the transpose well-defined using
# :meth:`deepinv.physics.LinearPhysics.adjointness_test()`,
# and that it has a unit norm using :meth:`deepinv.physics.LinearPhysics.compute_norm()`.

print(f"The stacked operator has norm={physics_stacked.compute_norm(x):.2f}")

if physics_stacked.adjointness_test(x) < 1e-5:
    print("The stacked operator has a well defined transpose")


# %%
# Concatenating forward operators.
# ----------------------------------------------------------------------------------------
#
# It is also possible to concatenate operators using the ``*`` operator between two forward operators.
# Here we create a new operator that first downsamples the image, and then converts it to grayscale.

physics_concat = physics2 * physics1

y = physics_concat(x)
xlin = physics_concat.A_dagger(y)  # compute the linear pseudo-inverse

dinv.utils.plot([x, y, xlin], titles=["image", "measurement", "linear rec."])


# %%
# Verifying the concatenated operator
# --------------------------------------------


print(f"The concatenated operator has norm={physics_concat.compute_norm(x):.2f}")

if physics_concat.adjointness_test(x) < 1e-5:
    print("The concatenated operator has a well defined transpose")
