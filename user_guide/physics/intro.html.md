<a id="physics-intro"></a>

# Introduction

This module contains a large collection of forward operators appearing in imaging applications.
The acquisition models are of the form

$$
y = \noise{\forw{x}},
$$

where $x\in\xset$ is an image, $y\in\yset$ are the measurements, $A:\xset\mapsto \yset$ is a
deterministic (linear or non-linear) operator capturing the physics of the acquisition and
$N:\yset\mapsto \yset$ is a mapping which characterizes the noise affecting the measurements.

All forward operators inherit the structure of the [`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics) class.

They are [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) which can be called with the `forward` method.

```pycon
>>> import torch
>>> import deepinv as dinv
>>> # load an inpainting operator that masks 50% of the pixels and adds Gaussian noise
>>> physics = dinv.physics.Inpainting(mask=.5, img_size=(1, 28, 28),
...                    noise_model=dinv.physics.GaussianNoise(sigma=.05))
>>> x = torch.rand(1, 1, 28, 28) # create a random image
>>> y = physics(x) # compute noisy measurements
>>> y2 = physics.A(x) # compute the A operator (no noise)
```

## Linear operators

Linear operators $A:\xset\mapsto \yset$ inherit the structure of the [`deepinv.physics.LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics) class.
They have important specific properties such as the existence of an adjoint $A^{\top}:\yset\mapsto \xset$.
Linear operators with a closed-form singular value decomposition are defined via [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics),
which enables the efficient computation of their pseudo-inverse and regularized inverse.
Composition and linear combinations of linear operators is still a linear operator.

```pycon
>>> import torch
>>> import deepinv as dinv
>>> # load a CS operator with 300 measurements, acting on 28 x 28 grayscale images.
>>> physics = dinv.physics.CompressedSensing(m=300, img_size=(1, 28, 28))
>>> x = torch.rand(1, 1, 28, 28) # create a random image
>>> y = physics(x) # compute noisy measurements
>>> y2 = physics.A(x) # compute the linear operator (no noise)
>>> x_adj = physics.A_adjoint(y) # compute the adjoint operator
>>> x_dagger = physics.A_dagger(y) # compute the pseudo-inverse operator
>>> x_prox = physics.prox_l2(x, y, .1) # compute a regularized inverse
```

#### TIP
Linear operators come with useful methods for approximating the [`operator norm`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.compute_norm)
$\|A\|$ and the [`condition number`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics.condition_number) $\kappa(A)$.
These values can be useful to set optimization hyperparameters, and understand the difficulty of the inverse problem.

More details can be found in the documentation of each class.

<a id="parameter-dependent-operators"></a>

## Parameter-dependent operators

Many (linear or non-linear) operators depend on (optional) `params` $\theta$ that describe the imaging system, i.e.
$y = \noise{\forw{x, \theta}}$ where the `forward` method can be called with a dictionary of `params` as an extra input.
The explicit dependency on $\theta$ is often useful for blind inverse problems, model identification,
imaging system optimization, etc. The following example shows how operators and their parameter can be instantiated and called as:

```pycon
>>> import torch
>>> from deepinv.physics import Blur
>>> x = torch.rand((1, 1, 16, 16))
>>> theta = torch.ones((1, 1, 2, 2)) / 4 # a basic 2x2 averaging filter
>>> # default usage
>>> physics = Blur(filter=theta) # we instantiate a blur operator with its convolution filter
>>> y = physics(x)
>>> theta2 = torch.randn((1, 1, 2, 2)) # a random 2x2 filter
>>> physics.update(filter=theta2)
>>> y2 = physics(x)
>>>
>>> # A second possibility
>>> physics = Blur() # a blur operator without convolution filter
>>> y = physics(x, filter=theta) # we define the blur by specifying its filter
>>> y = physics(x) # now, the filter is well-defined and this line does the same as above
>>>
>>> # The same can be done by passing in a dictionary including 'filter' as a key
>>> physics = Blur() # a blur operator without convolution filter
>>> params = {'filter': theta, 'dummy': None}
>>> y = physics(x, **params) # # we define the blur by passing in the dictionary
```

One can also differentiate the parameter as:

```pycon
>>> import torch
>>> from deepinv.physics import Blur
>>> x = torch.rand((1, 1, 16, 16))
>>> theta = torch.ones((1, 1, 2, 2)) / 4 # a basic 2x2 averaging filter
>>> physics = Blur(filter=theta, padding='circular') # we instantiate a blur operator with its convolution filter
>>> y = physics(x)
>>> theta_2 = torch.ones((1, 1, 3, 3)) / 9 # we'll compute the gradient of the physics with the new filter theta_2 comparing to the measurement with theta
>>> with torch.enable_grad():
...     loss = torch.sum(y - physics(x, filter=theta_2.requires_grad_(True))) / y.numel()
...     loss.backward()
>>> print(theta_2.grad.shape)
torch.Size([1, 1, 3, 3])
```

and optimize the parameter $\theta$, as shown in this example: [Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.html.md#sphx-glr-auto-examples-blind-inverse-problems-demo-optimizing-physics-parameter-py)

<a id="physics-generators"></a>

## Physics Generators

We provide some parameters generation methods to sample random parameters’ $\theta$.
Physics generators inherit from the [`deepinv.physics.generator.PhysicsGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator) class:

```pycon
>>> import torch
>>> import deepinv as dinv
>>>
>>> x = torch.rand((1, 1, 8, 8))
>>> physics = dinv.physics.Blur(filter=dinv.physics.functional.blur.gaussian_blur(sigma=(0.2, 0.2)))
>>> y = physics(x) # compute with Gaussian blur
>>> generator = dinv.physics.generator.MotionBlurGenerator(psf_size=(3, 3))
>>> params = generator.step(x.size(0)) # params = {'filter': torch.tensor(...)}
>>> y1 = physics(x, **params) # compute with motion blur
>>> assert not torch.allclose(y, y1) # different blurs, different outputs
>>> y2 = physics(x) # motion kernel is stored in the physics object as default kernel
>>> assert torch.allclose(y1, y2) # same blur, same output
```

If we want to generate both a new physics and noise parameters,
it is possible to sum generators as follows:

```pycon
>>> mask_generator = dinv.physics.generator.SigmaGenerator() \
...    + dinv.physics.generator.RandomMaskGenerator((32, 32))
>>> params = mask_generator.step(batch_size=4)
>>> print(sorted(params.keys()))
['mask', 'sigma']
```

#### TIP
It is also possible to mix generators of physics parameters through the
[`deepinv.physics.generator.GeneratorMixture`](https://deepinv.org/api/stubs/deepinv.physics.generator.GeneratorMixture.html.md#deepinv.physics.generator.GeneratorMixture) class.

<a id="physics-combining"></a>

## Combining Physics

It is possible to stack and compose multiple physics operators into a single operator.

Stacking operators $A_1$ and $A_2$ into a single operator

$$
A(x) = \begin{bmatrix} A_1(x) \\ A_2(x) \end{bmatrix}
$$

can be done with [`deepinv.physics.stack()`](https://deepinv.org/api/stubs/deepinv.physics.stack.html.md#deepinv.physics.stack). The stacked operator is

```pycon
>>> import torch
>>> import deepinv as dinv
>>> x = torch.rand((1, 1, 8, 8))
>>> physics1 = dinv.physics.BlurFFT(img_size=(1, 8, 8), filter=dinv.physics.functional.blur.gaussian_blur(sigma=.2))
>>> physics2 = dinv.physics.Downsampling(img_size=(1, 8, 8), factor=2, filter=None)
>>> physics3 = dinv.physics.stack(physics1, physics2)
>>> physics3 = physics1.stack(physics2) # equivalent to the previous line
>>> y = physics3(x) #
>>> print(y[0].shape)
torch.Size([1, 1, 8, 8])
>>> print(y[1].shape)
torch.Size([1, 1, 4, 4])
>>> physics4 = physics3.stack(physics1) # add a new operator to the stack
>>> len(physics4)
3
```

The measurements are stored as [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, which can be accessed by index
(see the [TensorList](https://deepinv.org/user_guide/other/utils.html.md#tensorlist) user guide for more details).
The resulting stacked operator is a [`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) object, and has some useful
methods:

```pycon
>>> print(physics3[0](x).shape) # access the first operator only
torch.Size([1, 1, 8, 8])
>>> print(physics3[1](x).shape) # access the second operator only
torch.Size([1, 1, 4, 4])
```

#### TIP
See also the custom classes [`deepinv.optim.StackedPhysicsDataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.StackedPhysicsDataFidelity.html.md#deepinv.optim.StackedPhysicsDataFidelity) and [`deepinv.loss.StackedPhysicsLoss`](https://deepinv.org/api/stubs/deepinv.loss.StackedPhysicsLoss.html.md#deepinv.loss.StackedPhysicsLoss)
provide easy ways to build data fidelity terms and self-supervised losses with stacked operators.

Composing operators $A_1$ and $A_2$ into a single operator

$$
A(x) = A_2(A_1(x))
$$

can be done by multiplying the operators:

```pycon
>>> import torch
>>> import deepinv as dinv
>>> x = torch.rand((1, 1, 8, 8))
>>> physics1 = dinv.physics.Downsampling(img_size=(1, 8, 8), factor=2, filter=None)
>>> physics2 = dinv.physics.BlurFFT(img_size=(1, 4, 4), filter=dinv.physics.functional.blur.gaussian_blur(sigma=.2))
>>> physics = physics2 * physics1
>>> y = physics(x) # equivalent to y = physics2(physics1.A(x))
>>> print(y.shape)
torch.Size([1, 1, 4, 4])
```

or by using [`deepinv.physics.compose()`](https://deepinv.org/api/stubs/deepinv.physics.compose.html.md#deepinv.physics.compose):

```pycon
>>> import torch
>>> import deepinv as dinv
>>> x = torch.rand((1, 1, 8, 8))
>>> physics1 = dinv.physics.Downsampling(img_size=(1, 8, 8), factor=2, filter=None)
>>> physics2 = dinv.physics.BlurFFT(img_size=(1, 4, 4), filter=dinv.physics.functional.blur.gaussian_blur(sigma=.2))
>>> physics = dinv.physics.compose(physics1, physics2)
>>> y = physics(x)
>>> print(y.shape)
torch.Size([1, 1, 4, 4])
```

<a id="physics-wrappers"></a>

## Physics Wrappers

Some wrappers are provided to adapt existing operators to a new problem.

For example, given an operator $A: \mathbb{R}^N\to\mathbb{R}^M$ and an image $x\in\mathbb{R}^P$ with $P\neq N$, we need to resize the image to the operator’s input size.
This can be done with the [`deepinv.physics.LinearPhysicsMultiScaler`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysicsMultiScaler.html.md#deepinv.physics.LinearPhysicsMultiScaler) class:

```pycon
>>> import torch
>>> import deepinv as dinv
>>> physics = dinv.physics.BlurFFT(img_size=(1, 32, 32), filter=dinv.physics.functional.blur.gaussian_blur(sigma=.2))
>>> x = torch.rand((1, 1, 8, 8))  # define an image 4 times smaller than the physics input size (scale = 2)
>>> new_physics = dinv.physics.LinearPhysicsMultiScaler(physics, (1, 32, 32), factors=[2, 4, 8])  # define a multi-scale physics with base img size (1, 32, 32)
>>> y = new_physics(x, scale=2)  # compute the measurements with the new physics
>>> print(y.shape)
torch.Size([1, 1, 32, 32])
>>> Aty = new_physics.A_adjoint(y, scale=2)  # compute the adjoint operator
>>> print(Aty.shape)  # the output is the same size as the input image
torch.Size([1, 1, 8, 8])
```

Another example is the [`deepinv.physics.PhysicsCropper`](https://deepinv.org/api/stubs/deepinv.physics.PhysicsCropper.html.md#deepinv.physics.PhysicsCropper) class, which pads the input image to the operator’s input size.

```pycon
>>> import torch
>>> import deepinv as dinv
>>> physics = dinv.physics.BlurFFT(img_size=(1, 16, 16), filter=dinv.physics.functional.blur.gaussian_blur(sigma=.2))
>>> x = torch.rand((1, 1, 18, 21))  # define an input image larger than the physics input size
>>> new_physics = dinv.physics.PhysicsCropper(physics, crop=(2,5))  # define a padded physics
>>> y = new_physics(x)  # compute the measurements with cropping
>>> print(y.shape)
torch.Size([1, 1, 16, 16])
>>> Aty = new_physics.A_adjoint(y)  # compute the adjoint operator with cropping
>>> print(Aty.shape)  # the output is the same size as the input image
torch.Size([1, 1, 18, 21])
```
