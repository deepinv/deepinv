# LinearPhysics

### *class* deepinv.physics.LinearPhysics(A=lambda x, \*\*kwargs: ..., A_adjoint=None, img_size=None, noise_model=ZeroNoise(), sensor_model=lambda x: ..., max_iter=50, tol=1e-4, solver='lsqr', implicit_backward_solver=True, device='cpu', \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Parent class for linear operators.

It describes the linear forward measurement process of the form

$$
y = N(A(x))
$$

where $x$ is an image of $n$ pixels, $y$ is the measurements of size $m$,
$A:\xset\mapsto \yset$ is a deterministic linear mapping capturing the physics of the acquisition
and $N:\yset\mapsto \yset$ is a stochastic mapping which characterizes the noise affecting
the measurements.

* **Parameters:**
  * **A** (*Callable*) – forward operator function which maps an image to the observed measurements $x\mapsto y$.
    It is recommended to normalize it to have unit norm.
  * **A_adjoint** (*None* *|* *Callable*) – transpose of the forward operator, which should verify the adjointness test.
    By default, it is set to `None`, which means that the adjoint is computed automatically using [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function).
    This automatic adjoint is computed using automatic differentiation, which is slower than a closed form adjoint, and can
    have a larger memory footprint. If you want to use the automatic adjoint, you should set the `img_size` parameter
    If you have a closed form for the adjoint, you can pass it as a callable function or rewrite the class method.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – (optional, only required if A_adjoint is not provided) Size of the signal/image `x`, e.g. `(C, ...)` where `C` is the number of channels and `...` are the spatial dimensions,
    used for the automatic adjoint computation.
  * **noise_model** (*Callable*) – function that adds noise to the measurements $N(z)$.
    See the noise module for some predefined functions.
  * **sensor_model** (*Callable*) – function that incorporates any sensor non-linearities to the sensing process,
    such as quantization or saturation, defined as a function $\eta(z)$, such that
    $y=\eta\left(N(A(x))\right)$. By default, the sensor_model is set to the identity $\eta(z)=z$.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – If the operator does not have a closed form pseudoinverse, the conjugate gradient algorithm
    is used for computing it, and this parameter fixes the maximum number of conjugate gradient iterations.
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – If the operator does not have a closed form pseudoinverse, a least squares algorithm
    is used for computing it, and this parameter fixes the relative tolerance of the least squares algorithm.
  * **solver** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – least squares solver to use. Choose between `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'`. See [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for more details.
  * **implicit_backward_solver** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, uses implicit differentiation for computing gradients through the [`deepinv.physics.LinearPhysics.A_dagger()`](#deepinv.physics.LinearPhysics.A_dagger) and [`deepinv.physics.LinearPhysics.prox_l2()`](#deepinv.physics.LinearPhysics.prox_l2), using [`deepinv.optim.linear.least_squares_implicit_backward()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares_implicit_backward.html.md#deepinv.optim.linear.least_squares_implicit_backward) instead of [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares). This can significantly reduce memory consumption, especially when using many iterations. If `False`, uses the standard autograd mechanism, which can be memory-intensive. Default is `True`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or cuda, every registered buffer and module parameters are recursively pushed onto the device during initialization.

<hr />

* **Examples:**
  Blur operator with a basic averaging filter applied to a 32x32 black image with
  a single white pixel in the center:
  ```pycon
  >>> from deepinv.physics.blur import Blur, Downsampling
  >>> x = torch.zeros((1, 1, 32, 32)) # Define black image of size 32x32
  >>> x[:, :, 16, 16] = 1 # Define one white pixel in the middle
  >>> w = torch.ones((1, 1, 3, 3)) / 9 # Basic 3x3 averaging filter
  >>> physics = Blur(filter=w)
  >>> y = physics(x)
  ```

  Linear operators can also be stacked. The measurements produced by the resulting
  model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, where each entry corresponds to the
  measurements of the corresponding operator (see [Combining Physics](https://deepinv.org/user_guide/physics/intro.html.md#physics-combining) for more information):
  ```pycon
  >>> physics1 = Blur(filter=w)
  >>> physics2 = Downsampling(img_size=((1, 32, 32)), filter="gaussian", factor=4)
  >>> physics = physics1.stack(physics2)
  >>> y = physics(x)
  ```

  Linear operators can also be composed by multiplying them:
  ```pycon
  >>> physics = physics1 * physics2
  >>> y = physics(x)
  ```

  Linear operators also come with an adjoint, a pseudoinverse, and proximal operators in a given norm:
  ```pycon
  >>> from deepinv.loss.metric import PSNR
  >>> physics = Blur(filter=w, padding='circular')
  >>> y = physics(x) # Compute measurements
  >>> x_dagger = physics.A_dagger(y) # Compute linear pseudoinverse
  >>> x_prox = physics.prox_l2(torch.zeros_like(x), y, 1.) # Compute prox at x=0
  >>> PSNR()(x, x_prox) > PSNR()(x, y) # Should be closer to the original
  tensor([True])
  ```

  The adjoint can be generated automatically using the [`deepinv.physics.adjoint_function()`](https://deepinv.org/api/stubs/deepinv.physics.adjoint_function.html.md#deepinv.physics.adjoint_function) method
  which relies on automatic differentiation, at the cost of a few extra computations per adjoint call:
  ```pycon
  >>> from deepinv.physics import LinearPhysics, adjoint_function
  >>> A = lambda x: torch.roll(x, shifts=(1,1), dims=(2,3)) # Shift image by one pixel
  >>> physics = LinearPhysics(A=A, A_adjoint=adjoint_function(A, (4, 1, 5, 5)))
  >>> x = torch.randn((4, 1, 5, 5))
  >>> y = physics(x)
  >>> torch.allclose(physics.A_adjoint(y), x) # We have A^T(A(x)) = x
  True
  ```

#### A_A_adjoint(y, \*\*kwargs)

A helper function that computes $A A^{\top}y$.

This function can speed up computation when $A A^{\top}$ is available in closed form.
Otherwise it just calls [`deepinv.physics.Physics.A()`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.A) and [`deepinv.physics.LinearPhysics.A_adjoint()`](#deepinv.physics.LinearPhysics.A_adjoint).

* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurement.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the product $AA^{\top}y$.

#### A_adjoint(y, \*\*kwargs)

Computes transpose of the forward operator $\tilde{x} = A^{\top}y$.
If $A$ is linear, it should be the exact transpose of the forward matrix.

#### NOTE
If the problem is non-linear, there is not a well-defined transpose operation,
but defining one can be useful for some reconstruction networks, such as [`deepinv.models.ArtifactRemoval`](https://deepinv.org/api/stubs/deepinv.models.ArtifactRemoval.html.md#deepinv.models.ArtifactRemoval).

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements.
  * **params** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – optional additional parameters for the adjoint operator.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) linear reconstruction $\tilde{x} = A^{\top}y$.

#### A_adjoint_A(x, \*\*kwargs)

A helper function that computes $A^{\top}Ax$.

This function can speed up computation when $A^{\top}A$ is available in closed form.
Otherwise it just calls [`deepinv.physics.Physics.A()`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics.A) and [`deepinv.physics.LinearPhysics.A_adjoint()`](#deepinv.physics.LinearPhysics.A_adjoint).

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the product $A^{\top}Ax$.

#### A_dagger(y, solver='CG', max_iter=None, tol=None, verbose=False, \*\*kwargs)

Computes the solution in $x$ to $y = Ax$ using a least squares solver.

This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – a measurement $y$ to reconstruct via the pseudoinverse.
  * **solver** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – least squares solver to use. Choose between `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'`. See [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for more details.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) The reconstructed image $x$.

#### A_vjp(x, v)

Computes the product between a vector $v$ and the Jacobian of the forward operator $A$ evaluated at $x$, defined as:

$$
A_{vjp}(x, v) = \left. \frac{\partial A}{\partial x}  \right|_x^\top  v = \conj{A} v.
$$

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – signal/image.
  * **v** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – vector of the size of the measurements.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the VJP product between $v$ and the Jacobian.

#### \_\_mul_\_(other, \*\*kwargs)

Concatenates two linear forward operators $A = A_1 \circ A_2$ via the \* operation

The resulting linear operator keeps the noise and sensor models of $A_1$.

* **Parameters:**
  **other** ([*deepinv.physics.LinearPhysics*](#deepinv.physics.LinearPhysics)) – Physics operator $A_2$
* **Returns:**
  ([`deepinv.physics.LinearPhysics`](#deepinv.physics.LinearPhysics)) concatenated operator

#### adjointness_test(u, \*\*kwargs)

Numerically check that $A^{\top}$ is indeed the adjoint of $A$.

* **Parameters:**
  **u** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – initialization point of the adjointness test method
* **Returns:**
  (float) a quantity that should be theoretically 0. In practice, it should be of the order of the chosen dtype precision (i.e. single or double).

#### compute_norm(x0, max_iter=100, tol=1e-3, verbose=True, squared=True, \*\*kwargs)

Computes the spectral $\ell_2$ norm (Lipschitz constant) of the operator $A$.

#### WARNING
By default, for backward compatibility, this method computes the **squared** spectral norm of $A$,
i.e., $\|A^{\top}A\|_2$. This behavior is deprecated and will change in a future version.
Set `squared=False` to compute the non-squared spectral norm $\|A\|_2$, or use
[`compute_sqnorm()`](#deepinv.physics.LinearPhysics.compute_sqnorm) to explicitly compute the squared norm.

Uses the [power method](https://en.wikipedia.org/wiki/Power_iteration).

* **Parameters:**
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an unbatched tensor sharing its shape, dtype and device with the initial iterate of the algorithm (its values are ignored)
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative variation criterion for convergence
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – print information
  * **squared** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True` (default, deprecated), computes $\|A^{\top}A\|_2$ (squared spectral norm of $A$).
    Use [`compute_sqnorm()`](#deepinv.physics.LinearPhysics.compute_sqnorm) instead.
    If `False`, computes $\|A\|_2$ (spectral norm of $A$).
  * **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optional parameters for the forward operator
* **Returns:**
  (torch.Tensor) spectral norm. If `squared=True`, returns $\|A^{\top}A\|_2$ (squared spectral norm of $A$).
  If `squared=False`, returns $\|A\|_2$ (spectral norm of $A$).
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### compute_sqnorm(x0, , max_iter=100, tol=1e-3, verbose=True, rng=None, \*\*kwargs)

Computes the squared spectral $\ell_2$ norm of the operator $A$.

This is equivalent to computing the spectral norm of $A^{\top}A$, i.e., $\|A^{\top}A\|_2$.

Uses the [power method](https://en.wikipedia.org/wiki/Power_iteration).

* **Parameters:**
  * **x0** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – an unbatched tensor sharing its shape, dtype and device with the initial iterate of the algorithm (its values are ignored)
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative variation criterion for convergence
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – print information
  * **kwargs** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – optional parameters for the forward operator
* **Returns:**
  (torch.Tensor) squared spectral norm of $A$, i.e., $\|A^{\top}A\|_2 = \|A\|_2^2$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### condition_number(x, max_iter=500, tol=1e-6, verbose=False, \*\*kwargs)

Computes an approximation of the condition number of the linear operator $A$.

Uses the LSQR algorithm, see [`deepinv.optim.linear.lsqr()`](https://deepinv.org/api/stubs/deepinv.optim.linear.lsqr.html.md#deepinv.optim.linear.lsqr) for more details.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Any input tensor (e.g. random)
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – relative variation criterion for convergence
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – print information
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) condition number of the operator

#### *property* device *: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Returns the device where the physics parameters/buffers are stored.

* **Returns:**
  device of the physics parameters.

#### prox_l2(z, y, gamma, solver='CG', max_iter=None, tol=None, verbose=False, \*\*kwargs)

Computes proximal operator of $f(x) = \frac{1}{2}\|Ax-y\|^2$, i.e.,

$$
\underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2
$$

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – measurements tensor
  * **z** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – signal tensor
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – hyperparameter of the proximal operator
  * **solver** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – solver to use for the proximal operator, see [`deepinv.optim.linear.least_squares()`](https://deepinv.org/api/stubs/deepinv.optim.linear.least_squares.html.md#deepinv.optim.linear.least_squares) for details
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – maximum number of iterations for iterative solvers
  * **tol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – tolerance for iterative solvers
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to print information during the solver execution
  * **scale** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – scale at which to apply the physics operator
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) estimated signal tensor

#### stack(other)

Stacks forward operators $A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$.

The measurements produced by the resulting model are [`deepinv.utils.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList) objects, where
each entry corresponds to the measurements of the corresponding operator.

#### NOTE
When using the `stack` operator between two noise objects, the operation will retain only the second
noise.

See [Combining Physics](https://deepinv.org/user_guide/physics/intro.html.md#physics-combining) for more information.

* **Parameters:**
  **other** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Physics operator $A_2$
* **Returns:**
  ([`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics)) stacked operator

<a id="sphx-glr-backref-deepinv-physics-linearphysics"></a>

## Examples using `LinearPhysics`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">![](auto_examples/adversarial-learning/images/thumb/sphx_glr_demo_gan_imaging_thumb.png)

[Imaging inverse problems with adversarial networks](https://deepinv.org/auto_examples/adversarial-learning/demo_gan_imaging.html.md)

  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_dataset_thumb.png)

[Bring your own dataset](https://deepinv.org/auto_examples/basics/demo_custom_dataset.html.md)

  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_optim_thumb.png)

[Use iterative reconstruction algorithms](https://deepinv.org/auto_examples/basics/demo_custom_optim.html.md)

  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">![](auto_examples/basics/images/thumb/sphx_glr_demo_custom_physics_thumb.png)

[Bring your own physics](https://deepinv.org/auto_examples/basics/demo_custom_physics.html.md)

  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">![](auto_examples/basics/images/thumb/sphx_glr_demo_pretrained_model_thumb.png)

[Use a pretrained model](https://deepinv.org/auto_examples/basics/demo_pretrained_model.html.md)

  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">![](auto_examples/basics/images/thumb/sphx_glr_demo_quickstart_thumb.png)

[5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.html.md)

  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper carbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_deblurring_thumb.png)

[Blind deblurring with kernel estimation network](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_deblurring.html.md)

  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example introduces the Richardson-Lucy algorithm for deconvolution in the blind setting, where both the underlying clean image and the blur kernel are unknown.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_blind_richardsonlucy_thumb.png)

[Blind Richardson-Lucy deconvolution](https://deepinv.org/auto_examples/blind-inverse-problems/demo_blind_richardsonlucy.html.md)

  <div class="sphx-glr-thumbnail-title">Blind Richardson-Lucy deconvolution</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">![](auto_examples/blind-inverse-problems/images/thumb/sphx_glr_demo_optimizing_physics_parameter_thumb.png)

[Calibrating physics operators](https://deepinv.org/auto_examples/blind-inverse-problems/demo_optimizing_physics_parameter.html.md)

  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_physics_distributed_thumb.png)

[Distributed Physics Operators](https://deepinv.org/auto_examples/distributed/demo_physics_distributed.html.md)

  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">![](auto_examples/distributed/images/thumb/sphx_glr_demo_pnp_distributed_thumb.png)

[Distributed Plug-and-Play (PnP) Reconstruction](https://deepinv.org/auto_examples/distributed/demo_pnp_distributed.html.md)

  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">![](auto_examples/distributed/images/thumb/sphx_glr_demo_unrolled_distributed_thumb.png)

[Distributed Training of Unfolded Networks](https://deepinv.org/auto_examples/distributed/demo_unrolled_distributed.html.md)

  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="The data is taken from the 2DeteCT benchmark kiss2025benchmarking and dataset kiss20232detect, which is an industrial CT dataset of various materials acquired using a proprietary scanner from CWI (i.e. sinogram-to-image). The setup is matched exactly to kiss2025benchmarking, such that you can compare DeepInverse image reconstruction methods with the values reported in kiss2025benchmarking.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_2detect_thumb.png)

[Reconstruct real CT sinograms with the 2DeteCT benchmark](https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct real CT sinograms with the 2DeteCT benchmark</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_astra_tomography_thumb.png)

[Low-dose CT with ASTRA backend and Total-Variation (TV) prior](https://deepinv.org/auto_examples/external-libraries/demo_astra_tomography.html.md)

  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_connect_spyrit_thumb.png)

[Single-pixel imaging with Spyrit](https://deepinv.org/auto_examples/external-libraries/demo_connect_spyrit.html.md)

  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_io_thumb.png)

[Loading scientific images](https://deepinv.org/auto_examples/external-libraries/demo_io.html.md)

  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs raw non-Cartesian multicoil kspace data from the FastMRI breast dataset solomonFastMRI2025, for mammography.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_mrinufft_breast_thumb.png)

[Reconstruct accelerated non-Cartesian breast MRI acquisition data](https://deepinv.org/auto_examples/external-libraries/demo_mrinufft_breast.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct accelerated non-Cartesian breast MRI acquisition data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from aghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">![](auto_examples/external-libraries/images/thumb/sphx_glr_demo_ri_basic_thumb.png)

[Radio interferometric imaging with deepinverse](https://deepinv.org/auto_examples/external-libraries/demo_ri_basic.html.md)

  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">![](auto_examples/metrics/images/thumb/sphx_glr_demo_test_time_tuning_thumb.png)

[Blind inverse problems with no reference metrics](https://deepinv.org/auto_examples/metrics/demo_test_time_tuning.html.md)

  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model terris2025reconstruct to solve inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_foundation_model_thumb.png)

[Inference and fine-tune a foundation model](https://deepinv.org/auto_examples/models/demo_foundation_model.html.md)

  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo reconstructs undersampled k-space data for 2D cardiac and brain MRI on:">![](auto_examples/models/images/thumb/sphx_glr_demo_mri_pretrained_thumb.png)

[Reconstruct undersampled k-space for cardiac and brain MRI](https://deepinv.org/auto_examples/models/demo_mri_pretrained.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct undersampled k-space for cardiac and brain MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs real prospectively undersampled multicoil brain k-space from yu2022validation.">![](auto_examples/models/images/thumb/sphx_glr_demo_prospective_mri_thumb.png)

[Reconstruct prospectively-undersampled raw multicoil MRI](https://deepinv.org/auto_examples/models/demo_prospective_mri.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstruct prospectively-undersampled raw multicoil MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">![](auto_examples/models/images/thumb/sphx_glr_demo_super_resolution_thumb.png)

[Super-resolution with SRResNet](https://deepinv.org/auto_examples/models/demo_super_resolution.html.md)

  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">![](auto_examples/models/images/thumb/sphx_glr_demo_training_thumb.png)

[Training a reconstruction model](https://deepinv.org/auto_examples/models/demo_training.html.md)

  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_3D_denoising_thumb.png)

[3D denoising of brain MRI with wavelet and TV priors](https://deepinv.org/auto_examples/optimization/demo_3D_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">3D denoising of brain MRI with wavelet and TV priors</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_TV_minimisation_thumb.png)

[Image deblurring with Total-Variation (TV) prior](https://deepinv.org/auto_examples/optimization/demo_TV_minimisation.html.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_custom_prior_thumb.png)

[Image deblurring with custom deep explicit prior.](https://deepinv.org/auto_examples/optimization/demo_custom_prior.html.md)

  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_dip_thumb.png)

[Reconstructing an image using the deep image prior.](https://deepinv.org/auto_examples/optimization/demo_dip.html.md)

  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior zoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_epll_thumb.png)

[Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting](https://deepinv.org/auto_examples/optimization/demo_epll.html.md)

  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_multispectral_demosaicing_thumb.png)

[Multispectral demosaicing from raw sensor data](https://deepinv.org/auto_examples/optimization/demo_multispectral_demosaicing.html.md)

  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">![](auto_examples/optimization/images/thumb/sphx_glr_demo_patch_priors_CT_thumb.png)

[Patch priors for limited-angle computed tomography](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.html.md)

  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm sheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting richardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_poisson_mlem_thumb.png)

[Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)](https://deepinv.org/auto_examples/optimization/demo_poisson_mlem.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">![](auto_examples/optimization/images/thumb/sphx_glr_demo_wavelet_prior_thumb.png)

[Image inpainting with wavelet prior](https://deepinv.org/auto_examples/optimization/demo_wavelet_prior.html.md)

  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise using the Generalized Anscombe Transform (GAT), which converts any Gaussian denoiser into a Poisson-Gaussian denoiser makitalo2012optimal.">![](auto_examples/physics/images/thumb/sphx_glr_demo_anscombe_thumb.png)

[Poisson-Gaussian Denoising with the Generalized Anscombe Transform](https://deepinv.org/auto_examples/physics/demo_anscombe.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">![](auto_examples/physics/images/thumb/sphx_glr_demo_blur_tour_thumb.png)

[Tour of blur operators](https://deepinv.org/auto_examples/physics/demo_blur_tour.html.md)

  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding liu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">![](auto_examples/physics/images/thumb/sphx_glr_demo_liu_jia_padding_thumb.png)

[Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding](https://deepinv.org/auto_examples/physics/demo_liu_jia_padding.html.md)

  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">![](auto_examples/physics/images/thumb/sphx_glr_demo_microscopy_3d_thumb.png)

[3D diffraction PSF](https://deepinv.org/auto_examples/physics/demo_microscopy_3d.html.md)

  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">![](auto_examples/physics/images/thumb/sphx_glr_demo_mri_tour_thumb.png)

[Tour of MRI functionality in DeepInverse](https://deepinv.org/auto_examples/physics/demo_mri_tour.html.md)

  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet2d_thumb.png)

[Positron emission tomography (PET) in 2D](https://deepinv.org/auto_examples/physics/demo_pet2d.html.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet3d_thumb.png)

[Positron emission tomography (PET) in 3D](https://deepinv.org/auto_examples/physics/demo_pet3d.html.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.html.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.html.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in here.">![](auto_examples/physics/images/thumb/sphx_glr_demo_physics_tour_thumb.png)

[Tour of forward sensing operators](https://deepinv.org/auto_examples/physics/demo_physics_tour.html.md)

  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">![](auto_examples/physics/images/thumb/sphx_glr_demo_remote_sensing_thumb.png)

[Remote sensing with satellite images](https://deepinv.org/auto_examples/physics/demo_remote_sensing.html.md)

  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spatial_unwrapping_thumb.png)

[Spatial unwrapping and modulo imaging](https://deepinv.org/auto_examples/physics/demo_spatial_unwrapping.html.md)

  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">![](auto_examples/physics/images/thumb/sphx_glr_demo_spc_thumb.png)

[Pattern Ordering in a Compressive Single Pixel Camera](https://deepinv.org/auto_examples/physics/demo_spc.html.md)

  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the plane-wave ultrafast ultrasound forward physics (deepinv.physics.UltrasoundPlaneWave) available in DeepInverse for pulse-echo imaging problems.">![](auto_examples/physics/images/thumb/sphx_glr_demo_ultrasound_tour_thumb.png)

[Tour of ultrafast ultrasound in DeepInverse](https://deepinv.org/auto_examples/physics/demo_ultrasound_tour.html.md)

  <div class="sphx-glr-thumbnail-title">Tour of ultrafast ultrasound in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in zhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_DPIR_deblur_thumb.png)

[DPIR method for PnP image deblurring.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_DPIR_deblur.html.md)

  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_custom_optim_thumb.png)

[PnP with custom optimization algorithm (Primal-Dual Condat-Vu)](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_custom_optim.html.md)

  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See hurault2023convergent for more details.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_mirror_descent_thumb.png)

[Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_mirror_descent.html.md)

  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_PnP_multiscale_thumb.png)

[Multi-scale Plug-and-Play for Inpainting](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_multiscale.html.md)

  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of romano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) hurault2021gradient which provides an explicit prior.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_RED_GSPnP_SR_thumb.png)

[Regularization by Denoising (RED) for Super-Resolution.](https://deepinv.org/auto_examples/plug-and-play/demo_RED_GSPnP_SR.html.md)

  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use deepinv.physics.UltrasoundPlaneWave to reconstructs an in-vivo carotid acquisition of the EPFL LTS5 ultrafast ultrasound dataset from raw RF ultrasound data.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_ultrasound_invivo_PnP_thumb.png)

[In-vivo ultrafast ultrasound reconstruction with Plug-and-Play](https://deepinv.org/auto_examples/plug-and-play/demo_ultrasound_invivo_PnP.html.md)

  <div class="sphx-glr-thumbnail-title">In-vivo ultrafast ultrasound reconstruction with Plug-and-Play</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">![](auto_examples/plug-and-play/images/thumb/sphx_glr_demo_vanilla_PnP_thumb.png)

[Vanilla PnP for computed tomography (CT).](https://deepinv.org/auto_examples/plug-and-play/demo_vanilla_PnP.html.md)

  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_custom_kernel_thumb.png)

[Building your custom MCMC sampling algorithm.](https://deepinv.org/auto_examples/sampling/demo_custom_kernel.html.md)

  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm kawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_ddrm_thumb.png)

[Image reconstruction with a diffusion model](https://deepinv.org/auto_examples/sampling/demo_ddrm.html.md)

  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from zhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffpir_thumb.png)

[Implementing DiffPIR](https://deepinv.org/auto_examples/sampling/demo_diffpir.html.md)

  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusers_thumb.png)

[Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse](https://deepinv.org/auto_examples/sampling/demo_diffusers.html.md)

  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_diffusion_sde_thumb.png)

[Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.html.md)

  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in chung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_dps_thumb.png)

[DPS – Posterior Sampling with Diffusion Models](https://deepinv.org/auto_examples/sampling/demo_dps.html.md)

  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">![](auto_examples/sampling/images/thumb/sphx_glr_demo_flow_matching_thumb.png)

[Flow-Matching for posterior sampling and unconditional generation](https://deepinv.org/auto_examples/sampling/demo_flow_matching.html.md)

  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example compares six approximations of the measurement-matching term used by diffusion posterior samplers.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_noisy_data_fidelity_thumb.png)

[Noisy data-fidelity terms for diffusion posterior sampling](https://deepinv.org/auto_examples/sampling/demo_noisy_data_fidelity.html.md)

  <div class="sphx-glr-thumbnail-title">Noisy data-fidelity terms for diffusion posterior sampling</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">![](auto_examples/sampling/images/thumb/sphx_glr_demo_sampling_thumb.png)

[Uncertainty quantification with PnP-ULA.](https://deepinv.org/auto_examples/sampling/demo_sampling.html.md)

  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_artifact2artifact_thumb.png)

[Self-supervised MRI reconstruction with Artifact2Artifact](https://deepinv.org/auto_examples/self-supervised-learning/demo_artifact2artifact.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_ei_transforms_thumb.png)

[Image transformations for Equivariant Imaging](https://deepinv.org/auto_examples/self-supervised-learning/demo_ei_transforms.html.md)

  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_imaging_thumb.png)

[Self-supervised learning with Equivariant Imaging for MRI.](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_imaging.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only sechaud26Equivariant.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_equivariant_splitting_thumb.png)

[Self-supervised learning with Equivariant Splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_equivariant_splitting.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_lowfieldmri_thumb.png)

[Low-field MRI denoising without ground truth](https://deepinv.org/auto_examples/self-supervised-learning/demo_lowfieldmri.html.md)

  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_multioperator_imaging_thumb.png)

[Self-supervised learning from incomplete measurements of multiple operators.](https://deepinv.org/auto_examples/self-supervised-learning/demo_multioperator_imaging.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_n2n_denoising_thumb.png)

[Self-supervised denoising with the Neighbor2Neighbor loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_n2n_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_poisson2sparse_thumb.png)

[Poisson denoising using Poisson2Sparse](https://deepinv.org/auto_examples/self-supervised-learning/demo_poisson2sparse.html.md)

  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss monroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_r2r_denoising_thumb.png)

[Self-supervised denoising with the Generalized R2R loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_r2r_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_scan_specific_thumb.png)

[Scan-specific zero-shot SSDU for MRI](https://deepinv.org/auto_examples/self-supervised-learning/demo_scan_specific.html.md)

  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse hendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU yaman2020self.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_splitting_loss_thumb.png)

[Self-supervised learning with measurement splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_splitting_loss.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_sure_denoising_thumb.png)

[Self-supervised denoising with the SURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_sure_denoising.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by tachella2024unsure.">![](auto_examples/self-supervised-learning/images/thumb/sphx_glr_demo_unsure_thumb.png)

[Self-supervised denoising with the UNSURE loss.](https://deepinv.org/auto_examples/self-supervised-learning/demo_unsure.html.md)

  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">![](auto_examples/transforms-equivariance/images/thumb/sphx_glr_demo_transforms_thumb.png)

[Image transforms for equivariance & augmentations](https://deepinv.org/auto_examples/transforms-equivariance/demo_transforms.html.md)

  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_DEQ_thumb.png)

[Deep Equilibrium (DEQ) algorithms for image deblurring](https://deepinv.org/auto_examples/unfolded/demo_DEQ.html.md)

  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm gregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_LISTA_thumb.png)

[Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing](https://deepinv.org/auto_examples/unfolded/demo_LISTA.html.md)

  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_custom_prior_unfolded_thumb.png)

[Learned iterative custom prior](https://deepinv.org/auto_examples/unfolded/demo_custom_prior_unfolded.html.md)

  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_deal_thumb.png)

[DEAL denoising and reconstruction](https://deepinv.org/auto_examples/unfolded/demo_deal.html.md)

  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_learned_primal_dual_thumb.png)

[Learned Primal-Dual algorithm for CT scan.](https://deepinv.org/auto_examples/unfolded/demo_learned_primal_dual.html.md)

  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., ADMM or HQS):  ">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constant_memory_thumb.png)

[Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.html.md)

  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_unfolded_constrained_LISTA_thumb.png)

[Unfolded Chambolle-Pock for constrained image inpainting](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constrained_LISTA.html.md)

  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">![](auto_examples/unfolded/images/thumb/sphx_glr_demo_vanilla_unfolded_thumb.png)

[Vanilla Unfolded algorithm for super-resolution](https://deepinv.org/auto_examples/unfolded/demo_vanilla_unfolded.html.md)

  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
