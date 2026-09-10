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
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using an iterative algorithm.">  <div class="sphx-glr-thumbnail-title">Use iterative reconstruction algorithms</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to reconstruct images using a pretrained model in one line.">  <div class="sphx-glr-thumbnail-title">Use a pretrained model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates blind image deblurring using the pretrained kernel estimation network from the paper :footcitecarbajal2023blind. The network estimates spatially-varying blur kernels from a blurred image, which are then used in a space-varying blur physics model to reconstruct the sharp image using a non-blind deblurring algorithm.">  <div class="sphx-glr-thumbnail-title">Blind deblurring with kernel estimation network</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.physics.Physics together with automatic differentiation to estimate unknown parameters of your forward operator.">  <div class="sphx-glr-thumbnail-title">Calibrating physics operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Physics Operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Many large-scale imaging problems involve operators that can be naturally decomposed as a stack of multiple sub-operators:">  <div class="sphx-glr-thumbnail-title">Distributed Plug-and-Play (PnP) Reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Astra tomography toolbox with deepinv, a popular toolbox for tomography with GPU acceleration.">  <div class="sphx-glr-thumbnail-title">Low-dose CT with ASTRA backend and Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use Spyrit linear models and measurements with DeepInverse. Here we use the HadamSplit2d linear model from Spyrit.">  <div class="sphx-glr-thumbnail-title">Single-pixel imaging with Spyrit</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various input/output functions provided by DeepInverse for handling medical and scientific imaging formats. We demonstrate loading and plotting from DICOM, NIfTI, ISMRMRD, PyTorch, NumPy and raster data sources.">  <div class="sphx-glr-thumbnail-title">Loading scientific images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In blind inverse problems, some parameters of the physics are unknown at test time. Running non-blind models requires knowing these parameters, which are often hard to estimate. For example, a denoiser needs the noise level \\sigma, a deblurring model needs the blur kernel.">  <div class="sphx-glr-thumbnail-title">Blind inverse problems with no reference metrics</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">  <div class="sphx-glr-thumbnail-title">3D denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to reconstruct a noisy and incomplete image using the deep image prior.">  <div class="sphx-glr-thumbnail-title">Reconstructing an image using the deep image prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use the expected patch log likelihood (EPLL) prior :footcitezoran2011learning. for denoising and inpainting of natural images. To this end, we consider the inverse problem y = Ax+\\epsilon, where A is either the identity (for denoising) or a masking operator (for inpainting) and \\epsilon\\sim\\mathcal{N}(0,\\sigma^2 I) is white Gaussian noise with standard deviation \\sigma.">  <div class="sphx-glr-thumbnail-title">Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a full-resolution multispectral image from raw snapshot mosaiced measurements using various reconstruction algorithms.">  <div class="sphx-glr-thumbnail-title">Multispectral demosaicing from raw sensor data</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to denoise images corrupted by Poisson-Gaussian noise &lt;deepinv.physics.PoissonGaussianNoise&gt; using the Generalized Anscombe Transform (GAT) &lt;deepinv.models.AnscombeDenoiser&gt;, which converts any Gaussian denoiser into a Poisson-Gaussian denoiser :footcitemakitalo2012optimal.">  <div class="sphx-glr-thumbnail-title">Poisson-Gaussian Denoising with the Generalized Anscombe Transform</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 2D blur operators in DeepInverse. In particular, we show how to use DiffractionBlurs (Fresnel diffraction), motion blurs and space varying blurs.">  <div class="sphx-glr-thumbnail-title">Tour of blur operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding &lt;deepinv.physics.functional.liu_jia_pad&gt; :footciteliu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of 3D blur operators in the library. In particular, we show how to use Diffraction Blurs (Fresnel diffraction) to simulate fluorescence microscopes.">  <div class="sphx-glr-thumbnail-title">3D diffraction PSF</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.">  <div class="sphx-glr-thumbnail-title">Remote sensing with satellite images</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows the use of the deepinv.physics.SpatialUnwrapping forward model and the deepinv.optim.ItohFidelity for unwrapping problems, which occur in modulo imaging, interferometry SAR and other imaging applications. It shows how to generate a wrapped phase image, apply blur and noise, and reconstruct the original phase using both DCT inversion and ADMM optimization.">  <div class="sphx-glr-thumbnail-title">Spatial unwrapping and modulo imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator to accelerate the sampling.">  <div class="sphx-glr-thumbnail-title">Building your custom MCMC sampling algorithm.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use the DDRM diffusion algorithm :footcitekawar2022denoising to reconstruct images and also compute the uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Image reconstruction with a diffusion model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we revisit the implementation of the DiffPIR diffusion algorithm for image reconstruction from :footcitezhu2023denoising. The full algorithm is implemented in deepinv.sampling.DiffPIR.">  <div class="sphx-glr-thumbnail-title">Implementing DiffPIR</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use deepinv.sampling.PosteriorDiffusion to perform posterior sampling. It also can be used to perform unconditional image generation with arbitrary denoisers, if the data fidelity term is not specified.">  <div class="sphx-glr-thumbnail-title">Building your diffusion posterior sampling method using SDEs</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will go over the steps in the Diffusion Posterior Sampling (DPS) algorithm introduced in :footcitechung2022diffusion. The full algorithm is implemented in deepinv.sampling.DPS.">  <div class="sphx-glr-thumbnail-title">DPS -- Posterior Sampling with Diffusion Models</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows how to restore a single image corrupted by Poisson noise using Poisson2Sparse, without requiring external training or knowledge of the noise level.">  <div class="sphx-glr-thumbnail-title">Poisson denoising using Poisson2Sparse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the Deep Equilibrium Attention Least Squares (DEAL) model in DeepInverse for both denoising and a simple reconstruction settings.">  <div class="sphx-glr-thumbnail-title">DEAL denoising and reconstruction</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="where both the data fidelity and the prior are learned modules, distinct for each iterations.">  <div class="sphx-glr-thumbnail-title">Learned Primal-Dual algorithm for CT scan.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
