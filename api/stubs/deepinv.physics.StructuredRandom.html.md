# StructuredRandom

### *class* deepinv.physics.StructuredRandom(img_size, output_size, n_layers=1, transform_func=dst1, transform_func_inv=dst1, diagonals=None, device='cpu', rng=None, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Structured random linear operator model corresponding to the operator

$$
A(x) = \prod_{i=1}^N (F D_i) x,
$$

where $F$ is a matrix representing a structured transform, $D_i$ are diagonal matrices, and $N$ refers to the number of layers. It is also possible to replace $x$ with $Fx$ as an additional 0.5 layer.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – input shape. If (C, H, W), i.e., the input is a 2D signal with C channels, then zero-padding will be used for oversampling and cropping will be used for undersampling.
  * **output_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape of outputs.
  * **n_layers** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – number of layers $N$. If `layers=N + 0.5`, a first $F$ transform is included, ie $A(x)=|\prod_{i=1}^N (F D_i) F x|^2$. Default is 1.
  * **transform_func** (*Callable*) – structured transform function. Default is [`deepinv.physics.functional.dst1()`](https://deepinv.org/api/stubs/deepinv.physics.functional.dst1.html.md#deepinv.physics.functional.dst1).
  * **transform_func_inv** (*Callable*) – structured inverse transform function. Default is [`deepinv.physics.functional.dst1()`](https://deepinv.org/api/stubs/deepinv.physics.functional.dst1.html.md#deepinv.physics.functional.dst1).
  * **diagonals** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list)) – list of diagonal matrices. If None, a random ${-1,+1}$ mask matrix will be used. Default is None.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device of the physics. Default is ‘cpu’.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Random number generator. Default is None.

<a id="sphx-glr-backref-deepinv-physics-structuredrandom"></a>

## Examples using `StructuredRandom`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div>
<!-- thumbnail-parent-div-close --></div>
