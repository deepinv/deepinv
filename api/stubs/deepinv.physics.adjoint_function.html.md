# adjoint_function

### deepinv.physics.adjoint_function(A, input_size, device='cpu', dtype=torch.float)

Provides the adjoint function of a linear operator $A$, i.e., $A^{\top}$.

The generated function can be simply called as `A_adjoint(y)`, for example:

```pycon
>>> import torch
>>> from deepinv.physics.forward import adjoint_function
>>> A = lambda x: torch.roll(x, shifts=(1,1), dims=(2,3)) # shift image by one pixel
>>> x = torch.randn((4, 1, 5, 5))
>>> y = A(x)
>>> A_adjoint = adjoint_function(A, (4, 1, 5, 5))
>>> torch.allclose(A_adjoint(y), x) # we have A^T(A(x)) = x
True
```

* **Parameters:**
  * **A** (*Callable*) – linear operator $A$.
  * **input_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – size of the input tensor e.g. (B, C, H, W).
    The first dimension, i.e. batch size, should be equal or lower than the batch size B
    of the input tensor to the adjoint operator.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device where the adjoint operator is computed.
* **Returns:**
  (Callable) function that computes the adjoint of $A$.

## Examples using `adjoint_function`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This examples shows you how to use DeepInverse with your own physics.">  <div class="sphx-glr-thumbnail-title">Bring your own physics</div>
</div>
<!-- thumbnail-parent-div-close --></div>
