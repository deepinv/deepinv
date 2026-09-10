# Haze

### *class* deepinv.physics.Haze(beta=0.1, offset=0, \*\*kwargs)

Bases: [`Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)

Standard haze model

The operator is defined as in He *et al.*<sup>[1](#footcite-he2010single)</sup>.

> $$
> y = t \odot I + a (1-t)
> $$

> where $t = \exp(-\beta d - o)$ is the medium transmission,  $I$ is the intensity (possibly RGB) image,
> $\odot$ denotes element-wise multiplication, $a>0$ is the atmospheric light,
> $d$ is the scene depth, and $\beta>0$ and $o$ are constants.

This is a non-linear inverse problems, whose unknown parameters are $I$, $d$, $a$.

* **Parameters:**
  * **beta** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – constant $\beta>0$
  * **offset** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – constant $o$

<hr />

* **References:**

* <a id='footcite-he2010single'>**[1]**</a> Kaiming He, Jian Sun, and Xiaoou Tang. Single image haze removal using dark channel prior. *IEEE transactions on pattern analysis and machine intelligence*, 33(12):2341–2353, 2010.

#### A(x, \*\*kwargs)

* **Parameters:**
  **x** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – The input x should be a tuple/list such that x[0] = image torch.tensor $I$,
  x[1] = depth torch.tensor $d$, x[2] = scalar or torch.tensor of one element $a$.
* **Returns:**
  (torch.Tensor) hazy image.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### A_dagger(y, \*\*kwargs)

Returns the trivial inverse where x[0] = y (trivial estimate of the image $I$),
x[1] = tensor of depth $d$ equal to one, x[2] = 1 for $a$.

<!-- note:

This trivial inverse can be useful for some reconstruction networks, such as ``deepinv.models.ArtifactRemoval``. -->
* **Parameters:**
  **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Hazy image.
* **Returns:**
  (deepinv.utils.TensorList) trivial inverse.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
