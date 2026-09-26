# TVLoss

### *class* deepinv.loss.TVLoss(weight=1.0)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Total variation loss ($\ell_2$ norm).

It computes the loss $\|D\hat{x}\|_2^2$,
where $D$ is a normalized linear operator that computes the vertical and horizontal first order differences
of the reconstructed image $\hat{x}$.

* **Parameters:**
  **weight** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – scalar weight for the TV loss.

#### forward(x_net, \*\*kwargs)

Computes the TV loss.

* **Parameters:**
  **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructed image.
* **Returns:**
  torch.Tensor loss of size (batch_size,)
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
