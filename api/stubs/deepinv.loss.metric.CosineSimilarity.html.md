# CosineSimilarity

### *class* deepinv.loss.metric.CosineSimilarity(\*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric)

Cosine similarity metric.

Computes cosine similarity between reconstruction $\hat{x}$ and ground truth $x$.
A higher value means more similar. The metric is calculated as:

$\text{CosineSim}(\hat{x}, x) =\dfrac{\langle \hat{x}, x \rangle}{\|\hat{x}\|_2 \, \|x\|_2}$,where $\langle \hat{x}, x \rangle$ is the Euclidean inner product.

#### NOTE
By default, no reduction is applied over the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import CosineSimilarity
>>> m = CosineSimilarity()
>>> x_net = x = torch.ones(3, 2, 8, 8) # B,C,H,W
>>> m(x_net, x)
tensor([1.0000, 1.0000, 1.0000])
```

* **Parameters:**
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – take complex magnitude before computing similarity.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – reduction over batch (“mean”, “sum”, “none”/None).
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalization for inputs (“l2”, “min_max”, or None).
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – crop before computing metric.
