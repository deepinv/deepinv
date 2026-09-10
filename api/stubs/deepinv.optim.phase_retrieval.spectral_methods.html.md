# spectral_methods

### deepinv.optim.phase_retrieval.spectral_methods(y, physics, x=None, n_iter=50, preprocessing=default_preprocessing, lamb=10.0, x_true=None, log=False, log_metric=cosine_similarity, early_stop=True, rtol=1e-5, verbose=False)

Utility function for spectral methods.

This function runs the Spectral Methods algorithm to find the principal eigenvector of the regularized weighted covariance matrix:

$$
M = \conj{B} \text{diag}(T(y)) B + \lambda I,

$$

where $B$ is the linear operator of the phase retrieval class, $T(\cdot)$ is a preprocessing function for the measurements, and $I$ is the identity matrix of corresponding dimensions. Parameter $\lambda$ tunes the strength of regularization.

To find the principal eigenvector, the function runs power iteration which is given by

$$
x_{k+1} &= M x_k \\
x_{k+1} &= \frac{x_{k+1}}{\|x_{k+1}\|}

$$

#### NOTE
This function assumes that the passed `x` is of consistent shape and dtype with the output of `physics.A_adjoint(y)`.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Instance of the physics modeling the forward matrix.
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Initial guess for the signals $x_0$.
  * **n_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of iterations.
  * **preprocessing** (*Callable*) – Function to preprocess the measurements. Default is $\max(1 - 1/x, -5)$.
  * **lamb** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Regularization parameter. Default is 10.
  * **log** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to log the metrics. Default is False.
  * **log_metric** (*Callable*) – Metric to log. Default is cosine similarity.
  * **early_stop** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Whether to early stop the iterations. Default is True.
  * **rtol** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Relative tolerance for early stopping. Default is 1e-5.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, prints information in case of an early stop. Default is False.
* **Returns:**
  The estimated signals $x$.

## Examples using `spectral_methods`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
