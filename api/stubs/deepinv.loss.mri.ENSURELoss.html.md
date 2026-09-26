# ENSURELoss

### *class* deepinv.loss.mri.ENSURELoss(sigma, physics_generator, tau=None, rng=None)

Bases: [`SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss)

ENSURE loss for image reconstruction in Gaussian noise.

The loss function is a special case of [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss) for MRI/inpainting with varying masks, and is designed for the following noise model:

$$
y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A_i(x).
$$

where $A_i\sim\mathcal{A}$ is assumed to be drawn from a set of measurement operators.
The loss is computed as

$$
\frac{1}{m}\|\Beta(A^{\dagger}y - \inverse{y})\|_2^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(\inverse{A^{\top}y+\tau b_i} -
\inverse{A^{\top}y}\right)
$$

where $R$ is the trainable network (which takes $A^\top y$ as input),
$A$ is the forward operator,
$y$ is the noisy measurement vector of size $m$,
$b\sim\mathcal{N}(0,I)$, $\tau\geq 0$ is a hyperparameter controlling the
Monte Carlo approximation of the divergence, and $\Beta=W^{-1}P$
where $P$ is the projection operator onto the range space of $\A^\top$
and $W$ is a weighting determined by the set of measurement operators where $W^2=\mathbb{E}\left[P\right]$.

The ENSURE loss was proposed in Aggarwal *et al.*<sup>[1](#footcite-aggarwal2023ensure)</sup> for MRI.

#### WARNING
This loss was originally proposed only to be used with [`artifact removal models`](https://deepinv.org/api/stubs/deepinv.models.ArtifactRemoval.html.md#deepinv.models.ArtifactRemoval) which can be written in the form $\inverse{\cdot}=r(A^\top\cdot)$.
If an artifact removal model is not used, then we evaluate the network directly instead.

We currently only provide an implementation for [`single-coil MRI`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI) and [`inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting),
where `A^top=A^dagger` such that $P=A^{\top}A$ and then $W$ is a weighted average over sampling masks.

* **Parameters:**
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Standard deviation of the Gaussian noise.
  * **physics_generator** ([*deepinv.physics.generator.PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)) – random physics generator used to compute the weighting $W$.
  * **tau** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Approximation constant for the Monte Carlo approximation of the divergence. Defaults to $0.1\sigma$.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – Optional random number generator. Default is None.

<hr />

* **References:**

* <a id='footcite-aggarwal2023ensure'>**[1]**</a> Hemant Kumar Aggarwal, Aniket Pramanik, Maneesh John, and Matthews Jacob. Ensure: a general approach for unsupervised training of deep image reconstruction algorithms. *IEEE Transactions on Medical Imaging*, 42(4):1133–1144, 2023.

#### div(x_net, y, f, physics)

Monte-Carlo estimation for the divergence of f(x).

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructions.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **f** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction network.
