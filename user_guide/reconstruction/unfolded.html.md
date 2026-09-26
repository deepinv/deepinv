<a id="unfolded"></a>

# Unfolded Algorithms

Unfolded architectures (sometimes called ‘unrolled architectures’) are obtained by replacing parts of these algorithms
by learnable modules. In turn, they can be trained in an end-to-end fashion to solve inverse problems.
It suffices to set the argument `unfold=True` when creating an optimization algorithm from the [optimization](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim) module.
By default, if a neural network is used in place of a regularization or data-fidelity step, the parameters of the network are learnable.
Moreover, among all the parameters of the algorithm (e.g. step size, regularization parameter, etc.), the user can choose which are learnable and which are not via the argument `trainable_params`.

The following example creates an unfolded architecture of 5 proximal gradient steps
using a DnCNN plug-and-play prior a standard L2 data-fidelity term. The network can be trained end-to-end, and
evaluated with any forward model (e.g., denoising, deconvolution, inpainting, etc.).
Here, the stepsize `stepsize`, the regularization parameter `lambda_reg`, and the denoiser parameter `sigma_denoiser` of the plug-and-play denoising prior are learnable.

```pycon
>>> import torch
>>> import deepinv as dinv
>>> from deepinv.optim import PGD
>>>
>>> # Create a trainable unfolded architecture
>>> model = PGD(
...     unfold=True,
...     data_fidelity=dinv.optim.L2(),
...     prior=dinv.optim.PnP(dinv.models.DnCNN()),
...     stepsize=1.0,
...     sigma_denoiser=0.1,
...     lambda_reg=1,
...     max_iter=5,
...     trainable_params=["stepsize", "sigma_denoiser", "lambda_reg"]
... )
>>> # Forward pass
>>> x = torch.randn(1, 3, 16, 16)
>>> physics = dinv.physics.Denoising()
>>> y = physics(x)
>>> x_hat = model(y, physics)
```

## Memory-efficient back-propagation

Some unfolded architectures rely on a least-squares solver to compute the proximal step w.r.t. the data-fidelity term (e.g., [`ADMM`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.ADMMIteration.html.md#deepinv.optim.optim_iterators.ADMMIteration) or [`HQS`](https://deepinv.org/api/stubs/deepinv.optim.optim_iterators.HQSIteration.html.md#deepinv.optim.optim_iterators.HQSIteration)). During backpropagation, a naive implementation requires storing the gradients of every intermediate step of the least squares solver, resulting in significant memory and computational costs.
We provide a memory-efficient back-propagation strategy that reduces the memory footprint during training, by computing the gradients of the least squares solver in closed-form, without storing any intermediate steps. This closed-form computation requires evaluating the least-squares solver one additional time during the gradient computation.
This is particularly useful when training deep unfolded architectures with many iterations.
To enable this feature, set the argument `implicit_backward_solver=True` (default is `True`) when creating the physics. It is supported for all linear physics
([`deepinv.physics.LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)).
Note that when setting `implicit_backward_solver=True`, we need to use a large enough number of iterations in the physics solver to ensure convergence (as the closed-form gradients assume that we converged to the least squares minimizer), otherwise the gradient might be inaccurate.
See also [Reducing the memory and computational complexity of unfolded network training](https://deepinv.org/auto_examples/unfolded/demo_unfolded_constant_memory.html.md#sphx-glr-auto-examples-unfolded-demo-unfolded-constant-memory-py) for more details.

```pycon
>>> import torch
>>> import deepinv as dinv
>>> from deepinv.optim import HQS
>>>
>>> # Create a trainable unfolded architecture
>>> model = HQS(
...     unfold=True,
...     data_fidelity=dinv.optim.L2(),
...     prior=dinv.optim.PnP(dinv.models.DnCNN()),
...     stepsize=1.0,
...     sigma_denoiser=1.0,
...     trainable_params=["stepsize", "sigma_denoiser"]
... )
>>> # Forward pass
>>> x = torch.randn(1, 3, 16, 16)
>>> physics = dinv.physics.Blur(filter=torch.ones(1, 1, 3, 3) / 9., implicit_backward_solver=True, max_iter=50)
>>> y = physics(x)
>>> x_hat = model(y, physics)
```

<a id="deep-equilibrium"></a>

## Deep Equilibrium

Deep Equilibrium models (DEQ) are a particular class of unfolded architectures where the backward pass
is performed via Fixed-Point iterations. DEQ algorithms can virtually unroll infinitely many layers leveraging
the **implicit function theorem**. The backward pass consists in looking for solutions of the fixed-point equation

$$
v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right)^{\top} v + u.
$$

where $u$ is the incoming gradient from the backward pass,
and $x^\star$ is the equilibrium point of the forward pass.
See [this tutorial](http://implicit-layers-tutorial.org/deep_equilibrium_models/) for more details.

For turning an optimization algorithm into a DEQ model, the `DEQ` argument of [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) must be an instance of [`deepinv.optim.DEQConfig`](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig), which defines the parameters for equilibrium-based implicit differentiation.
The [`deepinv.optim.DEQConfig`](https://deepinv.org/api/stubs/deepinv.optim.DEQConfig.html.md#deepinv.optim.DEQConfig) dataclass has the following attributes and default values:

```python
@dataclass
class DEQConfig:
    jacobian_free: bool = False
        # Whether to use a Jacobian-free backward pass.

    # Forward pass Anderson acceleration
    anderson_acceleration_forward: bool = False
        # Whether to use Anderson acceleration for solving the forward equilibrium.
    history_size_forward: int = 5
        # Number of past iterates used in Anderson acceleration for the forward pass.
    beta_anderson_acc_forward: float = 1.0
        # Momentum coefficient in Anderson acceleration for the forward pass.
    eps_anderson_acc_forward: float = 1e-4
        # Regularization parameter for Anderson acceleration in the forward pass.

    # Backward pass Anderson acceleration
    anderson_acceleration_backward: bool = False
        # Whether to use Anderson acceleration for solving the backward equilibrium.
    history_size_backward: int = 5
        # Number of past iterates used in Anderson acceleration for the backward pass.
    beta_anderson_acc_backward: float = 1.0
        # Momentum coefficient in Anderson acceleration for the backward pass.
    eps_anderson_acc_backward: float = 1e-4
        # Regularization parameter for Anderson acceleration in the backward pass.
    max_iter_backward: int = 50
        # Maximum number of iterations in the backward equilibrium solver.
```

By default, DEQ is disabled (`DEQ=None`). As soon as `DEQ` is not `None`, the above `DEQConfig` values are used.

#### NOTE
Currently, DEQ is only possible with [`Gradient Descent`](https://deepinv.org/api/stubs/deepinv.optim.GD.html.md#deepinv.optim.GD),  [`Proximal Gradient Descent`](https://deepinv.org/api/stubs/deepinv.optim.PGD.html.md#deepinv.optim.PGD) and [`Half-Quadratic-Splitting`](https://deepinv.org/api/stubs/deepinv.optim.HQS.html.md#deepinv.optim.HQS) optimization algorithms.

<a id="predefined-unfolded"></a>

## Predefined Unfolded Architectures

We also provide some off-the-shelf unfolded network architectures,
taken from the respective literatures.

#### Predefined unfolded architectures

| Model                                                                                        | Description                                 |
|----------------------------------------------------------------------------------------------|---------------------------------------------|
| [`deepinv.models.VarNet`](https://deepinv.org/api/stubs/deepinv.models.VarNet.html.md#deepinv.models.VarNet) | VarNet/E2E-VarNet MRI reconstruction models |
| [`deepinv.models.MoDL`](https://deepinv.org/api/stubs/deepinv.models.MoDL.html.md#deepinv.models.MoDL)     | MoDL MRI reconstruction model               |

<a id="custom-unfolded-blocks"></a>

## Predefined Unfolded Blocks

Some more specific unfolded architectures are also available.

The Primal-Dual Network (PDNet) uses [`deepinv.models.PDNet_PrimalBlock`](https://deepinv.org/api/stubs/deepinv.models.PDNet_PrimalBlock.html.md#deepinv.models.PDNet_PrimalBlock) and
[`deepinv.models.PDNet_DualBlock`](https://deepinv.org/api/stubs/deepinv.models.PDNet_DualBlock.html.md#deepinv.models.PDNet_DualBlock) as building blocks for the primal and dual steps respectively.
