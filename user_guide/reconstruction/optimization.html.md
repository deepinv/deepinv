<a id="optim"></a>

# Optimization

This module contains a collection of routines that optimize

$$
\begin{equation}
\label{eq:min_prob}
\tag{1}
\underset{x}{\arg\min} \quad \datafid{x}{y} + \lambda \reg{x},
\end{equation}

$$

where the first term $\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}$ enforces data-fidelity, the second
term $\regname:\xset\mapsto \mathbb{R}_{+}$ acts as a regularization and
$\lambda > 0$ is a regularization parameter. More precisely, the data-fidelity term penalizes the discrepancy
between the data $y$ and the forward operator $A$ applied to the variable $x$, as

$$
\datafid{x}{y} = \distance{A(x)}{y},

$$

where $\distance{\cdot}{\cdot}$ is a distance function, and where $A:\xset\mapsto \yset$ is the forward
operator (see [`deepinv.physics.Physics`](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics))

#### NOTE
The regularization term often (but not always) depends on a hyperparameter $\sigma$ that can be either fixed
or estimated. For example, if the regularization is implicitly defined by a denoiser,
the hyperparameter is the noise level.

A typical example of optimization problem is the $\ell_1$-regularized least squares problem, where the data-fidelity term is
the squared $\ell_2$-norm and the regularization term is the $\ell_1$-norm. In this case, a possible
algorithm to solve the problem is the Proximal Gradient Descent (PGD) algorithm writing as

$$
\qquad x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname} \left( x_k - \gamma \nabla \datafidname(x_k, y) \right),

$$

where $\operatorname{prox}_{\lambda \regname}$ is the proximity operator of the regularization term, $\gamma$ is the
step size of the algorithm, and $\nabla \datafidname$ is the gradient of the data-fidelity term.

The following example illustrates the implementation of the PGD algorithm with DeepInverse to solve the $\ell_1$-regularized
least squares problem.

```pycon
>>> import torch
>>> import deepinv as dinv
>>> from deepinv.optim import L2, TVPrior
>>>
>>> # Forward operator, here inpainting
>>> mask = torch.ones((1, 2, 2))
>>> mask[0, 0, 0] = 0
>>> physics = dinv.physics.Inpainting(img_size=mask.shape, mask=mask)
>>> # Generate data
>>> x = torch.ones((1, 1, 2, 2))
>>> y = physics(x)
>>> data_fidelity = L2()  # The data fidelity term
>>> prior = TVPrior()  # The prior term
>>> lambd = 0.1  # Regularization parameter
>>> # Compute the squared norm of the operator A
>>> norm_A2 = physics.compute_norm(y, tol=1e-4, verbose=False).item()
>>> stepsize = 1/norm_A2  # stepsize for the PGD algorithm
>>>
>>> # PGD algorithm
>>> max_iter = 20  # number of iterations
>>> x_k = torch.zeros_like(x)  # initial guess
>>>
>>> for it in range(max_iter):
...     u = x_k - stepsize*data_fidelity.grad(x_k, y, physics)  # Gradient step
...     x_k = prior.prox(u, gamma=lambd*stepsize)  # Proximal step
...     cost = data_fidelity(x_k, y, physics) + lambd*prior(x_k)  # Compute the cost
...
>>> print(cost < 1e-5)
tensor([True])
>>> print('Estimated solution: ', x_k.flatten())
Estimated solution:  tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

<a id="potentials"></a>

## Potentials

The class [`deepinv.optim.Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential) implements potential scalar functions $h : \xset \to \mathbb{R}$
used to define an optimization problems. For example, both $f$ and $\regname$ are potentials.
This class comes with methods for computing operators useful for optimization,
such as its proximal operator $\operatorname{prox}_{h}$, its gradient $\nabla h$, its convex conjugate $h^*$, etc.

The following classes inherit from [`deepinv.optim.Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential)

| Class                                                                                                  | $h(x)$                               | Requires                          |
|--------------------------------------------------------------------------------------------------------|--------------------------------------|-----------------------------------|
| [`deepinv.optim.Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman)           | $\phi(x)$ with $\phi$ convex         | None                              |
| [`deepinv.optim.Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance)         | $d(x,y)$                             | $y$                               |
| [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) | $d(A(x),y)$ where $d$ is a distance. | $y$ & operator $A$                |
| [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)               | $g_{\sigma}(x)$                      | optional denoising level $\sigma$ |

<a id="data-fidelity"></a>

### Data Fidelity

The base class [`deepinv.optim.DataFidelity`](https://deepinv.org/api/stubs/deepinv.optim.DataFidelity.html.md#deepinv.optim.DataFidelity) implements data fidelity terms $\distance{A(x)}{y}$
where $A$ is the forward operator, $x\in\xset$ is a variable and $y\in\yset$ is the data,
and where $d$ is a distance function from the class [`deepinv.optim.Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance).
The class [`deepinv.optim.Distance`](https://deepinv.org/api/stubs/deepinv.optim.Distance.html.md#deepinv.optim.Distance) is implemented as a child class from [`deepinv.optim.Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential).

This data-fidelity class thus comes with useful methods,
such as $\operatorname{prox}_{\distancename\circ A}$ and $\nabla (\distancename \circ A)$ (among others)
which are used by most optimization algorithms.

#### Data Fidelity Overview

| Data Fidelity                                                                                                          | $d(A(x), y)$                                                                                                     |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`deepinv.optim.L1`](https://deepinv.org/api/stubs/deepinv.optim.L1.html.md#deepinv.optim.L1)                                     | $\|A(x) - y\|_1$                                                                                                 |
| [`deepinv.optim.L2`](https://deepinv.org/api/stubs/deepinv.optim.L2.html.md#deepinv.optim.L2)                                     | $\|A(x) - y\|_2^2$                                                                                               |
| [`deepinv.optim.IndicatorL2`](https://deepinv.org/api/stubs/deepinv.optim.IndicatorL2.html.md#deepinv.optim.IndicatorL2)                   | Indicator function of $\|A(x) - y\|_2 \leq \epsilon$                                                             |
| [`deepinv.optim.PoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.PoissonLikelihood.html.md#deepinv.optim.PoissonLikelihood)       | $\datafid{A(x)}{y} =  -y^{\top} \log(A(x)+\beta)+1^{\top}A(x)$                                                   |
| [`deepinv.optim.LogPoissonLikelihood`](https://deepinv.org/api/stubs/deepinv.optim.LogPoissonLikelihood.html.md#deepinv.optim.LogPoissonLikelihood) | $N_0 (1^{\top} \exp(-\mu A(x))+ \mu \exp(-\mu y)^{\top}A(x))$                                                    |
| [`deepinv.optim.AmplitudeLoss`](https://deepinv.org/api/stubs/deepinv.optim.AmplitudeLoss.html.md#deepinv.optim.AmplitudeLoss)               | $\sum_{i=1}^{m}{(\sqrt{|b_i^{\top} x|^2}-\sqrt{y_i})^2}$                                                         |
| [`deepinv.optim.ZeroFidelity`](https://deepinv.org/api/stubs/deepinv.optim.ZeroFidelity.html.md#deepinv.optim.ZeroFidelity)                 | $\datafid{x}{y} = 0$.                                                                                            |
| [`deepinv.optim.ItohFidelity`](https://deepinv.org/api/stubs/deepinv.optim.ItohFidelity.html.md#deepinv.optim.ItohFidelity)                 | $\datafid{x}{y} = \|Dx - w_t(Dy)\|_2^2$ where $D$ is a finite difference operator and $w_t$ the modulo operator. |

<a id="priors"></a>

### Priors

Prior functions are defined as $\reg{x}$ where $x\in\xset$ is a variable and
where $\regname$ is a function.

The base class is [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) implemented as a child class from [`deepinv.optim.Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential)
and therefore it comes with methods for computing operators such as $\operatorname{prox}_{\regname}$ and $\nabla \regname$.  This base class is used to implement user-defined differentiable
priors (eg. Tikhonov regularization) but also implicit priors (eg. plug-and-play methods).

#### Priors Overview

| Prior                                                                                                  | $\reg{x}$                                                                 | Explicit $\regname$   |
|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|-----------------------|
| [`deepinv.optim.PnP`](https://deepinv.org/api/stubs/deepinv.optim.PnP.html.md#deepinv.optim.PnP)                   | $\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)$ | No                    |
| [`deepinv.optim.RED`](https://deepinv.org/api/stubs/deepinv.optim.RED.html.md#deepinv.optim.RED)                   | $\nabla \reg{x} = x - \operatorname{D}_{\sigma}(x)$                       | No                    |
| [`deepinv.optim.ScorePrior`](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior)     | $\nabla \reg{x}=\left(x-\operatorname{D}_{\sigma}(x)\right)/\sigma^2$     | No                    |
| [`deepinv.optim.ZeroPrior`](https://deepinv.org/api/stubs/deepinv.optim.ZeroPrior.html.md#deepinv.optim.ZeroPrior)       | $\regname(x) = 0$                                                         | Yes                   |
| [`deepinv.optim.Tikhonov`](https://deepinv.org/api/stubs/deepinv.optim.Tikhonov.html.md#deepinv.optim.Tikhonov)         | $\reg{x}=\|x\|_2^2$                                                       | Yes                   |
| [`deepinv.optim.L1Prior`](https://deepinv.org/api/stubs/deepinv.optim.L1Prior.html.md#deepinv.optim.L1Prior)           | $\reg{x}=\|x\|_1$                                                         | Yes                   |
| [`deepinv.optim.WaveletPrior`](https://deepinv.org/api/stubs/deepinv.optim.WaveletPrior.html.md#deepinv.optim.WaveletPrior) | $\reg{x} = \|\Psi x\|_{p}$ where $\Psi$ is a wavelet transform            | Yes                   |
| [`deepinv.optim.TVPrior`](https://deepinv.org/api/stubs/deepinv.optim.TVPrior.html.md#deepinv.optim.TVPrior)           | $\reg{x}=\|Dx\|_{1,2}$ where $D$ is a finite difference operator          | Yes                   |
| [`deepinv.optim.TVL1Prior`](https://deepinv.org/api/stubs/deepinv.optim.TVL1Prior.html.md#deepinv.optim.TVL1Prior)       | $\reg{x}=\|Dx\|_{1}$ where $D$ is a finite difference operator            | Yes                   |
| [`deepinv.optim.PatchPrior`](https://deepinv.org/api/stubs/deepinv.optim.PatchPrior.html.md#deepinv.optim.PatchPrior)     | $\reg{x} = \sum_i h(P_i x)$ for some prior $h(x)$ on the space of patches | Yes                   |
| [`deepinv.optim.PatchNR`](https://deepinv.org/api/stubs/deepinv.optim.PatchNR.html.md#deepinv.optim.PatchNR)           | Patch prior via normalizing flows.                                        | Yes                   |
| [`deepinv.optim.L12Prior`](https://deepinv.org/api/stubs/deepinv.optim.L12Prior.html.md#deepinv.optim.L12Prior)         | $\reg{x} = \sum_i\| x_i \|_2$                                             | Yes                   |

<a id="optim-iterators"></a>

## Predefined Algorithms

Optimization algorithm inherit from the base class [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim), which serves as a common interface
for all predefined optimization algorithms.

Classical optimizations algorithms are already implemented as subclasses of [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim), such as
[`deepinv.optim.GD`](https://deepinv.org/api/stubs/deepinv.optim.GD.html.md#deepinv.optim.GD), [`deepinv.optim.PGD`](https://deepinv.org/api/stubs/deepinv.optim.PGD.html.md#deepinv.optim.PGD), [`deepinv.optim.ADMM`](https://deepinv.org/api/stubs/deepinv.optim.ADMM.html.md#deepinv.optim.ADMM), etc.

For example, we can create the same proximal gradient algorithm as the one at the beginning of this page, in one line of code:

```pycon
>>> model = dinv.optim.PGD(prior=prior, data_fidelity=data_fidelity, stepsize=stepsize, lambda_reg=lambd, max_iter=max_iter)
>>> x_hat = model(y, physics)
>>> dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')
```

Some predefined optimizers are provided:

| Algorithm                                                                                | Iteration                                                                                                                                                                                                                 |
|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.optim.GD`](https://deepinv.org/api/stubs/deepinv.optim.GD.html.md#deepinv.optim.GD)       | $v_{k} = \nabla f(x_k) + \lambda \nabla \reg{x_k}$<br/><br/><br/>$x_{k+1} = x_k-\gamma v_{k}$<br/><br/>                                                                                                                   |
| [`deepinv.optim.PGD`](https://deepinv.org/api/stubs/deepinv.optim.PGD.html.md#deepinv.optim.PGD)     | $u_{k} = x_k - \gamma \nabla f(x_k)$<br/><br/><br/>$x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(u_k)$<br/><br/>                                                                                               |
| [`deepinv.optim.FISTA`](https://deepinv.org/api/stubs/deepinv.optim.FISTA.html.md#deepinv.optim.FISTA) | $u_{k} = z_k -  \gamma \nabla f(z_k)$<br/><br/><br/>$x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(u_k)$<br/><br/><br/>$z_{k+1} = x_{k+1} + \alpha_k (x_{k+1} - x_k)$<br/><br/>                                 |
| [`deepinv.optim.HQS`](https://deepinv.org/api/stubs/deepinv.optim.HQS.html.md#deepinv.optim.HQS)     | $u_{k} = \operatorname{prox}_{\gamma f}(x_k)$<br/><br/><br/>$x_{k+1} = \operatorname{prox}_{\sigma \lambda \regname}(u_k)$<br/><br/>                                                                                      |
| [`deepinv.optim.ADMM`](https://deepinv.org/api/stubs/deepinv.optim.ADMM.html.md#deepinv.optim.ADMM)   | $u_{k+1} = \operatorname{prox}_{\gamma f}(x_k - z_k)$<br/><br/><br/>$x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(u_{k+1} + z_k)$<br/><br/><br/>$z_{k+1} = z_k + \beta (u_{k+1} - x_{k+1})$<br/><br/>          |
| [`deepinv.optim.DRS`](https://deepinv.org/api/stubs/deepinv.optim.DRS.html.md#deepinv.optim.DRS)     | $u_{k+1} = \operatorname{prox}_{\gamma f}(z_k)$<br/><br/><br/>$x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(2*u_{k+1}-z_k)$<br/><br/><br/>$z_{k+1} = z_k + \beta (x_{k+1} - u_{k+1})$<br/><br/>                |
| [`deepinv.optim.PDCP`](https://deepinv.org/api/stubs/deepinv.optim.PDCP.html.md#deepinv.optim.PDCP)   | $u_{k+1} = \operatorname{prox}_{\sigma F^*}(u_k + \sigma K z_k)$<br/><br/><br/>$x_{k+1} = \operatorname{prox}_{\tau \lambda G}(x_k-\tau K^\top u_{k+1})$<br/><br/><br/>$z_{k+1} = x_{k+1} + \beta(x_{k+1}-x_k)$<br/><br/> |
| [`deepinv.optim.MD`](https://deepinv.org/api/stubs/deepinv.optim.MD.html.md#deepinv.optim.MD)       | $v_{k} = \nabla f(x_k) + \lambda \nabla \reg{x_k}$<br/><br/><br/>$x_{k+1} = \nabla h^*(\nabla h(x_k) - \gamma v_{k})$<br/><br/>                                                                                           |
| [`deepinv.optim.PMD`](https://deepinv.org/api/stubs/deepinv.optim.PMD.html.md#deepinv.optim.PMD)     | $v_{k} = \nabla f(x_k) + \lambda \nabla \reg{x_k}$<br/><br/><br/>$u_{k} = \nabla h^*(\nabla h(x_k) - \gamma v_{k})$<br/><br/><br/>$x_{k+1} = \operatorname{prox^h}_{\gamma \lambda \regname}(u_k)$<br/><br/>              |
| [`deepinv.optim.SIRT`](https://deepinv.org/api/stubs/deepinv.optim.SIRT.html.md#deepinv.optim.SIRT)   | $x_{k+1} = x_k + \tau V (A^{\top} W (y - A x_k))$<br/><br/>                                                                                                                                                               |
| [`deepinv.optim.MLEM`](https://deepinv.org/api/stubs/deepinv.optim.MLEM.html.md#deepinv.optim.MLEM)   | $x_{k+1} = \frac{x_k}{A^{\top} 1} \odot A^{\top} \frac{y}{A x_k}$<br/><br/>                                                                                                                                               |
| [`deepinv.optim.OSEM`](https://deepinv.org/api/stubs/deepinv.optim.OSEM.html.md#deepinv.optim.OSEM)   | $x_{k,l+1} = \frac{x_{k,l}}{A_l^{\top} 1} \odot A_l^{\top} \frac{y_l}{A_l x_{k,l}}$<br/><br/>                                                                                                                             |

<a id="initialization"></a>

### Initialization

By default, in these predefined algorithms, the iterates are initialized with the adjoint applied to the measurement $A^{\top}y$,
when the adjoint is defined, and with the observation $y$ if the adjoint is not defined.

Custom initialization can be defined in two ways:

1. When calling the model via the `init` argument in the `forward` method of [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim).
   In this case, `init` can be either a fixed initialization or a Callable function of the form `init(y, physics)` that takes as input
   the measurement $y$ and the physics `physics`. The output of the function or the fixed initialization can be either:
   - a tuple of tensors $(x_0, z_0)$ (where $x_0$ and $z_0$ are the initial primal and dual variables),
   - a single tensor $x_0$ (if no dual variables $z_0$ are used), or
   - a dictionary of the form `X = {'est': (x_0, z_0)}`.
2. When creating the optim model via the [`custom_init`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) argument.
   In this case, it must be set as a callable function `custom_init(y, physics)` that takes as input
   the measurement $y$ and the physics $A$ and returns the initialization in the same form as in case 1.

For example, for initializing the above PGD algorithm with the pseudo-inverse of the measurement operator $A^{\dagger}y$,
one can either use the `init` argument when calling the standard `PGD` model:

```pycon
>>> x_hat = model(y, physics, init=physics.A_dagger(y))
>>> dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')
```

or one can define a custom initialization function and pass it to the `custom_init` argument when creating the optimization model:

```pycon
>>> def pseudo_inverse_init(y, physics):
...     return physics.A_dagger(y)
>>> model = dinv.optim.PGD(custom_init=pseudo_inverse_init, prior=prior, data_fidelity=data_fidelity, stepsize=stepsize, lambda_reg=lambd, max_iter=max_iter)
>>> x_hat = model(y, physics)
>>> dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')
```

<a id="optim-params"></a>

### Optimization Parameters

The parameters of generic optimization algorithms, such as
stepsize, regularization parameter, standard deviation of denoiser prior can be passed as arguments to the constructor of the optimization algorithm.
Alternatively, the parameters can be defined via the dictionary `params_algo`. This dictionary contains keys that are strings corresponding to the name of the parameters.

| Parameters name                                                           | Meaning                                                                                                                                                                    | Recommended Values                                                                                                                                                                                                       |
|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"stepsize"`                                                              | Step size of the optimization algorithm.                                                                                                                                   | Should be positive. Depending on the algorithm,<br/><br/><br/>needs to be small enough for convergence;<br/><br/><br/>e.g. for PGD with `g_first=False`,<br/><br/><br/>should be smaller than $1/(\|A\|_2^2)$.<br/><br/> |
| `"lambda_reg"` (or `"lambda"` if passed via the dictionary `params_algo`) | Regularization parameter $\lambda$<br/><br/><br/>multiplying the regularization term.<br/><br/>                                                                            | Should be positive.                                                                                                                                                                                                      |
| `"g_param"` or `"sigma_denoiser"`                                         | Optional prior hyper-parameter which $\regname$ depends on.<br/><br/><br/>For priors based on denoisers,<br/><br/><br/>corresponds to the noise level $\sigma$ .<br/><br/> | Should be positive.                                                                                                                                                                                                      |
| `"beta"`                                                                  | Relaxation parameter used in<br/><br/><br/>ADMM, DRS, CP.<br/><br/>                                                                                                        | Should be positive.                                                                                                                                                                                                      |
| `"stepsize_dual"`                                                         | Step size in the dual update in the<br/><br/><br/>Primal Dual algorithm (only required by CP).<br/><br/>                                                                   | Should be positive.                                                                                                                                                                                                      |

Each parameter can be given as an iterable (i.e., a list) with a distinct value for each iteration or
a single float (same parameter value for each iteration).

Moreover, backtracking can be used to automatically adapt the stepsize at each iteration. Backtracking consists in choosing
the largest stepsize $\tau$ such that, at each iteration, sufficient decrease of the cost function $F$ is achieved.
More precisely, Given $\gamma \in (0,1/2)$ and $\eta \in (0,1)$ and an initial stepsize $\tau > 0$,
the following update rule is applied at each iteration $k$:

$$
\text{ while } F(x_k) - F(x_{k+1}) < \frac{\gamma}{\tau} || x_{k-1} - x_k ||^2, \,\, \text{ do } \tau \leftarrow \eta \tau

$$

In order to use backtracking, the argument  `backtracking` of [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) must be an instance of [`deepinv.optim.BacktrackingConfig`](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig),
which defines the parameters for backtracking line-search. The [`deepinv.optim.BacktrackingConfig`](https://deepinv.org/api/stubs/deepinv.optim.BacktrackingConfig.html.md#deepinv.optim.BacktrackingConfig) dataclass has the following attributes and default values:

```python
@dataclass
class BacktrackingConfig:
    gamma: float = 0.1
        # Armijo-like parameter controlling sufficient decrease
    eta: float = 0.9
        # Step reduction factor
    max_iter: int = 10
        # Maximum number of backtracking iterations
```

By default, backtracking is disabled (i.e., `backtracking=None`), and as soon as `backtracking` is not `None`, the above `BacktrackingConfig` is used by default.

#### NOTE
To use backtracking, the optimized function (i.e., both the the data-fidelity and prior) must be explicit and provide a computable cost for the current iterate.
If the prior is not explicit (e.g. a denoiser) i.e. the argument `explicit_prior`, of the prior [`deepinv.optim.Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior) is `False` or if the argument `has_cost` of the class [`deepinv.optim.BaseOptim`](https://deepinv.org/api/stubs/deepinv.optim.BaseOptim.html.md#deepinv.optim.BaseOptim) is `False`, backtracking is automatically disabled.

<a id="bregman"></a>

### Bregman

Bregman potentials are defined as $\phi(x)$ where $x\in\xset$ is a variable and
where $\phi$ is a convex scalar function, and are defined via the base class [`deepinv.optim.Bregman`](https://deepinv.org/api/stubs/deepinv.optim.Bregman.html.md#deepinv.optim.Bregman).

In addition to the methods inherited from [`deepinv.optim.Potential`](https://deepinv.org/api/stubs/deepinv.optim.Potential.html.md#deepinv.optim.Potential) (gradient
$\nabla \phi$, conjugate $\phi^*$ and its gradient $\nabla \phi^*$),
this class provides the Bregman divergence $D(x,y) = \phi(x) - \phi^*(y) - x^{\top} y$,
and is well suited for performing Mirror Descent.

#### Bregman potentials

| Class                                                                                                          | Bregman potential $\phi(x)$                                                                        |
|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| [`deepinv.optim.bregman.BregmanL2`](https://deepinv.org/api/stubs/deepinv.optim.BregmanL2.html.md#deepinv.optim.BregmanL2)       | $\|x\|_2^2$                                                                                        |
| [`deepinv.optim.bregman.BurgEntropy`](https://deepinv.org/api/stubs/deepinv.optim.BurgEntropy.html.md#deepinv.optim.BurgEntropy)   | $- \sum_i \log x_i$                                                                                |
| [`deepinv.optim.bregman.NegEntropy`](https://deepinv.org/api/stubs/deepinv.optim.NegEntropy.html.md#deepinv.optim.NegEntropy)     | $\sum_i x_i \log x_i$                                                                              |
| [`deepinv.optim.bregman.Bregman_ICNN`](https://deepinv.org/api/stubs/deepinv.optim.Bregman_ICNN.html.md#deepinv.optim.Bregman_ICNN) | [`Convolutional Input Convex NN`](https://deepinv.org/api/stubs/deepinv.models.ICNN.html.md#deepinv.models.ICNN) |

<a id="optim-utils"></a>

## Utils

We provide some useful routines for optimization algorithms.

- [`deepinv.optim.utils.gradient_descent()`](https://deepinv.org/api/stubs/deepinv.optim.utils.gradient_descent.html.md#deepinv.optim.utils.gradient_descent) implements the gradient descent algorithm.
