<a id="iterative"></a>

# Iterative Reconstruction (PnP, RED, etc.)

Many image reconstruction algorithms can be shown to be solving
minimization problems of the form

$$
\begin{equation*}
\label{eq:min_prob}
\underset{x}{\arg\min} \quad \datafid{x}{y} +  \lambda \reg{x},
\end{equation*}

$$

where $\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}$ is a data-fidelity term, $\regname:\xset\mapsto \mathbb{R}_{+}$
is a prior term, and $\lambda$ is a positive scalar. The data-fidelity term measures the discrepancy between the
reconstruction $x$ and the data $y$, and the prior term enforces some prior knowledge on the reconstruction.

The data fidelity $f$ term is generally set as the negative log-likelihood $\datafid{x}{y} \propto - \log p(y|A(x))$.
See the [available data fidelity terms](https://deepinv.org/user_guide/reconstruction/optimization.html.md#data-fidelity).

The prior term $g_{\sigma}$ can be chosen as (See [available options](https://deepinv.org/user_guide/reconstruction/optimization.html.md#priors)):

| Method                            | Prior                                                                                                             |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------|
| Variational                       | Explicit prior ($\ell_1$, total-variation, etc.).                                                                 |
| Plug-and-Play (PnP)               | Replace $\operatorname{prox}_{\lambda g}(x)=\denoiser{x}{\sigma}$ where $\denoisername$ is a pretrained denoiser. |
| Regularization by Denoising (RED) | Replace $\nabla g(x)= x-\denoiser{x}{\sigma}$ where $\denoisername$ is a pretrained denoiser.                     |

## Implementing an Algorithm

Iterative algorithms can be easily implemented using the [optim module](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim), which provides many
pre-implemented [data-fidelity](https://deepinv.org/user_guide/reconstruction/optimization.html.md#data-fidelity) and [prior terms](https://deepinv.org/user_guide/reconstruction/optimization.html.md#priors) (including PnP and RED),
as well as many off-the-shelf optimization algorithms.

For example, a PnP proximal gradient descent (PGD) algorithm for
solving the inverse problem $y = \noise{\forw{x}}$ reads

$$
u_{k} &=  x_k - \gamma \nabla \datafid{x_k}{y} \\
x_{k+1} &= \denoiser{u_k}{\sigma},
$$

where $f(x)=\frac{1}{2}\|y-\forw{x}\|^2$ is a standard data-fidelity term,
and the prior is implicitly defined by a median filter denoiser, can be implemented as follows:

```pycon
>>> import deepinv as dinv
>>> from deepinv.utils import load_example
>>>
>>> x = load_example("cameraman.png", img_size=512, grayscale=True, device='cpu')
>>>
>>> physics = dinv.physics.Inpainting((1, 512, 512), mask = 0.5,
...                                    noise_model=dinv.physics.GaussianNoise(sigma=0.01))
>>>
>>> data_fidelity = dinv.optim.data_fidelity.L2()
>>> prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
>>> model = dinv.optim.PGD(prior=prior, data_fidelity=data_fidelity, stepsize=1.0, sigma_denoiser=0.1)
>>> y = physics(x)
>>> x_hat = model(y, physics)
>>> dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')
```

#### NOTE
While we offer predefined optimization iterators (in this case proximal gradient descent), it is possible to use
the optim class with any custom optimization algorithm. See the example
[PnP with custom optimization algorithm (Primal-Dual Condat-Vu)](https://deepinv.org/auto_examples/plug-and-play/demo_PnP_custom_optim.html.md#sphx-glr-auto-examples-plug-and-play-demo-pnp-custom-optim-py) for more details.

<a id="predefined-iterative"></a>

## Predefined Iterative Algorithms

We also provide pre-implemented iterative optimization algorithms,
which can be loaded in a single line of code, and used
to solve any inverse problem. The following algorithms are available:

#### Predefined methods

| **Method**                                                                             | **Description**                           |
|----------------------------------------------------------------------------------------|-------------------------------------------|
| [`deepinv.optim.DPIR`](https://deepinv.org/api/stubs/deepinv.optim.DPIR.html.md#deepinv.optim.DPIR) | Custom PnP algorithm with early stopping. |
| [`deepinv.optim.EPLL`](https://deepinv.org/api/stubs/deepinv.optim.EPLL.html.md#deepinv.optim.EPLL) | Patch-based reconstruction algorithm      |
