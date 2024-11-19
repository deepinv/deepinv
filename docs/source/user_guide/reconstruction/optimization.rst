.. _optim:

Optimization
============

This package contains a collection of routines that optimize

.. math::
    \begin{equation}
    \label{eq:min_prob}
    \tag{1}
    \underset{x}{\arg\min} \quad \datafid{x}{y} + \lambda \reg{x},
    \end{equation}


where the first term :math:`\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity, the second
term :math:`\regname:\xset\mapsto \mathbb{R}_{+}` acts as a regularization and
:math:`\lambda > 0` is a regularization parameter. More precisely, the data-fidelity term penalizes the discrepancy
between the data :math:`y` and the forward operator :math:`A` applied to the variable :math:`x`, as

.. math::
    \datafid{x}{y} = \distance{A(x)}{y}

where :math:`\distance{\cdot}{\cdot}` is a distance function, and where :math:`A:\xset\mapsto \yset` is the forward
operator (see :meth:`deepinv.physics.Physics`)

.. note::

    The regularization term often (but not always) depends on a hyperparameter :math:`\sigma` that can be either fixed
    or estimated. For example, if the regularization is implicitly defined by a denoiser,
    the hyperparameter is the noise level.

A typical example of optimization problem is the :math:`\ell_1`-regularized least squares problem, where the data-fidelity term is
the squared :math:`\ell_2`-norm and the regularization term is the :math:`\ell_1`-norm. In this case, a possible
algorithm to solve the problem is the Proximal Gradient Descent (PGD) algorithm writing as

.. math::
    \qquad x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname} \left( x_k - \gamma \nabla \datafidname(x_k, y) \right),

where :math:`\operatorname{prox}_{\lambda \regname}` is the proximity operator of the regularization term, :math:`\gamma` is the
step size of the algorithm, and :math:`\nabla \datafidname` is the gradient of the data-fidelity term.

The following example illustrates the implementation of the PGD algorithm with DeepInverse to solve the :math:`\ell_1`-regularized
least squares problem.

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> from deepinv.optim import L2, TVPrior
    >>>
    >>> # Forward operator, here inpainting
    >>> mask = torch.ones((1, 2, 2))
    >>> mask[0, 0, 0] = 0
    >>> physics = dinv.physics.Inpainting(tensor_size=mask.shape, mask=mask)
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


Optimization algorithms such as the one above can be written as fixed point algorithms,
i.e. for :math:`k=1,2,...`

.. math::
    \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, f, g, A, y, ...)

where :math:`x` is a variable converging to the solution of the minimization problem, and
:math:`z` is an additional (dual) variable that may be required in the computation of the fixed point operator.


The function :meth:`deepinv.optim.optim_builder` returns an instance of :meth:`deepinv.optim.BaseOptim` with the
optimization algorithm of choice, either a predefined one (``"PGD"``, ``"ADMM"``, ``"HQS"``, etc.),
or with a user-defined one.



Optimization algorithm inherit from the base class :meth:`deepinv.optim.BaseOptim`, which serves as a common interface
for all optimization algorithms.


.. _data-fidelity:

Data Fidelity
-------------
This is the base class for the data fidelity term :math:`\distance{A(x)}{y}` where :math:`A` is the forward operator,
:math:`x\in\xset` is a variable and :math:`y\in\yset` is the data, and where :math:`d` is a distance function, from the class :meth:`deepinv.optim.Distance`.
The class :meth:`deepinv.optim.Distance` is implemented as a child class from :meth:`deepinv.optim.Potential`.

This data-fidelity class thus comes with methods, such as :math:`\operatorname{prox}_{\distancename\circ A}` and
:math:`\nabla (\distancename \circ A)` (among others), on which optimization algorithms rely.

.. list-table:: Data Fidelity Overview
   :widths: 25 30
   :header-rows: 1

   * - Data Fidelity
     - :math:`d(A(x), y)`
   * - :class:`deepinv.optim.L1`
     - :math:`\|A(x) - y\|_1`
   * - :class:`deepinv.optim.L2`
     - :math:`\|A(x) - y\|_2^2`
   * - :class:`deepinv.optim.IndicatorL2`
     - Indicator function of :math:`\|A(x) - y\|_2 \leq \epsilon`
   * - :class:`deepinv.optim.PoissonLikelihood`
     - Poisson negative log-likelihood: :math:`A(x) - y \log(A(x))`
   * - :class:`deepinv.optim.LogPoissonLikelihood`
     - Log-Poisson negative log-likelihood: :math:`N_0 (1^{\top} \exp(-\mu z)+ \mu \exp(-\mu y)^{\top}x)`
   * - :class:`deepinv.optim.AmplitudeLoss`
     - :math:`\sum_{i=1}^{m}{(\sqrt{|b_i^{\top} x|^2}-\sqrt{y_i})^2}`


.. _priors:

Priors
------
This is the base class for implementing prior functions :math:`\reg{x}` where :math:`x\in\xset` is a variable and
where :math:`\regname` is a function.

This class is implemented as a child class from :meth:`deepinv.optim.Potential` and therefore it comes with methods for computing
operators such as :math:`\operatorname{prox}_{\regname}` and :math:`\nabla \regname`.  This base class is used to implement user-defined differentiable
priors, such as the Tikhonov regularisation, but also implicit priors. For instance, in PnP methods, the method
computing the proximity operator is overwritten by a method performing denoising.

.. list-table:: Priors Overview
   :widths: 25 20 15
   :header-rows: 1

   * - Prior
     - :math:`\reg{x}`
     - Explicit :math:`\regname`
   * - :class:`deepinv.optim.PnP`
     - :math:`\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)`
     - No
   * - :class:`deepinv.optim.RED`
     - :math:`\nabla \reg{x} = x - \operatorname{D}_{\sigma}(x)`
     - No
   * - :class:`deepinv.optim.ScorePrior`
     - :math:`\nabla \reg{x}=\left(x-\operatorname{D}_{\sigma}(x)\right)/\sigma^2`
     - No
   * - :class:`deepinv.optim.Tikhonov`
     - :math:`\reg{x}=\|x\|_2^2`
     - Yes
   * - :class:`deepinv.optim.L1Prior`
     - :math:`\reg{x}=\|x\|_1`
     - Yes
   * - :class:`deepinv.optim.WaveletPrior`
     - :math:`\reg{x} = \|\Psi x\|_{p}` where :math:`\Psi` is a wavelet transform
     - Yes
   * - :class:`deepinv.optim.TVPrior`
     - :math:`\reg{x}=\|Dx\|_{1,2}` where :math:`D` is a finite difference operator
     - Yes
   * - :class:`deepinv.optim.PatchPrior`
     - :math:`\reg{x} = \sum_i h(P_i x)` for some prior :math:`h(x)` on the space of patches
     - Yes
   * - :class:`deepinv.optim.PatchNR`
     - Patch prior via normalizing flows.
     - Yes
   * - :class:`deepinv.optim.L12Prior`
     - :math:`\reg{x} = \sum_i\| x_i \|_2`
     - Yes


.. _bregman:

Bregman
-------
This is the base class for implementing Bregman potentials :math:`\phi(x)` where :math:`x\in\xset` is a variable and
where :math:`\phi` is a convex scalar function. All Bregman potentials inherit from

This class is implemented as a child class from :meth:`deepinv.optim.Potential` and therefore it comes with methods for computing
operators useful for Bregman optimization algorithms such as Mirror Descent: the gradient :math:`\nabla \phi`, the conjugate :math:`\phi^*` and its gradient :math:`\nabla \phi^*`, or the Bregman divergence :math:`D(x,y) = \phi(x) - \phi^*(y) - x^T y`.


.. list-table:: Priors Overview
   :widths: 35 20
   :header-rows: 1

    * - Class
      - Bregman Potential :math:`\phi(x)`
    * - :class:`deepinv.optim.bregman.BregmanL2`
      - :math:`\|x\|_2^2`
    * - :class:`deepinv.optim.bregman.BurgEntropy`
      - :math:`\sum_i x_i \log(x_i)`
    * - :class:`deepinv.optim.bregman.NegEntropy`
      - :math:`-\sum_i x_i \log(x_i)`
    * - :class:`deepinv.optim.bregman.Bregman_ICNN`
      - :math:`\sum_i x_i \log(x_i) - x_i`


.. _optim-params:

Parameters
---------------------
The parameters of the optimization algorithm, such as
stepsize, regularisation parameter, denoising standard deviation, etc.
are stored in a dictionary ``"params_algo"``, whose typical entries are:

.. list-table::
   :widths: 25 30 30
   :header-rows: 1

   * - Key
     - Meaning
     - Recommended Values
   * - ``"stepsize"``
     - Step size of the optimization algorithm.
     - | Should be positive. Depending on the algorithm,
       | needs to be small enough for convergence;
       | e.g. for PGD with ``g_first=False``,
       | should be smaller than :math:`1/(\|A\|_2^2)`.
   * - ``"lambda"``
     - | Regularization parameter :math:`\lambda`
       | multiplying the regularization term.
     - Should be positive.
   * - ``"g_param"``
     - | Optional parameter :math:`\sigma` which :math:`\regname` depends on.
       | For priors based on denoisers,
       | corresponds to the noise level.
     - Should be positive.
   * - ``"beta"``
     - | Relaxation parameter used in
       | ADMM, DRS, CP.
     - Should be positive.
   * - ``"stepsize_dual"``
     - | Step size in the dual update in the
       | Primal Dual algorithm (only required by CP).
     - Should be positive.

Each value of the dictionary can be either an iterable (i.e., a list with a distinct value for each iteration) or
a single float (same value for each iteration).


.. _optim-iterators:

Iterators
---------
An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
:math:`F = \datafidname + \lambda \regname` where :math:`\datafidname` is a data-fidelity term  that will be modeled
by an instance of physics and :math:`\regname` is a regularizer. The fixed point iteration takes the form

.. math::
    \qquad (x_{k+1}, z_{k+1}) = \operatorname{FixedPoint}(x_k, z_k, \datafidname, \regname, A, y, ...)

where :math:`x` is a variable converging to the solution of the minimization problem, and
:math:`z` is an additional variable that may be required in the computation of the fixed point operator.


The implementation of the fixed point algorithm in :meth:`deepinv.optim`,
following standard optimization theory, is split in two steps:

.. math::
    z_{k+1} = \operatorname{step}_{\datafidname}(x_k, z_k, y, A, ...)\\
    x_{k+1} = \operatorname{step}_{\regname}(x_k, z_k, y, A, ...)

where :math:`\operatorname{step}_{\datafidname}` and :math:`\operatorname{step}_g` are gradient and/or proximal steps
on :math:`\datafidname` and :math:`\regname`, while using additional inputs, such as :math:`A` and :math:`y`, but also stepsizes,
relaxation parameters, etc...

The fStep and gStep classes precisely implement these steps.

Some generic optimizer iterators are provided:

.. list-table::
   :widths: 25 30 30
   :header-rows: 1

   * - Algorithm
     - Iteration
     - Parameters

   * - :class:`Gradient Descent (GD) <deepinv.optim.optim_iterators.GDIteration>`
     - | :math:`v_{k} = \nabla f(x_k) + \nabla \reg{x_k}`
       | :math:`x_{k+1} = x_k-\gamma v_{k}`
     - ``"stepsize"``, ``"lambda"``, ``"g_param"``

   * - :class:`Proximal Gradient Descent (PGD) <deepinv.optim.optim_iterators.PGDIteration>`
     - | :math:`u_{k} = x_k - \gamma \nabla f(x_k)`
       | :math:`x_{k+1} = \operatorname{prox}_{\gamma \lambda \regname}(u_k)`
     - ``"stepsize"``, ``"lambda"``, ``"g_param"``

   * - :class:`Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) <deepinv.optim.optim_iterators.FISTAIteration>`
     - | :math:`u_{k} = z_k -  \gamma \nabla f(z_k)`
       | :math:`x_{k+1} = \operatorname{prox}_{\gamma \lambda g}(u_k)`
       | :math:`z_{k+1} = x_{k+1} + \alpha_k (x_{k+1} - x_k)`
     - ``"stepsize"``, ``"lambda"``, ``"g_param"``

   * - :class:`Half-Quadratic Splitting (HQS) <deepinv.optim.optim_iterators.HQSIteration>`
     - | :math:`u_{k} = \operatorname{prox}_{\gamma f}(x_k)`
       | :math:`x_{k+1} = \operatorname{prox}_{\sigma \lambda g}(u_k)`
     - ``"gamma"``, ``"lambda"``, ``"g_param"``

   * - :class:`Alternating Direction Method of Multipliers (ADMM) <deepinv.optim.optim_iterators.ADMMIteration>`
     - | :math:`u_{k+1} = \operatorname{prox}_{\gamma f}(x_k - z_k)`
       | :math:`x_{k+1} = \operatorname{prox}_{\gamma \lambda g}(u_{k+1} + z_k)`
       | :math:`z_{k+1} = z_k + \beta (u_{k+1} - x_{k+1})`
     - ``"gamma"``, ``"lambda"``, ``"g_param"``, ``"beta"``

   * - :class:`Douglas-Rachford Splitting (DRS) <deepinv.optim.optim_iterators.DRSIteration>`
     - | :math:`u_{k+1} = \operatorname{prox}_{\gamma f}(z_k)`
       | :math:`x_{k+1} = \operatorname{prox}_{\gamma \lambda g}(2*u_{k+1}-z_k)`
       | :math:`z_{k+1} = z_k + \beta (x_{k+1} - u_{k+1})`
     - ``"stepsize"``, ``"lambda"``, ``"g_param"``, ``"beta"``

   * - :class:`Chambolle-Pock (CP) <deepinv.optim.optim_iterators.CPIteration>`
     - | :math:`u_{k+1} = \operatorname{prox}_{\sigma F^*}(u_k + \sigma K z_k)`
       | :math:`x_{k+1} = \operatorname{prox}_{\tau \lambda G}(x_k-\tau K^\top u_{k+1})`
       | :math:`z_{k+1} = x_{k+1} + \beta(x_{k+1}-x_k)`
     - ``"gamma"``, ``"lambda"``, ``"g_param"``, ``"beta"``, ``"stepsize_dual"``

   * - :class:`Spectral Methods (SM) <deepinv.optim.optim_iterators.SMIteration>`
     - :math:`M = \conj{B} \text{diag}(T(y)) B + \lambda I`
     - (phase-retrieval only)


.. _optim-utils:

Utils
-----
We provide some useful routines for optimization algorithms.

- :class:`deepinv.optim.utils.conjugate_gradient` implements the conjugate gradient algorithm for solving linear systems.
- :class:`deepinv.optim.utils.gradient_descent` implements the gradient descent algorithm.
