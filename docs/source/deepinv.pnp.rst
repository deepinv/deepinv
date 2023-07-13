.. _pnp:

PnP and RED algorithms
======================

PnP and RED algorithms are optimization algorithms where optimisation steps on the prior term are replaced by denoising
operators.

When one replaces a proximity operator with a denoiser, the algorithm is called a Plug-and-Play (PnP) algorithm.
For instance, a PnP proximal gradient descent (PnP-PGD) algorithm for solving the inverse problem
:math:`y = \noise{\forw{x}}` reads

.. math::

    \begin{equation*}
    \begin{aligned}
    u_{k} &=  x_k - \gamma \lambda \nabla \datafid{x_k}{y} \\
    x_{k+1} &= \denoiser{u_k}{\sigma},
    \end{aligned}
    \end{equation*}

where :math:`\datafidname` is a data-fidelity term and :math:`\denoisername` is usually a denoiser, i.e. an operator
(or algorithm, or neural network) aiming at removing gaussian random noise with standard deviation :math:`\sigma`
from an image.

On the other hand, when one replaces a gradient step on the prior term with a denoising step, the algorithm is called a
Regularization by Denoising (RED) algorithm. For instance, a RED proximal gradient descent (RED-PGD) algorithm for
solving the inverse problem :math:`y = \noise{\forw{x}}` reads

.. math::

    \begin{equation*}
    \begin{aligned}
    u_{k} &=  x_k - \denoiser{x_k}{\sigma} \\
    x_{k+1} &= \operatorname{prox}_{\datafidname(\cdot, y)}(u_k).
    \end{aligned}
    \end{equation*}


Under restrictive assumptions on the denoiser, PnP and RED algorithms can be shown to be solving
minimization problems of the form

.. math::
    \begin{equation*}
    \label{eq:min_prob}
    \underset{x}{\arg\min} \quad \lambda \datafid{x}{y} + g_{\theta}(x),
    \end{equation*}


where the first term :math:`\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}` enforces data-fidelity and the second
term :math:`g_{\theta}:\xset\mapsto \mathbb{R}_{+}` is a prior implicitely learned by the denoising operator
:math:`\denoisername`, parametrized by the learnable parameters :math:`\theta` of the denoiser.

As a consequence, PnP algorithms in deepinv inherit from :meth:`deepinv.optim.BaseOptim` where the prior term is
replaced by a denoiser.


Priors and denoisers
--------------------
This is the base class for implementing prior functions :math:`\reg{x}` where :math:`x\in\xset` is a variable and
where :math:`\regname` is a function. It also encompasses implicitely defined priors, such as the ones arising in
PnP and RED algorithms and where the prior function :math:`\operatorname{prox}_{\regname}` (resp.
:math:`\nabla \regname`) is replaced by a denoiser.

While the base class is used to implement user-defined differentiable
priors, such as the Tikhonov regularisation, in the case of implicit priors for PnP and RED, the method
computing the proximity operator is overwritten by a method performing denoising.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.Prior
   deepinv.optim.PnP
   deepinv.optim.RED
   deepinv.optim.ScorePrior
   deepinv.optim.Tikhonov


We refer the reader to the Denoiser section of the :ref:`Models <models>` documentation for more details on how to implement a
denoiser.



Iterators
---------
An optim iterator is an object that implements a fixed point iteration for minimizing the sum of two functions
:math:`F = \lambda \datafidname + \regname` where :math:`\datafidname` is a data-fidelity term  that will be modeled by
an instance of physics and :math:`\regname` is a regularizer. The fixed point iteration takes the form

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
on :math:`\datafidname` and :math:`\regname`, while using additional inputs, such as :math:`A` and :math:`y`, but also
stepsizes, relaxation parameters, etc...

The fStep and gStep classes precisely implement these steps.
In the case of PnP and RED algorithms, the step on the prior term (gStep) contains
the call to the denoiser as per defined by the user (i.e. either :math:`\operatorname{prox}_{\regname}` or
:math:`\nabla \regname` are overwritten by the denoising function).

We refer the reader to the Generic optimizers section of the :ref:`Optim <optim>` documentation for more details on
the different fixed point iterations implemented.

