.. _unfolded:

Unfolded Algorithms
===================

The :ref:`optimization module <optim>` module also be turned into an unfolded trainable architecture.

Unfolded architectures (sometimes called 'unrolled architectures') are obtained by replacing parts of these algorithms
by learnable modules. In turn, they can be trained in an end-to-end fashion to solve inverse problems.
It suffices to set the argument ``unfold=True`` when creating an optimization algorithm from the :ref:`optimization <optim>` module.
By default, if a neural network is used in place of a regularization or data-fidelity step, the parameters of the network are learnable by default.
Moreover, among all the parameters of the algorithm (e.g. step size, regularization parameter, etc.), the use can use which are learnable and which are not via the argument ``trainable_params``.

In the following example, we create an unfolded architecture of 5 proximal gradient steps
using a DnCNN plug-and-play prior a standard L2 data-fidelity term. The network can be trained end-to-end, and
evaluated with any forward model (e.g., denoising, deconvolution, inpainting, etc.). 
Here, the stepsize ``stepsize``, the regularization parameter ``lambda_reg``, and the prior parameter ``g_param`` of the plug-and-play prior are learnable.

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> from deepinv.optim import ProximalGradientDescent
    >>>
    >>> # Create a trainable unfolded architecture
    >>> model = ProximalGradientDescent(  # doctest: +IGNORE_RESULT
    ...     iteration="PGD",
    ...     unfold=True,
    ...     data_fidelity=dinv.optim.L2(),
    ...     prior=dinv.optim.PnP(dinv.models.DnCNN()),
    ...     stepsize=1.0,
    ...     g_param=0.1,
    ...     lambda_reg=1,
    ...     max_iter=5,
    ...     trainable_params=["stepsize", "g_param", "lambda_reg"]
    ... )
    >>> # Forward pass
    >>> x = torch.randn(1, 3, 16, 16)
    >>> physics = dinv.physics.Denoising()
    >>> y = physics(x)
    >>> x_hat = model(y, physics)


.. _deep-equilibrium:

Deep Equilibrium
----------------
Deep Equilibrium models (DEQ) are a particular class of unfolded architectures where the backward pass
is performed via Fixed-Point iterations. DEQ algorithms can virtually unroll infinitely many layers leveraging
the **implicit function theorem**. The backward pass consists in looking for solutions of the fixed-point equation

.. math::

   v = \left(\frac{\partial \operatorname{FixedPoint}(x^\star)}{\partial x^\star} \right)^{\top} v + u.


where :math:`u` is the incoming gradient from the backward pass,
and :math:`x^\star` is the equilibrium point of the forward pass.
See `this tutorial <http://implicit-layers-tutorial.org/deep_equilibrium_models/>`_ for more details.

For turning an optimization algorithm into a DEQ model, it suffices to set the argument ``DEQ=True`` when creating an optimization algorithm from the :ref:`optimization <optim>` module.

.. _predefined-unfolded:

Predefined Unfolded Architectures
---------------------------------
We also provide some off-the-shelf unfolded network architectures,
taken from the respective literatures.

.. list-table:: Predefined unfolded architectures
   :header-rows: 1

   * - Model
     - Description
   * - :class:`deepinv.models.VarNet`
     - VarNet/E2E-VarNet MRI reconstruction models
   * - :class:`deepinv.models.MoDL`
     - MoDL MRI reconstruction model

.. _custom-unfolded-blocks:

Predefined Unfolded Blocks
--------------------------
Some more specific unfolded architectures are also available.

The Primal-Dual Network (PDNet) uses :class:`deepinv.models.PDNet_PrimalBlock` and
:class:`deepinv.models.PDNet_DualBlock` as building blocks for the primal and dual steps respectively.