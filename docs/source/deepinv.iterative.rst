.. _iterative:

Iterative Reconstruction (PnP, RED, etc.)
==================================================


Many image reconstruction algorithms can be shown to be solving
minimization problems of the form

.. math::
    \begin{equation*}
    \label{eq:min_prob}
    \underset{x}{\arg\min} \quad \datafid{x}{y} +  \lambda \reg{x},
    \end{equation*}

where :math:`\datafidname:\xset\times\yset \mapsto \mathbb{R}_{+}` is a data-fidelity term, :math:`\regname:\xset\mapsto \mathbb{R}_{+}`
is a prior term, and :math:`\lambda` is a positive scalar. The data-fidelity term measures the discrepancy between the
reconstruction :math:`x` and the data :math:`y`, and the prior term enforces some prior knowledge on the reconstruction.

PnP and RED algorithms are optimization algorithms where optimization over the prior term is replaced by denoising
operators. When one replaces a proximity operator with a denoiser, the algorithm is called a Plug-and-Play (PnP) algorithm,
whereas when one replaces a gradient step on the prior term with a denoising step, the algorithm is called a
Regularization by Denoising (RED) algorithm.


Implementing an Algorithm
----------------------------------------

Iterative algorithms can be easily implemented using the :ref:`optim module <optim>`, which provides many
pre-implemented :ref:`data-fidelity <data-fidelity>` and :ref:`prior terms <priors>` (including PnP and RED),
as well as many off-the-shelf optimization algorithms.


For example, a PnP proximal gradient descent (PGD) algorithm for
solving the inverse problem :math:`y = \noise{\forw{x}}` reads

.. math::

    \begin{equation*}
    \begin{aligned}
    u_{k} &=  x_k - \gamma \nabla \datafid{x_k}{y} \\
    x_{k+1} &= \denoiser{u_k}{\sigma},
    \end{aligned}
    \end{equation*}


where :math:`f(x)=\frac{1}{2}\|y-\forw{x}\|^2` is a standard data-fidelity term,
and the prior is implicitly defined by a median filter denoiser, can be implemented as follows:


    >>> import deepinv as dinv
    >>> from deepinv.utils import load_url_image
    >>>
    >>> url = ("https://huggingface.co/datasets/deepinv/images/resolve/main/cameraman.png?download=true")
    >>> x = load_url_image(url=url, img_size=512, grayscale=True, device='cpu')
    >>>
    >>> physics = dinv.physics.Inpainting((1, 512, 512), mask = 0.5,
    ...                                    noise_model=dinv.physics.GaussianNoise(sigma=0.01))
    >>>
    >>> data_fidelity = dinv.optim.data_fidelity.L2()
    >>> prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
    >>> model = dinv.optim.optim_builder(iteration="PGD", prior=prior, data_fidelity=data_fidelity,
    ...                                  params_algo={"stepsize": 1.0, "g_param": 0.1})
    >>> y = physics(x)
    >>> x_hat = model(y, physics)
    >>> dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')


.. note::

    While we offer predefined optimization iterators (in this case proximal gradient descent), it is possible to use
    the optim class with any custom optimization algorithm. See the example
    :ref:`sphx_glr_auto_examples_plug-and-play_demo_PnP_custom_optim.py` for more details.


Predefined Iterative Algorithms
----------------------------------------

We also provide pre-implemented iterative optimization algorithms,
which can be loaded in a single line of code, and used
to solve any inverse problem. The following algorithms are available:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.optim.DPIR
   deepinv.optim.EPLL



Deep Image Prior
------------------

The `deep image prior <https://arxiv.org/abs/1711.10925>`_ can be interpreted as
using an indicator function :math:`g(x) = \mathbf {1} _{x=d_{\theta}(z)}` which forces the reconstruction image
to be the output of a convolutional decoder network :math:`d_{\theta}` applied to a random input :math:`z`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DeepImagePrior

The choice of the architecture of :math:`d_{\theta}` is crucial for the success of the method. The
following architecture is available, which is based on a convolutional decoder network:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.ConvDecoder

