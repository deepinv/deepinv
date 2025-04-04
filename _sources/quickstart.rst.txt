:html_theme.sidebar_secondary.remove:

Quickstart
==========

Install the latest stable version of ``deepinv`` via pip:

.. code-block:: bash

    pip install deepinv

or install the latest version of ``deepinv`` directly from the github repository:

.. code-block:: bash

    pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv

Since ``deepinv`` is under active development, you can update to the latest version easily using:

.. code-block:: bash

    pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv


Once installed, you can try out the following image inpainting example:

.. doctest::

    >>> import deepinv as dinv
    >>> from deepinv.utils import load_url_image
    >>> url = ("https://huggingface.co/datasets/deepinv/images/resolve/main/cameraman.png?download=true")
    >>> x = load_url_image(url=url, img_size=512, grayscale=True, device='cpu')
    >>> physics = dinv.physics.Inpainting((1, 512, 512), mask = 0.5,
    ...     noise_model=dinv.physics.GaussianNoise(sigma=0.01))
    >>> data_fidelity = dinv.optim.data_fidelity.L2()
    >>> prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
    >>> model = dinv.optim.optim_builder(iteration="HQS", prior=prior, data_fidelity=data_fidelity,
    ...                             params_algo={"stepsize": 1.0, "g_param": 0.1})
    >>> y = physics(x)
    >>> x_hat = model(y, physics)
    >>> dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')


You are now ready to try further examples or experiment yourself using our comprehensive :ref:`User Guide <user_guide>`!