.. _contributing:

Contributing to DeepInverse
===========================

DeepInverse is a community-driven project and welcomes contributions of all forms.
We are building a comprehensive library of inverse problems and deep learning,
and we need your help to get there!

View our active list of contributors `here <https://github.com/deepinv/deepinv/graphs/contributors>`__.
We acknowledge all contributors!
Significant contributions will also be taken into account when deciding on the authorship of future publications.

Learn more about our code philosophy in the paper:
`DeepInverse: A Python package for solving imaging inverse problems with deep learning <https://arxiv.org/abs/2505.20160>`_.

Finding Issues to Work On
-------------------------

We welcome contributions in all areas!
Get started by looking for
`good first issue <https://github.com/deepinv/deepinv/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22>`_ or
`open to contribs <https://github.com/deepinv/deepinv/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22open%20to%20contribs%22>`_.
Or, to help you find something interesting or relevant to your
expertise, have a search in our `issues <https://github.com/deepinv/deepinv/issues>`_. Here are some keywords you could search for:

.. list-table::
   :widths: 25 25 25 25

   * - `optimization <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+optim>`_ 
     - `training <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+trainer>`_ 
     - `datasets <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+dataset>`_ 
     - `losses <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+loss>`_ 
   * - `diffusion <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+diffusion>`_ 
     - `mri <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+mri>`_ 
     - `tomography <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+tomography>`_ 
     - `docs <https://github.com/deepinv/deepinv/issues?q=is%3Aissue+state%3Aopen+docs>`_ 


How to Contribute
-----------------

Want to solve an issue or contribute something new to DeepInverse? Never contributed to DeepInverse before? Here's a step-by-step with the basics!

.. tip::
  Need help? Ask in `Discord <https://discord.gg/qBqY5jKw3p>`_, open an `issue <https://github.com/deepinv/deepinv/issues>`_, or find a `maintainer <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.

1. Fork DeepInverse and write your code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first steps of contributing to any open-source project is the same. Follow these `step-by-step instructions on the GitHub website <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_
to: fork your own copy of `DeepInverse <https://github.com/deepinv/deepinv>`_, clone it to your computer, create a branch, write code, commit and push your code.

Once you've opened a (draft) pull request (PR) in GitHub with your contribution, you should be able to see it under `Pull Requests <https://github.com/deepinv/deepinv/pulls>`_.
You're ready to move on!

.. note::

  Our maintainers will then try to assist you by working directly on your PR. Do not hesitate to ask questions or to leave comments directly on the Pull Request page. 

2. Write tests
~~~~~~~~~~~~~~

Tests are crucial for checking your code will always behave as intended, and we encourage you to follow a test-driven development methodology. Tests can consist of:

- Unit tests (e.g. check each method's return values and shapes);
- Integration tests (e.g. end-to-end behavior, interface with other classes).

How to write and run tests:

1. Install `deepinv` in editable mode so that all your changes are used when you run code:

.. code-block:: bash

    pip install -e .[test,dataset,denoisers,doc,lint,training]

2. Write your tests in ``deepinv/tests``. Check out the existing tests to see examples of where you could insert your tests. We use ``pytest`` and ``unittest.mock``.

.. hint::

  If you've contributed a new class (e.g. dataset, physics etc.), you should add it to any existing tests, e.g. those that check physics adjointness, dataset return format, etc.

1. Check your tests pass locally by running ``python -m pytest deepinv/tests`` in the root directory after making the desired changes. Learn more `here <https://realpython.com/pytest-python-testing/>`__. You can also run specific tests by providing the path to the test file, e.g. ``python -m pytest deepinv/tests/test_physics.py``, or even to a specific test function, e.g. ``python -m pytest -k "test_operators_adjointness" deepinv/tests/test_physics.py``.
2. Push your code to your PR. A maintainer will run the tests on CPU and GPU in the CI, and you will see the results in the `Test PR...` GitHub action.

.. note::
  Your code coverage will automatically be checked using ``codecov``.

.. tip::
  **Run into a problem**? Ask in `Discord <https://discord.gg/qBqY5jKw3p>`_ and we'll help you out.

3. Write docs
~~~~~~~~~~~~~

Writing good documentation is also crucial for helping other users use your code. This is how:

1. Write good quality `docstrings <https://realpython.com/how-to-write-docstrings-in-python/>`_ for each new class, method or function. Have a look at any other class or method in DeepInverse to see examples! Please follow our :ref:`docstring guidelines below <docstring_guidelines>`.
2. If you wrote a new class or function, add it to the lists in the `API reference <https://deepinv.github.io/deepinv/API.html>`_ and `User Guide <https://deepinv.github.io/deepinv/user_guide.html>`_. For API, add to the appropriate `.rst` file `here <https://github.com/deepinv/deepinv/tree/main/docs/source/api>`__. For User Guide, `here <https://github.com/deepinv/deepinv/tree/main/docs/source/user_guide>`__.
3. Want to share more about your new feature? Consider writing an `example <https://deepinv.github.io/deepinv/auto_examples/index.html>`_ in `examples/`!
4. Check that your documentation is correct by building the docs locally. First `cd docs`, then we use `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_ to build:
  
.. list-table::
   :widths: 40 50
   :header-rows: 1

   * - Command
     - Description of command
   * - ``make html``
     - Generates all the documentation
   * - ``make html-fast``
     - Generates documentation faster but without running the examples
   * - ``PATTERN=/path/to/file make html-pattern``
     - Generates documentation for files matching ``$(PATTERN)``
   * - ``make clean``
     - Cleans the documentation files
   * - ``make clean-win``
     - Cleans the documentation files (Windows OS)

.. caution:: 
  Note that if the build process fails, supplementary additional libraries may need to be manually installed (e.g. ``sphinx-gallery``): please follow instructions in the log.

.. tip::
  If the `Build Docs` GitHub action has run successfully, you can download the documentation as a zip file from the `Actions page <https://github.com/deepinv/deepinv/actions/workflows/documentation.yml>`_. Look for the workflow run corresponding to your pull request.


4. Code quality
~~~~~~~~~~~~~~~

Code quality is important to us. We require that your code is compliant with PEP8, the `black <https://black.readthedocs.io>`_ style and `ruff <https://docs.astral.sh/ruff/>`_ checkers:

1. Add `typing <https://fastapi.tiangolo.com/python-types/>`_ to your code and docstrings. Typing rules such as PEP585 are automatically checked using ruff.
2. Run ``black .`` in the root directory of your repository. This will automatically fix all formatting issues.
3. Run ``ruff check``, which will check all linting options we've enabled. If it fails, follow the suggestions to make a fix!
4. Push your code. The automatic checkers will run in GitHub actions, along with other actions that we have in place.

5. Interact with reviewers
~~~~~~~~~~~~~~~~~~~~~~~~~~

You're done! A maintainer will see your PR and will interact with you. They may suggest changes. It is your responsibility to make all requested fixes!

.. note::

  A maintainer may directly edit your code if appropriate. Make sure to `git pull` to integrate these changes locally.

Finding help
~~~~~~~~~~~~

.. tip::

  **Run into a problem, don't know where to start, or got a question/suggestion?**
  
  Ask in `Discord <https://discord.gg/qBqY5jKw3p>`_, open an `issue <https://github.com/deepinv/deepinv/issues>`_, or 
  send an email to a `maintainer <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_ and we'll help you out.

.. _docstring_guidelines:

Docstring Guidelines
--------------------

For class and function docstrings, we use the **reStructuredText (reST)** syntax.
See the `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ for more details.

Please follow these guidelines:

- Each parameter and return must be properly described, along with a type annotations for each ``:param`` field, as shown below:

  .. code-block:: rest

      :param <type> <name>: Description of the parameter.
      :return: Description of the return value.

- Docstrings can be split into multiple sections using the horizontal separator ``|sep|``, with section titles introduced by ``:Title:``.

- To provide usage examples, include an ``:Example:`` section. Code in this section will be executed during documentation generation.

- Use ``:math:`` for inline LaTeX-style mathematics, and ``.. math::`` for block equations.

- To include remarks, warnings, or tips, use the ``.. note::`` directive.

- To cite a paper:

  1. Add the BibTeX entry to the ``refs.bib`` file.
  2. Use ``:footcite:t:`<key>``` to cite in the format *Author et al. [1]*.
  3. Use ``:footcite:p:`<key>``` to cite with only the reference number *[1]*.

  For details on citing references with Sphinx, see the `sphinx-bibtex documentation <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_.

  All references will be compiled and listed automatically in the generated documentation.

Below is a minimal working example of a typical docstring that includes all these features:



.. code-block:: python

    class MyDenoiser:
        r"""
        Denoiser denoiser from the paper :footcite:t:`my_paper`.

        .. math::
            y = \D_\sigma{x + \sigma \omega}

        .. note::
            This is a note.

        |sep|

        :Example:

        >>> import torch
        >>> import deepinv as dinv
        >>> model = dinv.models.DRUNet()
        >>> x = torch.ones((1, 1, 8, 8))
        >>> y = model(x)

        :param int in_channels: number of input channels.
        :param int out_channels: number of output channels.
        :param str pretrained: path to pretrained weights or 'download'.
        """
        def __init__(self, in_channels: int, out_channels: int, pretrained: bool = None):
            pass

Contributing new physics
-------------------------

Providing new physics is one of the most impactful contributions you can make to DeepInverse, as it will allow users to solve new inverse problems. The process for contributing a new physics is similar to what is described above with some specificities:

- writing a new class that inherits from the appropriate physics base class, you can follow the basic design described in `Bring your own physics <https://deepinv.github.io/deepinv/auto_examples/basics/demo_custom_physics.html>`_
  
- registering your physics in the appropriate test suite and check that the tests are passing -- when inheriting from :class:`deepinv.physics.LinearPhysics` it involves the following modifications to `deepinv/tests/test_physics.py`:

  1. add a new entry corresponding to your physics configuration to the list variable ``OPERATORS``
  2. specify how to instantiate this configuration in the function ``find_operator``
  3. if applicable, write the tests specific to your physics, e.g. if it has a specific behavior that is not covered by the existing tests, see `test_MRI` in `here <https://github.com/deepinv/deepinv/blob/main/deepinv/tests/test_physics.py>`_ for an example

- adding your physics to the `API reference <https://deepinv.github.io/deepinv/api/deepinv.physics.html>`__ and `User Guide <https://deepinv.github.io/deepinv/user_guide/physics/physics.html>`__ as described in the previous section.

Here are some pull requests that you can refer to for examples of how to contribute new physics:

- :class:`deepinv.physics.Scattering` in `#1020 <https://github.com/deepinv/deepinv/pull/1020>`_

- :class:`deepinv.physics.SpatialUnwrapping` in `#723 <https://github.com/deepinv/deepinv/pull/723>`_

- :class:`deepinv.physics.TomographyWithAstra` in `#474 <https://github.com/deepinv/deepinv/pull/474>`_

Contributing new datasets
--------------------------

In order to contribute a new dataset, you must provide tests alongisde it to check that it functions as expected. The DeepInverse code base is regularly tested on automatic continuous integration (CI) servers in order to ensure that the code works the way it is supposed to. Unfortunately, the CI servers have limited resources and they can generally not host the datasets.

We get around this by mocking datasets in the tests. First, write the tests and the implementation, and make sure that the tests pass locally, on the real data. Then, write `mocking code <https://en.wikipedia.org/wiki/Mock_object>`_, code that intercepts calls to input/output (IO) related functions, e.g. `os.listdir`, and make them return a hard coded value, thereby making execution go as if the data was there. For more details and examples, see `this pull request <https://github.com/deepinv/deepinv/pull/490>`_.

Once the implementation, the tests and the mocking code are written, that they pass locally and on the CI servers, the maintainers will be able to review the code and merge it into the main branch if everything goes well. You should bear in mind though that the maintainers won't have the time to make sure the tests pass on the real data, so they will have to trust that you did things correctly.


Maintainers commands
--------------------

Maintainers can use the following slash commands as comments on a pull request to trigger specific tests:

- `/test-examples`: runs **all** sphinx gallery examples on CPU using CPU-enabled runners.
- `/gpu-tests`: runs tests and generates docs on GPU using GPU-enabled runners.
