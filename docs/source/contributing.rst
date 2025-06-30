:html_theme.sidebar_secondary.remove:

Contributing to DeepInverse
===========================

DeepInverse is a community-driven project and welcomes contributions of all forms.
We are building a comprehensive library of inverse problems and deep learning,
and we need your help to get there!

View our active list of contributors `here <https://github.com/deepinv/deepinv/graphs/contributors>`__.

How to Contribute
-----------------

Make sure that you download all the required dependencies for testing
by running in the root directory of the repository:

.. code-block:: bash

    pip install .[test,dataset,denoisers,doc]


We acknowledge all contributors in the documentation and in the source code. Significant contributions
will also be taken into account when deciding on the authorship of future publications.

Please contribute to ``deepinv`` by forking the `main
repository <https://github.com/deepinv/deepinv/>`_ on GitHub,
then submit a "Pull Request" (PR). When preparing the PR, please make sure to
check the following points:

- **Code quality**: your code is compliant with PEP8 and the `black <https://black.readthedocs.io>`_ style. This can be done easily
  by installing the ``black`` library and running ``black .`` in the root directory of the repository after
  making the desired changes.
- **Typing**: your docstrings and code are adequately typed. Typing rules such as PEP585 are automatically checked using a partial `ruff <https://docs.astral.sh/ruff/>`_ checker.
- **Tests**: write tests in ``deepinv/tests`` to test your code's intended functionality,
  including unit tests (e.g. checking each method's return values) and integration tests (i.e. end-to-end behaviour),
  following a test-driven development methodology. We use ``pytest`` and ``unittest.mock`` to write our tests.
  All existing tests should pass on your local machine. This can be done by installing ``pytest`` and running
  ``python -m pytest deepinv/tests`` in the root directory of the repository after making the desired changes.
  Learn more `here <https://realpython.com/pytest-python-testing/>`__.
  Your code coverage will automatically be checked using ``codecov``.
- **Docs**: the documentation is updated if necessary. Our documentation is written in `reST <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ 
  and built with `sphinx <https://www.sphinx-doc.org/en/master/usage/installation.html>`_.
  After making the desired changes, check the documentation by installing
  ``sphinx`` and building the docs by running one of the commands in the table below in the ``docs`` directory.
  Note that if the build process fails, supplementary additional libraries may need to be manually installed
  (e.g. ``sphinx-gallery``): please follow instructions in the log.

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

Finding Help
------------

If you are not familiar with the GitHub contribution workflow, you can also open an issue on the
`issue tracker <https://github.com/deepinv/deepinv/issues>`_ and also ask any question in our discord server
`Discord server <https://discord.gg/qBqY5jKw3p>`_. We will then try to address the issue as soon as possible.
You can also send an email to any of the `maintainers <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_ with your questions or ideas.

Learn more about our code philosophy in the paper:
`DeepInverse: A Python package for solving imaging inverse problems with deep learning <https://arxiv.org/abs/2505.20160>`_.

Contributing new datasets
--------------------------

In order to contribute a new dataset, you must provide tests alongisde it to check that it functions as expected. The DeepInverse code base is regularly tested on automatic continuous integration (CI) servers in order to ensure that the code works the way it is supposed to. Unfortunately, the CI servers have limited resources and they can generally not host the datasets.

We get around this by mocking datasets in the tests. First, write the tests and the implementation, and make sure that the tests pass locally, on the real data. Then, write `mocking code <https://en.wikipedia.org/wiki/Mock_object>`_, code that intercepts calls to input/output (IO) related functions, e.g. `os.listdir`, and make them return a hard coded value, thereby making execution go as if the data was there. For more details and examples, see `this pull request <https://github.com/deepinv/deepinv/pull/490>`_.

Once the implementation, the tests and the mocking code are written, that they pass locally and on the CI servers, the maintainers will be able to review the code and merge it into the main branch if everything goes well. You should bear in mind though that the maintainers won't have the time to make sure the tests pass on the real data, so they will have to trust that you did things correctly.
