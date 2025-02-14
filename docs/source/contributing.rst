:html_theme.sidebar_secondary.remove:

How to Contribute
=================


DeepInverse is a community-driven project and welcomes contributions of all forms.
We are ultimately aiming for a comprehensive library of inverse problems and deep learning,
and we need your help to get there!

Make sure that you download all the required dependencies for testing
by running in the root directory of the repository:

.. code-block:: bash

    pip install .[test,dataset,denoisers,doc]


We will acknowledge all contributors in the documentation and in the source code. Significant contributions
will also be taken into account when deciding on the authorship of future publications.

The preferred way to contribute to ``deepinv`` is to fork the `main
repository <https://github.com/deepinv/deepinv/>`_ on GitHub,
then submit a "Pull Request" (PR). When preparing the PR, please make sure to
check the following points:

- The code is compliant with the `black <https://github.com/psf/black>`_ style. This can be done easily
  by installing the black library and running ``black .`` in the root directory of the repository after
  making the desired changes.
- The automatic tests pass on your local machine. This can be done by running ``python -m pytest deepinv/tests``
  in the root directory of the repository after making the desired changes.
- The documentation is updated if necessary.
  After making the desired changes, this can be done in the directory
  ``docs`` by running one of the commands in the table below.

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

If you are not familiar with the GitHub contribution workflow, you can also open an issue on the
`issue tracker <https://github.com/deepinv/deepinv/issues>`_ and also ask any question in our discord server
`Discord server <https://discord.gg/qBqY5jKw3p>`_. We will then try to address the issue as soon as possible.
You can also send an email to any of the main developers with your questions or ideas.



