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


Contributing new datasets
----------------

Contributing a new dataset can be a little trickier than other contributions, here are a few guidelines to help you with that. The DeepInverse code base is regularly tested on automatic continuous integration (CI) servers in order to ensure that the code works the way it is supposed to. Unfortunately, the CI servers have limited resources and they can generally not host the datasets. This makes testing the implementation of the datasets a bit cumbersome as only the logic can be tested and not the consistency with the actual data.

This does not mean that you should not contribute datasets, they are really valuable and make the library more complete! It does not mean that you can simply skip testing the implementation either. Our current guidelines is to first write the tests and the implementation and to make sure that the tests pass locally, on the real data. As it is not possible to do so on the CI servers, the next step is to write mocking code, code that intercepts calls to input/output (IO) related functions, e.g. `os.listdir`, and make them return a hard coded value, thereby making execution go as if the data was there.

Once the implementation, the tests and the mocking code are written, that they pass locally and on the CI servers, the maintainers will be able to review the code and merge it into the main branch if everything goes well. You should bear in mind though that the maintainers won't have the time to make sure the tests pass on the real data, so they will have to trust that you did things correctly.
