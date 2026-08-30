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

.. _step_by_step_contribute:

Step-by-step: How to Contribute
-------------------------------

Want to solve an issue or contribute something new to DeepInverse? Never contributed to DeepInverse before? Here's a step-by-step with the basics!

.. tip::
  Need help? Ask in `Discord <https://discord.gg/qBqY5jKw3p>`_, open an `issue <https://github.com/deepinv/deepinv/issues>`_, or find a `maintainer <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.

1. Fork DeepInverse and write your code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first steps of contributing to any open-source project is the same. Follow these `step-by-step instructions on the GitHub website <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_
to: fork your own copy of `DeepInverse <https://github.com/deepinv/deepinv>`_, clone it to your computer, create a branch, write code, commit and push your code.
Make sure you follow the DeepInverse :ref:`style guide <style_guides>`, so that your contribution maintains our high standard of code and docs.

.. note::
  LLM usage is ok, but for first-time contributors, we request that their contributions are mainly human-written and will not accept PRs generated 100% by AI. See our :ref:`LLM policy <llm-policy>`.

Once you've opened a (draft) pull request (PR) in GitHub with your contribution, you should be able to see it under `Pull Requests <https://github.com/deepinv/deepinv/pulls>`_.
You're ready to move on!

.. note::

  Our maintainers will then try to assist you by working directly on your PR. Do not hesitate to ask questions or to leave comments directly on the Pull Request page.

2. Install DeepInverse
~~~~~~~~~~~~~~~~~~~~~~

From the root of your cloned repository, install ``deepinv`` in editable mode so
that your local changes are used when you run code. Choose one of the following
installation methods.

Some contributions require software beyond Python packages. This is, for
instance, the case for :class:`PET physics <deepinv.physics.PET>`. 
If you want to contribute related features, use the repository's full `Pixi <https://pixi.sh>`_ development environment:

.. code-block:: bash

    pixi install -e full

Run commands in this environment with ``pixi run -e full``, for example
``pixi run -e full python -m pytest deepinv/tests``.

For contributions requiring only Python packages, `uv <https://docs.astral.sh/uv/>`_
can create a virtual environment and install the development dependencies:

.. code-block:: bash

    uv venv
    uv pip install -e ".[test,dataset,denoisers,doc,lint,training]"

Run commands in this environment with ``uv run --no-sync``, for example
``uv run --no-sync python -m pytest deepinv/tests``.

Alternatively, use Python's built-in ``venv`` module and ``pip``:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[test,dataset,denoisers,doc,lint,training]"

.. _write_tests:

3. Write tests
~~~~~~~~~~~~~~

Tests are crucial for checking your code will always behave as intended, and we encourage you to follow a test-driven development methodology. Tests can consist of:

- Unit tests (e.g. check each method's return values and shapes);
- Performance tests (e.g. an algorithm performs as well as expected on a dataset, it converges etc.);
- Integration tests (e.g. end-to-end behavior, interface with other classes).

How to write and run tests:

1. Write your tests in ``deepinv/tests``. Check out the existing tests to see examples of where you could insert your tests. We use ``pytest`` and ``unittest.mock``.

.. hint::

  If you've contributed a new class (e.g. dataset, physics etc.), you should add it to any existing tests, e.g. those that check physics adjointness, dataset return format, etc.

2. Check your tests pass locally by running ``python -m pytest deepinv/tests`` in the root directory after making the desired changes. Learn more `here <https://realpython.com/pytest-python-testing/>`__. You can also run specific tests by providing the path to the test file, e.g. ``python -m pytest deepinv/tests/test_physics.py``, or even to a specific test function, e.g. ``python -m pytest -k "test_operators_adjointness" deepinv/tests/test_physics.py``.
3. Push your code to your PR. A maintainer will run the tests on CPU and GPU in the CI, and you will see the results in the `Test PR...` GitHub action.

.. note::
  Your code coverage will automatically be checked using ``codecov``.

.. tip::
  **Run into a problem**? Ask in `Discord <https://discord.gg/qBqY5jKw3p>`_ and we'll help you out.

.. _write_docs:

4. Write docs
~~~~~~~~~~~~~

Writing good documentation is also crucial for helping other users use your code. This is how:

1. Write good quality `docstrings <https://realpython.com/how-to-write-docstrings-in-python/>`_ for each new class, method or function. Have a look at any other class or method in DeepInverse to see examples! Please follow our :ref:`docstring guidelines below <docstring_guidelines>`.
2. If you wrote a new class or function, add it to the lists in the `API reference <https://deepinv.org/API.html>`_ and `User Guide <https://deepinv.org/user_guide.html>`_. For API, add to the appropriate `.rst` file `here <https://github.com/deepinv/deepinv/tree/main/docs/source/api>`__. For User Guide, `here <https://github.com/deepinv/deepinv/tree/main/docs/source/user_guide>`__.
3. Want to share more about your new feature? Consider writing an `example <https://deepinv.org/auto_examples/index.html>`_ in `examples/`!
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
  If the `Build Docs` GitHub action has run successfully, you can download the documentation as a zip file from the `Actions page <https://github.com/deepinv/deepinv/actions/workflows/docs_cpu.yml>`_. Look for the workflow run corresponding to your pull request.

.. _code_quality:

5. Code quality
~~~~~~~~~~~~~~~

Code quality is important to us. We require that your code is compliant with PEP8, the `black <https://black.readthedocs.io>`_ style and `ruff <https://docs.astral.sh/ruff/>`_ checkers:

1. Add `typing <https://fastapi.tiangolo.com/python-types/>`_ to your code and docstrings. Typing rules such as PEP585 are automatically checked using ruff.
2. Run ``black .`` in the root directory of your repository. This will automatically fix all formatting issues.
3. Run ``ruff check``, which will check all linting options we've enabled. If it fails, follow the suggestions to make a fix!
4. Push your code. The automatic checkers will run in GitHub actions, along with other actions that we have in place.
5. Ensure you follow our :ref:`style guide <code_quality_guide>`

Alternatively, you can install `pre-commit <https://pre-commit.com/>`_ with:

.. code-block:: bash

    pip install pre-commit
    pre-commit install

This runs `ruff` and `black` alongside other tests every time you create a commit.

.. _log_changes:

6. Log your changes
~~~~~~~~~~~~~~~~~~~

We keep a summary of all changes in the `changelog.rst <https://deepinv.org/changelog.html>`_ file in the documentation.
We separate contributions into three categories: **Added** for new features, **Changed** for changes in existing features, and **Fixed** for bug fixes.
To do so, you should first add your GitHub information at the end of the file following the format:

.. code-block:: rest

  .. _<your name>: https://github.com/<your GitHub username>

You can then add a line to the appropriate category under the **Current** section, describing your contribution in a concise way.
This line should follow the format:

.. code-block:: rest

  - <description of your contribution> (:gh:`<pull request number>` by `<your name>`_)


You also need to summarise your changes in the Pull Request description, and tick whether you used LLM tools to generate the code. See :ref:`LLM policy <llm-policy>` for more details.

7. Interact with reviewers
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

.. _llm-policy:

LLM policy
~~~~~~~~~~

DeepInverse allows contributions where code is partially written by an LLM. However, we require that a human contributes both during code writing, and during PR submission and review. Why?

1. Community: easy-fixable issues are for humans, not bots, and humans keep the project alive;
2. Review: PR review works because reviewers trust authors and their intentions, and don't always have to resort to line-by-line reviews of code that no human has read before.

Therefore, DeepInverse does not welcome PRs a) consisting fully of LLM-generated code, or b) that are submitted by an AI agent, or an agent acting on behalf of a human, especially for first time contributors. DeepInverse maintainers may close a PR if they suspect that the PR is AI-generated. Therefore, to help maintainers trust that you are a human coder, we request that, when submitting a PR, you tick whether an LLM or AI agent helped you write the code, or generated it fully, and/or submitted the PR.

.. _style_guides:

Contributing style guides
-------------------------

The DeepInverse community maintains a high, opinionated standard of code and documentation in order to provide a didactic library that leads the field of imaging, rather than a collection of code files.
All contributors are responsible for using their human judgement to uphold this standard, which is especially important in the era of LLM coding.
The purpose of this style guide is to help any devs (experienced, new, maintainers, LLMs, agents) always stick with best practice while contributing **and** reviewing.

.. _docstring_guidelines:

Docstring Guidelines
~~~~~~~~~~~~~~~~~~~~

For class and function docstrings, we use the **reStructuredText (reST)** syntax.
See the `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ for more details.
Please follow these guidelines:

- Begin with a concise one-line summary of what the code does.

- Describe exactly what the user needs to know in order to use the function/class, and why they should use it.

- Only include technical implementation details that help the user understand how to use the code, put them lower in the docstring. Omit details that are not immediately important.

- Write docs extremely concisely, and prioritise readability over completeness. State facts once only and don't repeat points. Write in concise technical English, not prose. Don't break lines unnecessarily. For example:

      Acquisition angles in degrees. Returns ``None`` for vector-based geometries, for which the acquisition trajectory is fully described by `self.projection_geometry["Vectors"]`.

  can be much more concisely written as:

      Astra projection geometry angles tensor in degrees. If Astra vector geom, return None.

- Only write inline comments where absolutely necessary, where it is non-trivial to understand code behaviour.

- Properly describe each parameter and return, along with a type annotations for each `:param` field, as shown below:

  .. code-block:: rest

      :param <type> <name>: Description of the parameter. Keep inline to aid readability. Add default value if not obvious from the func/class signature.
      :return: Description of the return value.

- Split docstrings into multiple sections using the horizontal separator `|sep|`, and introduce section titles with `:Title:`.

- To provide usage examples, include an `:Example:` section. Code in this section will be executed during documentation generation.

- Use `:math:` for inline LaTeX-style mathematics, and `.. math::` for block equations.

- To include remarks, warnings, or tips, use the `.. note::`, `.. warning::` or `.. tip::`  directives.

- Link objects with Sphinx roles such as `:class:`, `:func:`, `:meth:`, and `:ref:`.

- Use single tick marks ` for inline code.

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
        Denoiser from the paper :footcite:t:`my_paper`.

        .. math::
            y = \D_\sigma{x + \sigma \omega}

        .. note::
            This is a note.

        |sep|

        :Example:

        >>> import torch
        >>> import deepinv as dinv
        >>> model = dinv.models.DRUNet()
        >>> x = torch.ones((1, 3, 8, 8))
        >>> y = model(x, sigma=0.01)

        :param int in_channels: number of input channels.
        :param int out_channels: number of output channels.
        :param str pretrained: path to pretrained weights or 'download'.
        """
        def __init__(self, in_channels: int, out_channels: int, pretrained: bool = None):
            pass

**Guidelines are similar for examples.** A good example should teach the user why and how they should use a particular functionality.

- If you introduce a reconstruction method, detail the algorithm, cite it and link to the original code;
- Reuse existing DeepInverse loaders, transforms, solvers, plotting functions, and example assets as much as possible;
- Examples should be clear and concise: avoid copy-pasting text from other examples, go straight to the point.
- Do not introduce unnecessary classes or functions unless necessary, and avoid large number of lines for boilerplate. Prioritise linear readability of the function.

For example, here's a first paragraph from a **bad** example:


  Many reconstruction models expose a hyperparameter that has to be matched to the problem at hand:
  a denoiser needs the noise level :math:`\sigma`, a deblurring model needs the blur kernel.
  In a benchmark this parameter is easy to pick,
  because the ground truth :math:`x` is available and we can simply maximize the PSNR. In a real
  deployment the ground truth is precisely what we are trying to recover, so that criterion is not
  available and the parameter is usually left at a hand-tuned default.

It is bad because it does not immediately tell the user why they should use this example, and how they would use it.
It is also unnecessarily verbose and includes phrases that are not immediately important. 
It also does not relate to existing examples already in DeepInverse.
Furthermore, a technical subtlety is that physics parameters are more clearly explained by how they relate to physics, not the denoisers.
This is a DeepInverse opinionation that this paragraph has conveyed.
Here's a better version:


  In blind inverse problems, one often needs to estimate physics parameters such as noise level :math:`\sigma` or blur kernel.
  When ground truth :math:`x` is available, we can simply maximize the PSNR: see :ref:`sphx_glr_auto_examples_blind-inverse-problems_demo_optimizing_physics_parameter.py` for an example.
  However, in real applications :math:`x` is not available. 

.. _code_quality_guide:

Code quality
~~~~~~~~~~~~

- **Naming**: classes use ``CapWords`` without underscores. Functions, methods, parameters, and variables use ``snake_case``. Private helpers and implementation details begin with ``_``.
- **Cleanliness**: no global constants unless absolutely necessary; no unnecessary private single-use funcs; inline comments should only be used for describing non-obvious functionality, pitfalls or edge cases.
- **Typing**: use concise modern typing as much as possible, but prefer readability over completeness. Where adding typing or more specific types would hurt readability, don't add it;
- **Tests**: prefer adding a method / class to an existing generic test through a registry rather than a new standalone test;
- **Tests**: the purpose of tests is that it checks the code does what it claims, not that the code executes without raising errors.
  Therefore, adding an optimisation method requires to check that it converges to the limit point; adding a neural network from an external library should check metric on a dataset, etc. Limit cases should be added and checked (e.g. non standard tensor shapes, etc).
  Note that in most cases, such checks are already implemented (see above), but it's to the user to check that sufficient tests are checked.

Code scope
~~~~~~~~~~

- Implement the minimal change that fully satisfies the behavior described in a Pull Request (PR) description or an issue;
- **Pull Request length**: keep PRs under 3000 lines, especially for first-time contributors. Instead, submit large feature contributions incrementally.

General technical details
~~~~~~~~~~~~~~~~~~~~~~~~~

- Reuse existing abstractions in DeepInverse as much as possible to help modularity;
- Anticipate potential future abstractions. For instance, if you propose a modification for some specific 1D application, try to open to door to 1D globally;
- Avoid adding new dependencies unless the feature genuinely requires it. Import optional dependencies with a try-except block, with a message `to use ..., x is required. Install it using...`;
- Preserve backward compatibility and avoid breaking changes;
- Cite code whenever possible; see citation instructions :ref:`above <docstring_guidelines>`;
- When copying code from external codebases, seek permission from the original author and include a `third-party licence <https://github.com/deepinv/deepinv/tree/main/deepinv/models/third_party>`_;
- `nn.Modules` like physics should not possess device and dtype attributes; only tensors, parameters and buffers have devices and dtypes;
- No mutable default arguments such as dicts;
- Every operation should be batched.
- If you propose a new technical convention, add it to this list so that future contributors and reviewers don't forget it!

.. _backward_compatibility:

Backwards Compatibility
~~~~~~~~~~~~~~~~~~~~~~~

If you propose breaking changes, you must prevent your contribution breaking existing user workflows. To do this:

- Start by deprecating the former behavior with an opt-in way to switch to the new behavior.
- Keep the old functionality as default, and add a deprecation notice in the docstring.
- Also add a deprecation warning in the code using one of our existing deprecation helpers in ``deepinv/utils/decorators.py``:

  - ``_deprecated_class``: deprecate a class.
  - ``_deprecated_func``, ``_deprecated_func_replaced_by``: deprecate a function/method.
  - ``_deprecated_argument``, ``_deprecated_alias``: deprecate an argument of a function/method.
  - ``_deprecate_attribute``: deprecate an attribute.

- After a delay deemed sufficient, finally drop support for the deprecated feature. 
- Update :ref:`the changelog <log_changes>` at both stages with the new deprecations and dropped features in order to help users with the migration process.

Even though we generally try to avoid unexpected breaking changes, the library is at an early stage of development and we tolerate them in certain cases. Specifically, we allow them when the benefits are considered to far outweigh the negative consequences, especially when proper deprecation would take a lot more effort than the change itself.

As a contributor making a new pull request, it might be tricky to determine a suitable way to handle potential breaking changes. Please do not let this delay your submission needlessly. The maintainers acknowledge this and will provide the necessary guidance when reviewing your changes.

We also generally endorse the recommendations from `scikit-learn's contributing guide <https://scikit-learn.org/dev/developers/contributing.html#maintaining-backwards-compatibility>`_.

Contributing new physics
~~~~~~~~~~~~~~~~~~~~~~~~

Adding a physical operator follows the general contribution guidelines. Specifically, your contribution must include proper :ref:`tests <write_tests>` and :ref:`documentation <write_docs>`, as well as meet our :ref:`code quality standards <code_quality>`. Additionally, the provided code is expected to follow specific design rules to ensure seamless integration into the codebase, this means:

- Implementing a new class that inherits from the appropriate physics base class. Refer to the design outlined in `Bring your own physics <https://deepinv.org/auto_examples/basics/demo_custom_physics.html>`_ for guidance.

- Registering the physics in the appropriate test suite and verifying that the tests pass -- when inheriting from :class:`deepinv.physics.LinearPhysics`, it involves the following modifications to `deepinv/tests/test_physics.py`:

  1. Adding a new entry corresponding to your physics configuration to the list variable ``OPERATORS``

  2. Defining how to instantiate this configuration in the function ``find_operator``

  3. If applicable, write the tests specific to your physics, e.g., if it has a specific behavior that is not covered by the existing tests, see `test_MRI` in `here <https://github.com/deepinv/deepinv/blob/main/deepinv/tests/test_physics.py>`_ for an example

- Completing the `API reference <https://deepinv.org/api/deepinv.physics.html>`__ and `User Guide <https://deepinv.org/user_guide/physics/physics.html>`__ with the new operator, and checking that the documentation builds correctly.

Refer to these pull requests for examples of contributing new physics:

- :class:`deepinv.physics.Scattering` in `#1020 <https://github.com/deepinv/deepinv/pull/1020>`_

- :class:`deepinv.physics.SpatialUnwrapping` in `#723 <https://github.com/deepinv/deepinv/pull/723>`_

- :class:`deepinv.physics.TomographyWithAstra` in `#474 <https://github.com/deepinv/deepinv/pull/474>`_

Contributing new datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

In order to contribute a new dataset, you must provide tests alongside it to check that it functions as expected. The DeepInverse code base is regularly tested on automatic continuous integration (CI) servers in order to ensure that the code works the way it is supposed to. Unfortunately, the CI servers have limited resources and they can generally not host the datasets.

We get around this by mocking datasets in the tests. First, write the tests and the implementation, and make sure that the tests pass locally, on the real data. Then, write `mocking code <https://en.wikipedia.org/wiki/Mock_object>`_, code that intercepts calls to input/output (IO) related functions, e.g. `os.listdir`, and make them return a hard coded value, thereby making execution go as if the data was there. For more details and examples, see `this pull request <https://github.com/deepinv/deepinv/pull/490>`_.

Once the implementation, the tests and the mocking code are written, that they pass locally and on the CI servers, the maintainers will be able to review the code and merge it into the main branch if everything goes well. You should bear in mind though that the maintainers won't have the time to make sure the tests pass on the real data, so they will have to trust that you did things correctly.


How to review PRs
-----------------

Reviewing PRs is a brilliant way to contribute to the DeepInverse community. Anyone can review PRs, especially if it covers your area of expertise. Here's a checklist for reviewers for all PRs:

- You have written your review `courteously, respectfully and constructively <https://google.github.io/eng-practices/review/reviewer/comments.html>`_.
- Check that mathematical, methodological or algorithmic contributions are technically correct and match their relevant scientific publications;
- Check that the author has followed the steps of the :ref:`contributing guide <step_by_step_contribute>`, including adding tests, appropriate docstrings, API and User Guide documentation, examples, and changelog;
- Check that the contribution is allowed under the :ref:`LLM policy <llm-policy>`. If not, close the PR;
- Check that the new code and documentation meets the :ref:`DeepInverse style guides <style_guides>` to ensure we maintain our high standard of code and docs.
- Check the code satisfies :ref:`backward compatibility <backward_compatibility>`. If compatibility must be broken, suggest ways to deprecate rather than immediately breaking existing user code.

Thank you for reviewing PRs!

Maintainer commands
~~~~~~~~~~~~~~~~~~~

Maintainers can use the following slash commands as comments on a pull request to trigger specific tests (see `workflows summary <https://github.com/deepinv/deepinv/blob/main/.github/workflows.md>`_ for more details):

- `/test-examples`: runs **all** sphinx gallery examples on CPU using CPU-enabled runners.
- `/gpu-tests`: runs tests and generates docs on GPU using GPU-enabled runners.
