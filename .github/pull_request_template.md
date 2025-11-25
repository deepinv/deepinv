
Thank you for contributing to DeepInverse!

Please refer to our [contributing guidelines](https://deepinv.github.io/deepinv/contributing.html) for full instructions on how to contribute, including writing tests, documentation and code style.

Once the GitHub tests have been approved by a maintainer (only required for first-time contributors), and the `Build Docs` GitHub action has run successfully, you can download the documentation as a zip file from the [Actions page](https://github.com/deepinv/deepinv/actions/workflows/documentation.yml). Look for the workflow run corresponding to your pull request.


### Checks to be done before submitting your PR

- [ ] `python3 -m pytest deepinv/tests` runs successfully.
- [ ] `black .` runs successfully.
- [ ] `make html` runs successfully (in the `docs/` directory).
- [ ] Updated docstrings related to the changes (as applicable).
- [ ] Added an entry to the [changelog.rst](https://github.com/deepinv/deepinv/blob/main/docs/source/changelog.rst).
