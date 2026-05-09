
Thank you for contributing to DeepInverse!

Please read our [contributing guidelines](https://deepinv.github.io/deepinv/contributing.html) for instructions on how to contribute, including writing tests, documentation and code style.

Once the GitHub tests have been approved by a maintainer (only required for first-time contributors), and the `Build Docs` GitHub action has run successfully, you can download the documentation as a zip file from the [Actions page](https://github.com/deepinv/deepinv/actions/workflows/docs_cpu.yml). Look for the workflow run corresponding to your pull request.


### Checks to be done before submitting your PR

- [ ] `python3 -m pytest deepinv/tests` runs successfully.
- [ ] `black .` and `ruff check .` run successfully.
- [ ] `make html` runs successfully (in the `docs/` directory).
- [ ] Updated docstrings related to the changes (as applicable).
- [ ] Added an entry to the [changelog.rst](https://github.com/deepinv/deepinv/blob/main/docs/source/changelog.rst).

### LLM policy

LLM usage is ok, but not PRs generated 100% by AI. See our [LLM policy](https://deepinv.github.io/deepinv/contributing.html#llm-policy) Tick below as appropriate:

- [ ] I did not use LLM tools to write the code
- [ ] LLM tools helped me to write part of the code
- [ ] An LLM tool wrote all of the code.
- [ ] An agent submitted the PR and wrote the description.