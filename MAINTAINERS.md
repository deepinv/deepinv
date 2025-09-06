# Maintainers

**Contents**

- [Current Maintainers](#current-maintainers)
- [Emeritus](#emeritus)
- [Responsibilities](#responsibilities)

This document contains a list of maintainers in this repo. See [below](#responsibilities) for the maintainer responsibilities.

If you're a contributor and are interested in becoming a maintainer, please reach out to one of us below!

## Current Maintainers

| Maintainer | GitHub | Affiliation | Email |
|------|------|-------|-------|
| Julián Tachella | [@tachella](https://github.com/tachella) | CNRS, ENS Lyon | [📧](mailto:julian.tachella@cnrs.fr) |
| Samuel Hurault | [@samuro95](https://github.com/samuro95/) | CNRS, ENS Paris | [📧](mailto:huraultsamuel@gmail.com) |
| Andrew Wang | [@andrewwango](https://github.com/andrewwango) | University of Edinburgh | [📧](mailto:andrew.wang@ed.ac.uk) |
| Minh-Hai Nguyen | [@mh-nguyen712](https://github.com/mh-nguyen712) | IRIT, CBI, CNRS, Université de Toulouse | [📧](mailto:nguyenhai7120qh@gmail.com) |
| Jérémy Scanvic | [@jscanvic](https://github.com/jscanvic) | CNRS, ENS Lyon | [📧](mailto:jeremy.scanvic@ens-lyon.fr) |

## Emeritus

| Maintainer | GitHub | Affiliation | Email |
|------|------|-------|-------|
| Matthieu Terris | [@matthieutrs](https://github.com/matthieutrs) | Université Paris-Saclay, Inria, CEA | [📧](mailto:matthieu.terris@gmail.com) |
| Dongdong Chen | [@edongdongchen](https://github.com/edongdongchen) | Heriot-Watt University | [📧](mailto:d.chen@hw.ac.uk) |

## Responsibilities

Maintainers are active and visible members of the community, and have [maintain-level permissions on a repository](https://docs.github.com/en/organizations/managing-access-to-your-organizations-repositories/repository-permission-levels-for-an-organization). Maintainers serve the community using this privilege, by having the below day-to-day responsibilities. Read more maintainer best practices [here](https://opensource.guide/best-practices/).

- **Security**: security is your number one priority. Maintainer's GitHub keys must be password protected securely and any reported security vulnerabilities are addressed before features or bugs, following the [security guidelines](SECURITY.md).

- **Uphold code of conduct**: model the behavior set forward by the [code of conduct](CODE_OF_CONDUCT.md) and raise any violations to other maintainers.

- **Pull requests**: ensure PRs adhere to our [contributing guidelines](https://deepinv.github.io/deepinv/contributing.html) and have completed the checklist in the [PR template](.github/pull_request_template.md). Triage the PR and assign them to maintainers for review. Provide actionable feedback to guide the PR towards a conclusion. In cases of uncertainty, or where contributed code is out of your domain of expertise, reach out to additional maintainers or contributors.

- **Issues**: manage labels, review issues regularly, and triage by labelling them, e.g. `bug`, `documentation`, `duplicate`, `enhancement`, `good first issue`, `help wanted`, `blocker`, `invalid`, `question`, `wontfix`, and `untriaged`. Request further information from a submitter if an issue is not clear. See the [issue template](.github/ISSUE_TEMPLATE.md) for an example.

- **Philosophy**: maintainers uphold the design philosophy of DeepInverse, as detailed in the paper [DeepInverse: A Python package for solving imaging inverse problems with deep learning](https://arxiv.org/abs/2505.20160). Maintainers make executive decisions to uphold this, e.g. in accepting or rejecting contributions, arranging the code into submodules, or structuring the documentation.

- **Responsiveness**: promptly respond to [issues](https://github.com/deepinv/deepinv/issues) and [discussions](https://github.com/deepinv/deepinv/discussions). Allocate time for reviewing, commenting and continuing conversations.

- **Meetings**: maintainers commit to regularly attending team meetings to review team progress, assign work, and contribute to discussions about blockers.

- **Maintain overall health**: keep the `main` branch at production quality at all times, and backport features as needed. Monitor and improve CI workflows where appropriate, including docs build, automatic tests, code coverage, formatting and linting.

- **Dependencies**: maintain up-to-date dependencies on third party projects to reduce the risk of security vulnerabilities.

- **Releases**: make frequent project releases to the community. Handle issues arising from nightly versions breaking compatibility with stable releases.

- **Adding maintainers**: propose contributors to become maintainers, particularly those with a record of frequent, high quality contributions, understanding of our vision, and thoughtful conversations. Maintainers are added by consensus agreement of all current maintainers.

- **Removing maintainers**: There are many reasons why a maintainer may choose to take a step back or move on from the project. Existing maintainers can choose to leave the project at any time, with or without reason. If a maintainer becomes inactive or fails to keep the day-to-day responsibilities listed above that define maintainer status, they may be encouraged by other maintainers to step down, with no hard feelings. If there is a disagreement, a decision to remove the maintainer will be made on consensus agreement of all other maintainers. All past maintainers will be listed as an [emeritus maintainer](#emeritus), and may rejoin at any time. Of course, a maintainer will be removed using consensus agreement from other maintainers if they violate the [code of conduct](CODE_OF_CONDUCT.md), or take other actions that negatively impact the project.