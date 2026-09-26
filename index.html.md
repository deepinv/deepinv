# DeepInverse: a Python library for imaging with deep learning

[![pip install](https://img.shields.io/pypi/dm/deepinv.svg?logo=pypi&label=pip%20install&color=fedcba)](https://pypistats.org/packages/deepinv) [![stars](https://img.shields.io/github/stars/deepinv/deepinv?style=flat&label=%E2%AD%90%20Star%20us%20on%20GitHub)](https://github.com/deepinv/deepinv) [![discord](https://dcbadge.limes.pink/api/server/qBqY5jKw3p?style=flat)](https://discord.gg/qBqY5jKw3p) [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepinv/deepinv/blob/gh-pages/auto_examples/_notebooks/basics/demo_quickstart.ipynb) [![youtube](https://img.shields.io/badge/YouTube-deepinv-red?logo=youtube)](https://www.youtube.com/@deepinv) [![paper](https://joss.theoj.org/papers/10.21105/joss.08923/status.svg)](https://doi.org/10.21105/joss.08923)

[![Test Status](https://github.com/deepinv/deepinv/actions/workflows/test_cpu.yml/badge.svg)](https://github.com/deepinv/deepinv/actions/workflows/test_cpu.yml) [![GPU Test Status](https://github.com/deepinv/deepinv/actions/workflows/test_gpu.yml/badge.svg?branch=main&event=push)](https://github.com/deepinv/deepinv/actions/workflows/test_gpu.yml) [![Docs Status](https://github.com/deepinv/deepinv/actions/workflows/docs_cpu.yml/badge.svg)](https://github.com/deepinv/deepinv/actions/workflows/docs_cpu.yml) [![GPU Docs Status](https://github.com/deepinv/deepinv/actions/workflows/docs_gpu.yml/badge.svg?branch=main&event=push)](https://github.com/deepinv/deepinv/actions/workflows/docs_gpu.yml) [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/release/python-3100/) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![codecov](https://codecov.io/gh/deepinv/deepinv/branch/main/graph/badge.svg?token=77JRvUhQzh)](https://codecov.io/gh/deepinv/deepinv)

DeepInverse is an open-source PyTorch-based library for solving imaging inverse problems with deep learning.
The library is part of the [official PyTorch Ecosystem](https://pytorch.landscape2.io/?item=modeling--computer-vision--deepinverse).
`deepinv` accelerates deep learning research across imaging domains, enhances research reproducibility via a common modular framework of problems and algorithms, and lowers the entrance bar to new practitioners.

GitHub: [https://github.com/deepinv/deepinv](https://github.com/deepinv/deepinv)

![deepinv schematic](figures/deepinv_schematic.png)

## Get started

Check out our [5 minute quickstart tutorial](https://deepinv.org/auto_examples/basics/demo_quickstart.html), our [comprehensive examples](https://deepinv.org/auto_examples/index.html), or our [User Guide](https://deepinv.org/user_guide.html.md#user-guide).

`deepinv` features

* A large framework of [predefined imaging operators](https://deepinv.org/user_guide/physics/intro.html.md#physics-intro)
* Many [state-of-the-art deep neural networks](https://deepinv.org/user_guide/reconstruction/introduction.html.md#reconstructors), including pretrained out-of-the-box [reconstruction models](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-models) and [denoisers](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers)
* Comprehensive frameworks for [plug-and-play restoration](https://deepinv.org/user_guide/reconstruction/iterative.html.md#iterative), [optimization](https://deepinv.org/user_guide/reconstruction/optimization.html.md#optim) and [unfolded architectures](https://deepinv.org/user_guide/reconstruction/unfolded.html.md#unfolded)
* [Training losses](https://deepinv.org/user_guide/training/loss.html.md#loss) for inverse problems
* [Sampling algorithms and diffusion models](https://deepinv.org/user_guide/reconstruction/sampling.html.md#sampling) for uncertainty quantification
* A framework for [building datasets](https://deepinv.org/user_guide/training/datasets.html.md#datasets) for inverse problems

### Mailing list

Join our **mailing list** for occasional updates on releases and new features:

<link rel="stylesheet" href="_static/subscribe/subscribe.css">
<div id="subscribe-container"><div class="substack-clone-box"><div class="substack-clone-row">
<input id="emailInput" type="email" placeholder="Type your email…" class="substack-clone-input" oninput="validateEmail()"/>
<button id="subscribeBtn" class="substack-clone-button" disabled onclick="submitAndRedirect()">Subscribe</button>
</div></div></div>
<script src="_static/subscribe/subscribe.js"></script>

<a id="install"></a>

## Install

Install the latest **stable release** of `deepinv` with python 3.10 or higher:

### pip

```bash
pip install deepinv
```

### conda

```bash
conda install -c conda-forge deepinv
```

### uv

```bash
uv pip install deepinv
```

### pixi

```bash
pixi init && pixi add python
pixi add --pypi deepinv
```

Or, to also install **all optional dependencies**:

### pip

```bash
pip install deepinv[dataset,denoisers,physics]
```

### conda

```bash
conda install -c conda-forge deepinv
# fallback to pip for optional dependencies
pip install deepinv[dataset,denoisers,physics]
```

### uv

```bash
uv pip install deepinv[dataset,denoisers,physics]
```

### pixi

```bash
pixi add --pypi "deepinv[dataset,denoisers,physics]"
```

Since `deepinv` is under active development, you can install the **latest nightly version** using:

### pip

```bash
pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
```

### conda

```bash
# requires pre-installing torch and torchvision with conda
pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
```

### uv

```bash
uv pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
```

### pixi

```bash
pixi add --pypi "deepinv @ git+https://github.com/deepinv/deepinv.git"
```

Or, for **updating** an existing installation:

### pip

```bash
pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv
```

### conda

```bash
pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv
```

### uv

```bash
uv pip install --upgrade --force-reinstall --no-deps git+https://github.com/deepinv/deepinv.git#egg=deepinv
```

### pixi

```bash
pixi add --pypi "deepinv @ git+https://github.com/deepinv/deepinv.git"
```

## Finding help

If you have any questions or suggestions, please join the conversation in our
[Discord server](https://discord.gg/qBqY5jKw3p). The recommended way to get in touch with the developers about any bugs or issues is to
[open an issue](https://github.com/deepinv/deepinv/issues).

### Maintainers

Get in touch with our [MAINTAINERS](https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md).

## Contributing

DeepInverse is a [community-driven project](https://deepinv.org/community.html.md#community) and we encourage contributions of all forms.
We are building a comprehensive library of inverse problems and deep learning,
and we need your help to get there!

Interested? [Check out how you can contribute](https://deepinv.org/contributing.html.md#contributing)!

## Citation

If you use DeepInverse in your research, please cite [our paper on JOSS](https://joss.theoj.org/papers/10.21105/joss.08923):

```bash
@article{tachella2025deepinverse,
    title = {DeepInverse: A Python package for solving imaging inverse problems with deep learning},
    journal = {Journal of Open Source Software},
    doi = {10.21105/joss.08923},
    url = {https://doi.org/10.21105/joss.08923},
    year = {2025},
    publisher = {The Open Journal},
    volume = {10},
    number = {115},
    pages = {8923},
    author = {Tachella, Julián and Terris, Matthieu and Hurault, Samuel and Wang, Andrew and Davy, Leo and Scanvic, Jérémy and Sechaud, Victor and Vo, Romain and Moreau, Thomas and Davies, Thomas and Chen, Dongdong and Laurent, Nils and Monroy, Brayan and Dong, Jonathan and Hu, Zhiyuan and Nguyen, Minh-Hai and Sarron, Florian and Weiss, Pierre and Escande, Paul and Massias, Mathurin and Modrzyk, Thibaut and Levac, Brett and Liaudat, Tobías I. and Song, Maxime and Hertrich, Johannes and Neumayer, Sebastian and Schramm, Georg},
}
```

## Star history

[![Star History Chart](https://api.star-history.com/chart?repos=deepinv/deepinv&type=date&legend=top-left&sealed_token=_m7-ngEzgaicNO-u585LK2zkRyHzwKnkM4SNVz6AhngSG7DpKD9wHcVOSqlwsi2X-cTgbZgVQ1FvK-bznTJ7pyOIY4L0-c83JnpoDxMBCkI27h-UOkx2B1d_j1sPoRQcT8q31PZSR7RTOCs34Bfm3fb0PiUJyNtv5syxkOIJb75nuwzomOtNwVCZwQtG)](https://api.star-history.com/chart?repos=deepinv/deepinv&type=date&legend=top-left&sealed_token=_m7-ngEzgaicNO-u585LK2zkRyHzwKnkM4SNVz6AhngSG7DpKD9wHcVOSqlwsi2X-cTgbZgVQ1FvK-bznTJ7pyOIY4L0-c83JnpoDxMBCkI27h-UOkx2B1d_j1sPoRQcT8q31PZSR7RTOCs34Bfm3fb0PiUJyNtv5syxkOIJb75nuwzomOtNwVCZwQtG)

Keywords: image processing, image reconstruction, imaging, computational imaging, inverse problems, deep learning,
mri, superresolution, computed tomography, plug-and-play, deblurring, diffusion models,
unfolded, deep equilibrium models
