# BM4D

### *class* deepinv.models.BM4D(device='cpu')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

BM4D denoiser.

The BM4D denoiser is a generalization of the BM3D denoiser (introduced by Dabov *et al.*<sup>[1](#footcite-dabov2007image)</sup>) to volumetric images.

#### NOTE
Unlike other denoisers from the library, this denoiser is applied sequentially to each noisy image in the batch
(no parallelization).

#### NOTE
This denoiser does not support backpropagation.

#### TIP
This module wraps the BM4D denoiser from the [BM4D python package](https://pypi.org/project/bm4d/).
It can be installed with `pip install bm4d`.

<hr />

* **References:**

* <a id='footcite-dabov2007image'>**[1]**</a> Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transform-domain collaborative filtering. *IEEE Transactions on image processing*, 16(8):2080–2095, 2007.
