# DnCNN

### *class* deepinv.models.DnCNN(in_channels=3, out_channels=3, depth=20, bias=True, nf=64, pretrained='download', pretrained_2d_isotropic=False, device='cpu', dim=2)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

DnCNN convolutional denoiser.

The architecture was introduced by Zhang *et al.*<sup>[1](#footcite-zhang2017beyond)</sup> and is composed of a series of
convolutional layers with ReLU activation functions. The number of layers can be specified by the user. Unlike the
original paper, this implementation does not include batch normalization layers.

The network can be initialized with pretrained weights, which can be downloaded from an online repository. The
pretrained weights are trained with the default parameters of the network, i.e. 20 layers, 64 channels and biases.

* **Parameters:**
  * **in_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – input image channels
  * **out_channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – output image channels
  * **depth** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of convolutional layers
  * **bias** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – use bias in the convolutional layers
  * **nf** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels per convolutional layer
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – use a pretrained network. If `pretrained=None`, the weights will be initialized at random
    using Pytorch’s default initialization. If `pretrained='download'`, the weights will be downloaded from an
    online repository (only available for architecture with depth 20, 64 channels and biases).
    It is possible to download weights trained via the regularization method in Pesquet *et al.*<sup>[2](#footcite-pesquet2021learning)</sup>, using `pretrained='download_lipschitz'`.
    When building a 3D network, it is possible to initialize with 2D pretrained weights by using `pretrained='download_2d'` or `pretrained='download_lipschitz_2d'`, which provides a good starting point for fine-tuning.
    Finally, `pretrained` can also be set as a path to the user’s own pretrained weights.
    See [pretrained-weights](https://deepinv.org/user_guide/reconstruction/pretrained-models.html.md#pretrained-weights) for more details.
  * **pretrained_2d_isotropic** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – when loading 2D pretrained weights into a 3D network, whether to initialize the 3D kernels isotropically. By default the weights are loaded axially, i.e., by initializing the central slice of the 3D kernels with the 2D weights.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to put the model on.
  * **dim** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – Whether to build 2D or 3D network (if str, can be “2”, “2d”, “3D”, etc.)

<hr />

* **References:**

* <a id='footcite-zhang2017beyond'>**[1]**</a> Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: residual learning of deep cnn for image denoising. *IEEE transactions on image processing*, 26(7):3142–3155, 2017.
* <a id='footcite-pesquet2021learning'>**[2]**</a> Jean-Christophe Pesquet, Audrey Repetti, Matthieu Terris, and Yves Wiaux. Learning maximally monotone operators for image recovery. *SIAM Journal on Imaging Sciences*, 14(3):1206–1237, 2021.

#### forward(x, sigma=None)

Run the denoiser on noisy image. The noise level is not used in this denoiser.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level (not used)

#### NOTE
The argument `sigma` is included for compatibility with the base class [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser) but is not used in this model.

<a id="sphx-glr-backref-deepinv-models-dncnn"></a>

## Examples using `DnCNN`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use a mirror descent algorithm for solving an inverse problem with Poisson noise. See :footcitehurault2023convergent for more details.">  <div class="sphx-glr-thumbnail-title">Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard PnP algorithm with DnCNN denoiser for computed tomography.">  <div class="sphx-glr-thumbnail-title">Vanilla PnP for computed tomography (CT).</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This code shows you how to use sampling algorithms to quantify uncertainty of a reconstruction from incomplete and noisy measurements.">  <div class="sphx-glr-thumbnail-title">Uncertainty quantification with PnP-ULA.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Some unfolded architectures rely on a least-squares solver &lt;deepinv.optim.linear.least_squares&gt; to compute the proximal step w.r.t. the data-fidelity term (e.g., deepinv.optim.optim_iterators.ADMMIteration or deepinv.optim.optim_iterators.HQSIteration):  ">  <div class="sphx-glr-thumbnail-title">Reducing the memory and computational complexity of unfolded network training</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
