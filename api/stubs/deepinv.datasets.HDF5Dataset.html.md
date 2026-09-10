# HDF5Dataset

### *class* deepinv.datasets.HDF5Dataset(path, train=None, split=None, transform=None, load_physics_generator_params=False, dtype=torch.float, complex_dtype=torch.cfloat, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

DeepInverse HDF5 dataset

DeepInverse features its own file format for imaging datasets designed as a
subset of the [HDF5 file format](https://www.hdfgroup.org/solutions/hdf5/).
A dataset in this format is typically obtained from a base dataset of ground-truth
images and measured through a forward operator using the function [`deepinv.datasets.generate_dataset()`](https://deepinv.org/api/stubs/deepinv.datasets.generate_dataset.html.md#deepinv.datasets.generate_dataset).
This class features the code to load them in memory.

<hr />

* **Basics:**

The file containing the dataset is opened in the constructor and remains
opened until the method [`close()`](#deepinv.datasets.HDF5Dataset.close) is called.

* **Parameters:**
  **path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Path to the HDF5 file containing the dataset.

<hr />

* **Splits:**

The dataset is structured in splits that are either freely named or
understood specifically as training and testing splits. By convention,
the training split is named `train` and the testing split `test`.
In both cases, the parameter `split` can be used to select one of
the splits available in the dataset. For the specific case of training
and testing splits, they can be loaded in using the boolean parameter
`train`. If `train=True`, the training split is loaded, otherwise
the testing split is loaded.

By default, if neither `split` nor `train` is provided, it attempts to
load the training split. We don’t recommend relying on this behaviour which
is likely to change in future versions of the library.

#### WARNING
If both `split` and `train` are provided, then `split` takes
precedence and `train` is ignored. We recommend that you only
specify one of the two parameters to avoid errors.

#### NOTE
A single instance of the class holds a single split of the dataset. If
you wish to load multiple splits, you must instantiate the class once
per split. See for instance [Training a reconstruction model](https://deepinv.org/auto_examples/models/demo_training.html.md#sphx-glr-auto-examples-models-demo-training-py).

* **Parameters:**
  * **split** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – The name of the split to load, for instance `"train"`, `"test"` or `"val"``. It can be left unspecified if `train` is used instead.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `split` is left unspecified, uses `"train"` as the split name if set to `True` and `"test"` otherwise. Note that if `split` is specified, this parameter is ignored with a warning.

<hr />

* **Entries:**

HDF5 datasets adhere to our [conventions for datasets](https://deepinv.org/user_guide/training/datasets.html).
In particular, their entries are either pairs of ground truth images and measurements `(x, y)` or triplets with additional
physics parameters `(x, y, params)`. It is possible that the dataset does not contain ground truth data and in this case
the ground truth is replaced by a scalar NaN tensor.

Physics parameters represent additional information about the measurement
process. For instance the mask for inpainting or the blur kernel for
deblurring. HDF5 datasets can contain the physics parameters used to
generate each set of measurements and in that case, they are returned with
each entry as a dictionary as long as the parameter
`load_physics_generator_params` is set to `True`. Note that if the
parameter is set and the dataset does not contain any physics parameter, an
empty dictionary is returned nonetheless.

Measurements intended to be used with [`deepinv.physics.StackedPhysics`](https://deepinv.org/api/stubs/deepinv.physics.StackedPhysics.html.md#deepinv.physics.StackedPhysics) are stored across multiple members of
the HDF5 file, one per operator. In that case, member names follow the format `y{i}_{split_name}` where `i`
denotes the stack index (starting at 0) and they are loaded as a [`deepinv.utils.tensorlist.TensorList`](https://deepinv.org/api/stubs/deepinv.utils.TensorList.html.md#deepinv.utils.TensorList).

#### NOTE
Physics parameters are identified using a fallback logic. Namely, every member with a name of the form `{prefix}_{split_name}`
that is neither interpreted as containing ground truths or measurements (including stacked measurements) defaults
to being interpreted as containing physics parameter, with `prefix` denoting the parameter name.
In particular, the joint presence of physics parameters and stacked measurements is generally supported as long
as custom physics parameter names cannot be misinterpreted as ground truth or measurements names, for instance
`x`, `y`, and `y0` are unsupported parameter names.

#### NOTE
HDF5 datasets always contain measurements even though our conventions permit datasets
with only ground truths (with or without physics parameters).

* **Parameters:**
  **load_physics_generator_params** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Return the physics parameters with each entry. If no physics parameter is featured in the dataset, an empty dictionary is returned nonetheless.

<hr />

* **Pre-processing:**

The data loaded in from the disk is not necessarily returned as is. The
pre-processing pipeline contains two steps. First, the real and complex
numbers are cast to user-provided dtypes using the parameters `dtype`
and `complex_dtype`. Then, an optional transform provided by the user
through the parameter `transform` is applied to the ground truth
image.

#### NOTE
The user-provided transformation is only applied to the ground truth
image. It does not affect the measurements or the physics parameters.

* **Parameters:**
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The dtype for real-valued numbers, by default `torch.float`.
  * **complex_dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The dtype for complex-valued numbers, by default `torch.cfloat`.
  * **transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) *,* *Callable* *,* *None*) – An optional transformation applied to the ground truth.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

#### close()

Closes the HDF5 dataset. Use when you are finished with the dataset.

#### *property* unsupervised *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

Test if the split is unsupervised (i.e. contains no ground truths).

<a id="sphx-glr-backref-deepinv-datasets-hdf5dataset"></a>

## Examples using `HDF5Dataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Follow this example to get started with DeepInverse in under 5 minutes.">  <div class="sphx-glr-thumbnail-title">5 minute quickstart tutorial</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In this example, we show how to solve a deblurring inverse problem using an explicit prior.">  <div class="sphx-glr-thumbnail-title">Image deblurring with custom deep explicit prior.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use the DPIR method to solve a PnP image deblurring problem. The DPIR method is described in :footcitezhang2021plug. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).">  <div class="sphx-glr-thumbnail-title">DPIR method for PnP image deblurring.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Implementation of :footciteromano2017little using as plug-in denoiser the Gradient-Step Denoiser (GSPnP) :footcitehurault2021gradient which provides an explicit prior.">  <div class="sphx-glr-thumbnail-title">Regularization by Denoising (RED) for Super-Resolution.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an inpainting inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning from incomplete measurements of multiple operators.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the Neighbor2Neighbor loss, which exploits the local correlation of natural images.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Neighbor2Neighbor loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, using noisy images only via the Generalized Recorrupted2Recorrupted (GR2R) loss :footcitemonroy2025generalized, which exploits knowledge about the noise distribution. You can change the noise distribution by selecting from predefined noise models such as Gaussian, Poisson, and Gamma noise.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the Generalized R2R loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images only via the SURE loss, which exploits knowledge about the noise distribution.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the SURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a denoiser network in a fully self-supervised way, i.e., using noisy images with unknown noise level only via the UNSURE loss, which is introduced by :footcitetachella2024unsure.">  <div class="sphx-glr-thumbnail-title">Self-supervised denoising with the UNSURE loss.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function. The custom prior in use is The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.">  <div class="sphx-glr-thumbnail-title">Learned iterative custom prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
