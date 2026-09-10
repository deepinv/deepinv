# generate_dataset

### deepinv.datasets.generate_dataset(train_dataset, physics, save_dir, test_dataset=None, val_dataset=None, dataset_filename='dinv_dataset', overwrite_existing=True, train_datapoints=None, test_datapoints=None, val_datapoints=None, physics_generator=None, save_physics_generator_params=True, batch_size=4, num_workers=0, supervised=True, verbose=True, show_progress_bar=False, device='cpu')

Generates dataset of signal/measurement pairs from base dataset.

It generates the measurement data using the forward operator provided by the user.
The dataset is saved in HDF5 format and can be easily loaded using the [`deepinv.datasets.HDF5Dataset`](https://deepinv.org/api/stubs/deepinv.datasets.HDF5Dataset.html.md#deepinv.datasets.HDF5Dataset) class.
The generated dataset contains `train` and `test` splits.

The base dataset of ground-truth images must return tensors `x` or tuples `(x, ...)`. We provide a large library of predefined
popular imaging datasets. See [datasets user guide](https://deepinv.org/user_guide/training/datasets.html.md#datasets) for more information.

Optionally, if random physics generator is used to generate data, also save physics generator params.
This is useful e.g. if you are performing a parameter estimation task and want to evaluate the learnt parameters,
or for measurement consistency/data fidelity, and require knowledge of the params when constructing the loss.

#### NOTE
We support all dtypes supported by `h5py` including complex numbers, which will be stored as complex dtype.

#### NOTE
By default, we overwrite existing datasets if they have been previously created. To avoid this, set `overwrite_existing=False`.

* **Parameters:**
  * **train_dataset** ([*torch.utils.data.Dataset*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) – base dataset of ground-truth images. Must return tensors `x` or tuples `(x, ...)`.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator used to generate the measurement data.
    It can be either a single operator or a list of forward operators. In the latter case, the dataset will be
    assigned evenly across operators.
  * **save_dir** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – folder where the dataset and forward operator will be saved.
  * **test_dataset** ([*torch.utils.data.Dataset*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) – if included, the function will also generate measurements associated to the test dataset.
  * **val_dataset** ([*torch.utils.data.Dataset*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)) – if included, the function will also generate measurements associated to the validation dataset.
  * **dataset_filename** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – desired filename of the dataset (without extension).
  * **overwrite_existing** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, create new dataset file, overwriting any existing dataset with the same `dataset_filename`.
    If `False` and dataset file already exists, does not create new dataset.
  * **train_datapoints** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – Desired number of datapoints in the training dataset. If set to `None`, it will use the
    number of datapoints in the base dataset. This is useful for generating a larger train dataset via data
    augmentation (which should be chosen in the train_dataset).
  * **test_datapoints** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – Desired number of datapoints in the test dataset. If set to `None`, it will use the
    number of datapoints in the base test dataset.
  * **val_datapoints** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* *None*) – Desired number of datapoints in the val dataset.
  * **physics_generator** (*None* *,* [*deepinv.physics.generator.PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator)) – Optional physics generator for generating
    the physics operators. If not None, the physics operators are randomly sampled at each iteration using the generator.
  * **save_physics_generator_params** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – save physics generator params too, ignored if `physics_generator` not used.
  * **batch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – batch size for generating the measurement data
    (it affects the speed of the generating process, and the physics generator batch size)
  * **num_workers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of workers for generating the measurement data
    (it only affects the speed of the generating process)
  * **supervised** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Generates supervised pairs `(x,y)` of measurements and signals.
    If set to `False`, it will generate a training dataset with measurements only `(y)`
    and a test dataset with pairs `(x,y)`
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Output progress information in the console.
  * **show_progress_bar** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Show progress bar during the generation
    of the dataset (if verbose is set to `True`).
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device, e.g. cpu or gpu, on which to generate measurements. All data is moved back to cpu before saving.

## Examples using `generate_dataset`:

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
