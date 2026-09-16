# ImageFolder

### *class* deepinv.datasets.ImageFolder(root, x_path=None, y_path=None, loader=None, estimate_params=None, transform=None, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset loading images from files.

By default, the images are loaded from image files (png, jpg etc.) located in `root`.

For more flexibility, set `x_path` or `y_path` to load ground truth `x` and/or measurements `y` from specific file patterns.

#### TIP
To load data from subfolders, use globs such as `x_path = "GT/**/*.png", y_path = "meas/**/*.png"`.

#### TIP
Set `y_path` only to load measurements following the file pattern. The measurement-only data will be returned as a tuple `(torch.nan, y)`.

#### TIP
Use `use_dict_output=True` to return a dict with keys `"x"`, `"y"`, and `"params"` instead of a tuple. This is recommended for better readability and flexibility in returned outputs.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – dataset root directory.
  * **x_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – file glob pattern for ground truth data, defaults to None.
  * **y_path** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* *None*) – file glob pattern for measurement data, defaults to None.
  * **loader** (*Callable*) – optional function that takes filename string and loads file. If `None`, defaults to `PIL.Image.open`.
  * **estimate_params** (*Callable*) – optional function that takes tensors `x,y` and returns dict of `params`. Advanced usage only.
  * **transform** (*Callable* *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – optional callable transform. If `tuple` or `list` of length 2, `x` is transformed with first transform and `y` with second.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple. Default `False` for backward compatibility.

<hr />

Examples:

Using default loading from root folder with image files. Folder structure:

```default
root
├── img1.png
└── img2.png

dataset = ImageFolder(root)
dataset[0]
tensor(...)  # Returns x only
```

Loading paired tensors from nested folders using custom glob and loader. Folder structure:

```default
data/
├── GT/
│   ├── scene1/
│   │   └── x0.pt
│   └── scene2/
│       └── x1.pt
└── meas/
    ├── scene1/
    │   └── y0.pt
    └── scene2/
        └── y1.pt

dataset = ImageFolder(
    root,
    x_path="GT/**/*.pt",
    y_path="meas/**/*.pt",
    loader=torch.load
)
dataset[0]
(tensor(...), tensor(...))  # Returns (x, y) pair
```

Loading unpaired measurements only. Folder structure:

```default
data/
└── meas/
    ├── meas0.png
    └── meas1.png

dataset = ImageFolder(
    "data/",
    y_path="meas/*.png"
)
dataset[0]
(torch.nan, tensor(...))  # Returns unpaired y
```

<a id="sphx-glr-backref-deepinv-datasets-imagefolder"></a>

## Examples using `ImageFolder`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to denoise low-intensity STED fluorescence microscopy images of live-cell mitochondria using the pretrained foundation model deepinv.models.RAM. We load real Abberior STED microscopy data from :footciteosunavargas2025denoising, process it in batches, and visualize the results both with deepinv.utils.plot and with the interactive 3D viewer deepinv.utils.plot_napari.">  <div class="sphx-glr-thumbnail-title">Low-intensity STED fluorescence microscopy denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to fit deepinv.loss.metric.NIQE on a new dataset, and use it to evaluate denoiser performance.">  <div class="sphx-glr-thumbnail-title">Fitting NIQE on a custom dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard TV prior for image deblurring. The problem writes as y = Ax + \\epsilon where A is a convolutional operator and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The TV prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image deblurring with Total-Variation (TV) prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to solve Poisson inverse problems using the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm :footcitesheppMaximumLikelihoodReconstruction1982, also known as the Richardson-Lucy algorithm in the deconvolution setting :footciterichardsonBayesianBasedIterativeMethod1972,lucyIterativeTechniqueRectification1974.">  <div class="sphx-glr-thumbnail-title">Poisson Inverse Problems with Maximum-Likelihood Expectation-Maximization (MLEM)</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Plug-and-Play (PnP) is known to be challenging to apply to certain inverse problems like inpainting. One way to overcome this is to use a multi-scale approach instead of the standard single-scale approach.">  <div class="sphx-glr-thumbnail-title">Multi-scale Plug-and-Play for Inpainting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This a toy example to show you how to use DEQ to solve a deblurring problem. Note that this is a small dataset for training. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Deep Equilibrium (DEQ) algorithms for image deblurring</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
