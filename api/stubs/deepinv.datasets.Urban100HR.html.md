# Urban100HR

### *class* deepinv.datasets.Urban100HR(root=None, download=False, transform=None, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [Urban100](https://paperswithcode.com/dataset/urban100).

The Urban100 dataset <sup>[1](#footcite-huang2015single)</sup> contains 100 images of urban scenes.
It is commonly used as a test set to evaluate the performance of super-resolution models.

**Raw data file structure:**

```default
self.root --- Urban100_HR --- img_001.png
           |               |
           |               -- img_100.png
           |
           -- Urban100_HR.tar.gz
```

This dataset wrapper gives access to the 100 high resolution images in the Urban100_HR folder.
Raw dataset source : [https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz](https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz)

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform:** (*Callable*) – (optional)  A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```pycon
  >>> import shutil
  >>> from deepinv.datasets import Urban100HR
  >>> dataset = Urban100HR(root="./Urban100", download=True)  # download raw data at root and load dataset
  Dataset has been successfully downloaded.
  >>> print(dataset.check_dataset_exists())                      # check that raw data has been downloaded correctly
  True
  >>> print(len(dataset))                                        # check that we have 100 images
  100
  >>> shutil.rmtree("./Urban100")                             # remove raw data from disk
  ```

<hr />

* **References:**

* <a id='footcite-huang2015single'>**[1]**</a> Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 5197–5206. 2015.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- Urban100_HR --- img_001.png
           |               |
           |               -- img_100.png
           |
           -- xxx
```

<a id="sphx-glr-backref-deepinv-datasets-urban100hr"></a>

## Examples using `Urban100HR`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train various networks using adversarial training for deblurring problems. We demonstrate running training and inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and UAIR implemented in the library, and how to simply train your own GAN by using deepinv.training.AdversarialTrainer. These examples can also be easily extended to train more complicated GANs such as CycleGAN.">  <div class="sphx-glr-thumbnail-title">Imaging inverse problems with adversarial networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="In many large-scale imaging problems, the size of the image/volume to reconstruct is very large, making it impossible to train reconstruction networks (in this example, unfolded networks) with a single GPU. The deepinv.distributed framework enables training a model on multiple GPUs, by carefully parallelizing the data fidelity and denoising steps inside the network.">  <div class="sphx-glr-thumbnail-title">Distributed Training of Unfolded Networks</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Single-image super-resolution (SISR) is the inverse problem of recovering a high-resolution (HR) image x from a low-resolution (LR) observation y = \\downarrow_s(x), where \\downarrow_s denotes downsampling by factor s.">  <div class="sphx-glr-thumbnail-title">Super-resolution with SRResNet</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example provides a very simple quick start introduction to training reconstruction networks with DeepInverse for solving imaging inverse problems.">  <div class="sphx-glr-thumbnail-title">Training a reconstruction model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
