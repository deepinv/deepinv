# BSDS500

### *class* deepinv.datasets.BSDS500(root=None, download=False, train=True, splits=None, transform=None, rotate=False, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for [BSDS500](https://github.com/BIDS/BSDS500).

BSDS500 dataset for image restoration benchmarks. BSDS stands for The Berkeley Segmentation Dataset and Benchmark from Martin *et al.*<sup>[1](#footcite-martin2001database)</sup>.
Originally, BSDS500 was used for image segmentation. However, this dataset only loads the ground truth images.
The dataset consists of RGB color images of size 481 x 321 or 321 x 481 and is divided into three splits:

- “train”: contains 200 training images
- “val”: contains 100 validation images
- “test”: contains 200 test images

Despite the name, the “val” split is often used for testing (e.g., it is a superset of CBSD68), while the “train” and “test” splits are used for training.

This dataset uses the file structure from the github repository [https://github.com/BIDS/BSDS500](https://github.com/BIDS/BSDS500)
from the institute which published the dataset.

**Raw data file structure:**

```default
self.root --- BSDS500-master --- (all files from the github repo)
```

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*pathlib.Path*](https://docs.python.org/3.9/library/pathlib.html#pathlib.Path)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **train** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, the standard training dataset (containing the splits “train” and “test”) will be loaded. If `False`,
    the standard test set (containing the “val” split) is loaded (which is a superset of CBSD68). Default at True
  * **splits** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *of* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Alternatively to the `train` parameter, the precise splits used can be defined. E.g., pass `["train", "val"]`
    to load the “train” and “val” splits. None for using the splits defined by the `train` parameter. Default None.
  * **transform** (*Callable*) – (optional) A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`.
  * **rotate** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If set to `True` images are rotated to have all the same orientation. This can be important to use a torch dataloader.
    Default at False.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **References:**

* <a id='footcite-martin2001database'>**[1]**</a> D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In *Proc. 8th Int’l Conf. Computer Vision*, volume 2, 416–423. July 2001.

<a id="sphx-glr-backref-deepinv-datasets-bsds500"></a>

## Examples using `BSDS500`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This is a simple example to show how to use vanilla unfolded Plug-and-Play. The DnCNN denoiser and the algorithm parameters (stepsize, regularization parameters) are trained jointly. For simplicity, we show how to train the algorithm on a  small dataset. For optimal results, use a larger dataset.">  <div class="sphx-glr-thumbnail-title">Vanilla Unfolded algorithm for super-resolution</div>
</div>
<!-- thumbnail-parent-div-close --></div>
