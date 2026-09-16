# BSD100HR

### *class* deepinv.datasets.BSD100HR(root=None, download=False, transform=None, verbose=True, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [BSD100](https://paperswithcode.com/dataset/bsd100).

The BSD100 dataset <sup>[1](#footcite-martin2001database)</sup> is a dataset consisting of 100 images commonly used for testing performance of image reconstruction algorithms.
Images have sizes ranging from 240×160 to 480×320 pixels.

**Raw data file structure:**

```default
self.root --- BSD100_HR.tar.gz
        |
        --- BSD100_HR --- 3096.png
        |               |
        |               --- 8023.png
        |               --- 12084.png
        |               --- ...
        |
        --- xxx
```

Raw dataset source : [https://huggingface.co/datasets/eugenesiow/BSD100](https://huggingface.co/datasets/eugenesiow/BSD100)

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform:** (*Callable*) – (optional)  A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Print a message if the dataset has been correctly downloaded. Default `True`.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **References:**

* <a id='footcite-martin2001database'>**[1]**</a> D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In *Proc. 8th Int’l Conf. Computer Vision*, volume 2, 416–423. July 2001.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- BSD100_HR --- 3096.png
        |               |
        |               --- 8023.png
        |               --- 12084.png
        |               --- ...
        |
        --- xxx
```
