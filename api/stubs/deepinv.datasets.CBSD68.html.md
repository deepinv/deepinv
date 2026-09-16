# CBSD68

### *class* deepinv.datasets.CBSD68(root=None, download=False, transform=None, rotate=False, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for [CBSD68](https://paperswithcode.com/dataset/cbsd68).

Color BSD68 dataset for image restoration benchmarks is part of The Berkeley Segmentation Dataset and Benchmark from Martin *et al.*<sup>[1](#footcite-martin2001database)</sup>.
It is used for measuring image restoration algorithms performance. It contains 68 images.

**Raw data file structure:**

```default
self.root --- data-00000-of-00001.arrow
           -- dataset_info.json
           -- state.json
```

This dataset wraps the HuggingFace version of the dataset.
HF source : [https://huggingface.co/datasets/deepinv/CBSD68](https://huggingface.co/datasets/deepinv/CBSD68)

#### NOTE
Using the CBSD68 dataset requires the `datasets` library. It can be installed via `pip install datasets`.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform** (*Callable*) – (optional) A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **rotate** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If set to `True` images are rotated to have all the same orientation. This can be important to use a torch dataloader.
    Default at False.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as a dict with key `"x"` instead of an image (default `False`).

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```pycon
  >>> import shutil
  >>> from deepinv.datasets import CBSD68
  >>> dataset = CBSD68(root="CBSD68", download=True)  # download raw data at root and load dataset
  Dataset has been successfully downloaded.
  >>> print(dataset.check_dataset_exists())                # check that raw data has been downloaded correctly
  True
  >>> print(len(dataset))                                  # check that we have 68 images
  68
  >>> shutil.rmtree("CBSD68")                         # remove raw data from disk
  ```

#### NOTE
This class requires the `datasets` package to be installed. Install with `pip install datasets`.

<hr />

* **References:**

* <a id='footcite-martin2001database'>**[1]**</a> D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In *Proc. 8th Int’l Conf. Computer Vision*, volume 2, 416–423. July 2001.

<hr />

* **Used in benchmarks:**

- [CBSD68 gaussian denoising](https://deepinv.org/auto_benchmarks/cbsd68_gaussian_denoising.html.md#cbsd68-gaussian-denoising)

#### check_dataset_exists()

Verify that the HuggingFace dataset folder exists and contains the raw data file.

`self.root` should have the following structure:

```default
self.root --- data-00000-of-00001.arrow
           -- xxx
           -- xxx
```

This is a soft verification as we don’t check all the files in the folder.
