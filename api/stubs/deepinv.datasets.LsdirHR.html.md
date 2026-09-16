# LsdirHR

### *class* deepinv.datasets.LsdirHR(root=None, mode='train', download=False, transform=None, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [LSDIR](https://ofsoundof.github.io/lsdir-data/).

Published in Li *et al.*<sup>[1](#footcite-li2023lsdir)</sup>.

A large-scale dataset for image restoration tasks such as image super-resolution (SR),
image denoising, JPEG deblocking, deblurring, and demosaicing, and real-world SR.

**Raw data file structure:**

```default
self.root --- 0001000 --- 0000001.png
           |           |
           |           -- 0001000.png
           |  ...
           |
           -- 0085000 --- 0084001.png
           |           |
           |           -- 0084991.png
           |
           |
           -- val1 --- HR --- val --- 0000001.png
           |        -- X2          |
           |        -- X3          -- 0000250.png
           |        -- X4
```

#### WARNING
Downloading this dataset requires `huggingface-hub`. It is gated, please request access ([https://huggingface.co/ofsoundof/LSDIR](https://huggingface.co/ofsoundof/LSDIR)) and make sure you are logged in using `hf auth login` (CLI) or `from huggingface_hub import login, login()`.

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Select a split of the dataset between ‘train’ or ‘val’. Default at ‘train’.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform:** (*Callable*) – (optional)  A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```default
  from deepinv.datasets import LsdirHR
  val_dataset = LsdirHR(root="Lsdir", mode="val", download=True)  # download raw data at root and load dataset
  print(val_dataset.verify_split_dataset_integrity())             # check that raw data has been downloaded correctly
  print(len(val_dataset))                                         # check that we have 250 images
  ```

<hr />

* **References:**

* <a id='footcite-li2023lsdir'>**[1]**</a> Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, and others. Lsdir: a large scale dataset for image restoration. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1775–1787. 2023.

#### verify_split_dataset_integrity()

Verify the integrity and existence of the specified dataset split.

The expected structure of the dataset directory is as follows:

```default
self.root --- 0001000 --- 0000001.png
        |           |
        |           -- 0001000.png
        |  ...
        |
        -- 0085000 --- 0084001.png
        |           |
        |           -- 0084991.png
        |
        |
        -- val1 --- HR --- val --- 0000001.png
        |        -- X2          |
        |        -- X3          -- 0000250.png
        |
```
