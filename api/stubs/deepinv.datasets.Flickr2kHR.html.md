# Flickr2kHR

### *class* deepinv.datasets.Flickr2kHR(root=None, download=False, transform=None, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [Flickr2K](https://github.com/limbee/NTIRE2017).

The Flickr2k dataset introduced by Agustsson and Timofte<sup>[1](#footcite-agustsson2017ntire)</sup> contains 2650 2K images.

**Raw data file structure:**

```default
self.root --- Flickr2K --- 000001.png
           |            |
           |            -- 002650.png
           |
           -- Flickr2K.zip
```

Partial raw dataset source (only HR images) : [https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip](https://huggingface.co/datasets/yangtao9009/Flickr2K/resolve/main/Flickr2K.zip)
<br/>
Full raw dataset source (HR and LR images) : [https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)
<br/>
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
  ```default
  from deepinv.datasets import Flickr2kHR
  root = "/path/to/dataset/Flickr2K"
  dataset = Flickr2kHR(root=root, download=True)  # download raw data at root and load dataset
  print(dataset.check_dataset_exists())           # check that raw data has been downloaded correctly
  print(len(dataset))                             # check that we have 100 images
  ```

<hr />

* **References:**

* <a id='footcite-agustsson2017ntire'>**[1]**</a> Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: dataset and study. In *Proceedings of the IEEE conference on computer vision and pattern recognition workshops*, 126–135. 2017.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- Flickr2K --- 000001.png
           |            |
           |            -- 002650.png
           |
           -- xxx
```
