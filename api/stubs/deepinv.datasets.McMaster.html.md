# McMaster

### *class* deepinv.datasets.McMaster(root=None, download=False, transform=None, verbose=True, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [McMaster](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm).

The McMaster dataset <sup>[1](#footcite-zhang2011color)</sup> is a dataset consisting of 18 images commonly used for testing performance of
color demosaicing and image reconstruction algorithms.
Images have a fixed size of 500×500 pixels.

**Raw data file structure:**

```default
self.root --- McM.zip
        |
        --- McM --- 1.tif
        |         |
        |         --- 2.tif
        |         --- 3.tif
        |         --- ...
        |
        --- xxx
```

Raw dataset source : [https://www4.comp.polyu.edu.hk/~cslzhang/DATA/McM.zip](https://www4.comp.polyu.edu.hk/~cslzhang/DATA/McM.zip)

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

* <a id='footcite-zhang2011color'>**[1]**</a> Lei Zhang, Xiaolin Wu, Antoni Buades, and Xin Li. Color demosaicking by local directional interpolation and nonlocal adaptive thresholding. *Journal of Electronic imaging*, 20(2):023016–023016, 2011.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- McM --- 1.tif
        |            |
        |            --- 2.tif
        |            --- 3.tif
        |            --- ...
        |
        --- xxx
```
