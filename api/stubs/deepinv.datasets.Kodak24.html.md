# Kodak24

### *class* deepinv.datasets.Kodak24(root=None, download=False, transform=None, verbose=True, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [Kodak24](http://r0k.us/graphics/kodak/).

The Kodak24 dataset <sup>[1](#footcite-kodak1993)</sup> is a dataset consisting of 24 images commonly used for testing performance of
image reconstruction algorithms. Images have a fixed size of 768×512 (or 512×768) pixels.

**Raw data file structure:**

```default
self.root --- Kodak-Lossless-True-Color-Image-Suite-master.zip
        |
        --- Kodak-Lossless-True-Color-Image-Suite-master --- PhotoCD_PCD0992 --- 01.png
        |                                                                     |
        |                                                                     --- 02.png
        |                                                                     --- ...
        |
        --- xxx
```

Raw dataset source : [https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite](https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite)

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

* <a id='footcite-kodak1993'>**[1]**</a> Eastman Kodak Company. Kodak lossless true color image suite. [http://r0k.us/graphics/kodak/](http://r0k.us/graphics/kodak/), 1993.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- Kodak-Lossless-True-Color-Image-Suite-master --- PhotoCD_PCD0992 --- 01.png
        |                                                                        |
        |                                                                        --- 02.png
        |                                                                        --- ...
        |
        --- xxx
```
