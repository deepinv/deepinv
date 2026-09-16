# Set5HR

### *class* deepinv.datasets.Set5HR(root=None, download=False, transform=None, verbose=True, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [Set5](https://paperswithcode.com/dataset/set5).

The Set5 dataset <sup>[1](#footcite-bevilacqua2012low)</sup> is a dataset consisting of 5 images commonly used for testing performance of image reconstruction algorithms.
Images have sizes ranging from 256×256 to 512×512 pixels.

**Raw data file structure:**

```default
self.root --- Set5_HR.tar.gz
        |
        --- Set5_HR --- baby.png
        |             |
        |             --- bird.png
        |             --- butterfly.png
        |             --- ...
        |
        --- xxx
```

Raw dataset source : [https://huggingface.co/datasets/eugenesiow/Set5](https://huggingface.co/datasets/eugenesiow/Set5)

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

* <a id='footcite-bevilacqua2012low'>**[1]**</a> Marco Bevilacqua, Aline Roumy, Christine Guillemot, and Marie-Line Alberi-Morel. Low-complexity single-image super-resolution based on nonnegative neighbor embedding. In *Proceedings of the British Machine Vision Conference (BMVC)*, 135.1–135.10. 2012.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- Set5_HR --- baby.png
        |             |
        |             --- bird.png
        |             --- butterfly.png
        |             --- ...
        |
        --- xxx
```
