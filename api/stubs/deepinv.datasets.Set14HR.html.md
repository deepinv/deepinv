# Set14HR

### *class* deepinv.datasets.Set14HR(root=None, download=False, transform=None, verbose=True, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [Set14](https://paperswithcode.com/dataset/set14).

The Set14 dataset <sup>[1](#footcite-huang2015single)</sup> is a dataset consisting of 14 images commonly used for testing performance of image reconstruction algorithms.
Images have sizes ranging from 276×276 to 512×768 pixels.

**Raw data file structure:**

```default
self.root --- Set14_HR.tar.gz
        |
        --- Set14_HR --- baboon.png
        |             |
        |             --- butterfly.png
        |             --- face.png
        |             --- ...
        |
        --- xxx
```

This dataset wrapper gives access to the 14 high resolution images in the `Set14_HR` folder.
Raw dataset source : [https://huggingface.co/datasets/eugenesiow/Set14](https://huggingface.co/datasets/eugenesiow/Set14)

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform:** (*Callable*) – (optional)  A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Print a message if the dataset has been correctly downloaded. Default `True`.
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```default
  import shutil
  from deepinv.datasets import Set14HR
  dataset = Set14HR(root="Set14", download=True)  # download raw data at root and load dataset
  Dataset has been successfully downloaded.
  print(dataset.check_dataset_exists())                # check that raw data has been downloaded correctly
  True
  print(len(dataset))                                  # check that we have 14 images
  14
  shutil.rmtree("Set14")                          # remove raw data from disk
  ```

<hr />

* **References:**

* <a id='footcite-huang2015single'>**[1]**</a> Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 5197–5206. 2015.

#### check_dataset_exists()

Verify that the image folders exist and contain all the images.

`self.root` should have the following structure:

```default
self.root --- Set14_HR --- baboon.png
        |             |
        |             --- butterfly.png
        |             --- face.png
        |             --- ...
        |
        --- xxx
```
