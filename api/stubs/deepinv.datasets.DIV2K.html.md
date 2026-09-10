# DIV2K

### *class* deepinv.datasets.DIV2K(root=None, mode='train', download=False, transform=None, use_dict_output=False)

Bases: [`ImageFolder`](https://deepinv.org/api/stubs/deepinv.datasets.ImageFolder.html.md#deepinv.datasets.ImageFolder)

Dataset for [DIV2K Image Super-Resolution Challenge](https://data.vision.ee.ethz.ch/cvl/DIV2K).

The DIV2K dataset from Agustsson and Timofte<sup>[1](#footcite-agustsson2017ntire)</sup> is a high-quality image dataset originally built for image super-resolution tasks.

Images have varying sizes with up to 2040 vertical pixels, and 2040 horizontal pixels.

**Raw data file structure:**

```default
self.root --- DIV2K_train_HR --- 0001.png
           |                  |
           |                  -- 0800.png
           |
           -- DIV2K_valid_HR --- 0801.png
           |                  |
           |                  -- 0900.png
           -- DIV2K_train_HR.zip
           -- DIV2K_valid_HR.zip
```

* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Select a split of the dataset between ‘train’ or ‘val’. Default at ‘train’.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform:** (*Callable*) – (optional)  A function/transform that takes in a PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```pycon
  >>> import shutil
  >>> from deepinv.datasets import DIV2K
  >>> dataset = DIV2K(root="DIV2K", mode="val", download=True)  # download raw data at root and load dataset
  Dataset has been successfully downloaded.
  >>> print(dataset.verify_split_dataset_integrity())                # check that raw data has been downloaded correctly
  True
  >>> print(len(dataset))                                            # check that we have 100 images
  100
  >>> shutil.rmtree("DIV2K")                                    # remove raw data from disk
  ```

<hr />

* **References:**

* <a id='footcite-agustsson2017ntire'>**[1]**</a> Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: dataset and study. In *Proceedings of the IEEE conference on computer vision and pattern recognition workshops*, 126–135. 2017.

<hr />

* **Used in benchmarks:**

- [DIV2K Super Resolution 2x](https://deepinv.org/auto_benchmarks/div2k_super_resolution_2x.html.md#div2k-super-resolution-2x)
- [DIV2K Inpainting easy](https://deepinv.org/auto_benchmarks/div2k_inpainting_easy.html.md#div2k-inpainting-easy)
- [DIV2K Gaussian Deblurring](https://deepinv.org/auto_benchmarks/div2k_gaussian_deblurring.html.md#div2k-gaussian-deblurring)

#### verify_split_dataset_integrity()

Verify the integrity and existence of the specified dataset split.

This method checks if `DIV2K_train_HR` or `DIV2K_valid_HR` folder within
`self.root` exists and validates the integrity of its contents by comparing
the MD5 checksum of the folder with the expected checksum.

The expected structure of the dataset directory is as follows:

```default
self.root --- DIV2K_train_HR --- 0001.png
           |                  |
           |                  -- 0800.png
           |
           -- DIV2K_valid_HR --- 0801.png
           |                  |
           |                  -- 0900.png
           -- xxx
```

<a id="sphx-glr-backref-deepinv-datasets-div2k"></a>

## Examples using `DIV2K`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to fit deepinv.loss.metric.NIQE on a new dataset, and use it to evaluate denoiser performance.">  <div class="sphx-glr-thumbnail-title">Fitting NIQE on a custom dataset</div>
</div>
<!-- thumbnail-parent-div-close --></div>
