# FMD

### *class* deepinv.datasets.FMD(root=None, img_types=None, noise_levels=(1, 2, 4, 8, 16), fovs=tuple(range(1, 20 + 1)), download=False, transform=None, target_transform=None, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset for [Fluorescence Microscopy Denoising](https://github.com/yinhaoz/denoising-fluorescence).

Introduced by Zhang *et al.*<sup>[1](#footcite-zhang2018poisson)</sup>.

1) The Fluorescence Microscopy Denoising (FMD) dataset is dedicated to
<br/>
Poisson-Gaussian denoising.
<br/>
2) The dataset consists of 12,000 real fluorescence microscopy images
<br/>
obtained with commercial confocal, two-photon, and wide-field microscopes
<br/>
and representative biological samples such as cells, zebrafish,
<br/>
and mouse brain tissues.
<br/>
3) Image averaging is used to effectively obtain ground truth images
<br/>
and 60,000 noisy images with different noise levels.
<br/>

**Raw data file structure:**

```default
self.root --- Confocal_BPAE_B  --- avg16 --- 1  --- HV110_P0500510000.png
           |                    |         |      |
           |                    |         |      -- HV110_P0500510049.png
           |                    |         -- 20
           |                    -- avg2
           |                    -- avg4
           |                    -- avg8
           |                    -- gt
           |                    -- raw
           -- ...
           -- WideField_BPAE_R --- ...
           -- Confocal_BPAE_G.tar
           |
           -- WideField_BPAE_R.tar
```

1) There are 12 image types :
<br/>
Confocal_BPAE_B, Confocal_BPAE_G, Confocal_BPAE_R, Confocal_FISH, Confocal_MICE
<br/>
TwoPhoton_BPAE_B, TwoPhoton_BPAE_G, TwoPhoton_BPAE_R, TwoPhoton_MICE
<br/>
WideField_BPAE_B, WideField_BPAE_G, WideField_BPAE_R
<br/>
2) Each image type has its own folder.
<br/>
3) Each folder contains 6 subfolders- : gt, raw, avg2, avg4, avg8 and avg16.
<br/>
4) gt contains clean images, the others have different noise levels applied to images.
<br/>
5) Each subfolder has 20 subsubfolders, corresponding to the “field of view”.
<br/>
6) Each subsubfolder has the same 50 png file names of size (512, 512).
<br/>
7) 12 type of img x 5 levels of noise x 20 “fov” x 50 img = 60 000 noisy img
<br/>
* **Parameters:**
  * **root** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Root directory of dataset. Directory path from where we load and save the dataset.
  * **img_types** ([*list*](https://docs.python.org/3.9/library/stdtypes.html#list) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – Types of microscopy image among 12.
  * **noise_levels** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – Level of noises applied to the image among [1, 2, 4, 8, 16].
  * **fovs** (*Sequence* *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]*) – “Field of view”, value between 1 and 20.
  * **download** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, downloads the dataset from the internet and puts it in root directory.
    If dataset is already downloaded, it is not downloaded again. Default at False.
  * **transform:** (*Callable*) – (optional) A function/transform that takes in a noisy PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **target_transform** (*Callable*) – (optional) A function/transform that takes in a clean PIL image
    and returns a transformed version. E.g, `torchvision.transforms.RandomCrop`
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys “x”, “y”, “params” instead of tuple (default `False`).

<hr />

* **Examples:**
  Instantiate dataset and download raw data from the Internet
  ```default
  import shutil
  from deepinv.datasets import FMD
  img_types = ["TwoPhoton_BPAE_R"]
  dataset = FMD(root="fmd", img_types=img_types, download=True)  # download raw data at root and load dataset
  print(len(dataset))                                            # check that we have 5000 images
  shutil.rmtree("fmd")                                           # remove raw data from disk
  ```

<hr />

* **References:**

* <a id='footcite-zhang2018poisson'>**[1]**</a> Yide Zhang, Yinhao Zhu, Evan Nichols, Qingfei Wang, Siyuan Zhang, Cody Smith, and Scott Howard. A poisson-gaussian denoising dataset with real fluorescence microscopy images. In *CVPR*. 2019.

#### *class* NoisySampleIdentifier(img_type, noise_dirname, fov, fname)

Bases: [`NamedTuple`](https://docs.python.org/3.9/library/typing.html#typing.NamedTuple)

Data structure for identifying noisy data sample files.

* **Parameters:**
  * **img_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Foldername corresponding to one type of image among 12.
  * **noise_dirname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Foldername corresponding to one level of noise,
    ‘raw’ - level 1, ‘avg2’ - 2, ‘avg4’ - 4, ‘avg8’ - 8, ‘avg16’ - 16
  * **fov** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Field of view, value between 1 and 20.
  * **fname** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Filename of a png file containing 1 noisy image.

#### fname *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Alias for field number 3

#### fov *: [int](https://docs.python.org/3.9/library/functions.html#int)*

Alias for field number 2

#### img_type *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Alias for field number 0

#### noise_dirname *: [str](https://docs.python.org/3.9/library/stdtypes.html#str)*

Alias for field number 1
