.. _datasets:

Datasets
========

The datasets module lets you use datasets with DeepInverse, for testing and training.

Datasets can be either:

* Loaded in from your own data, see :ref:`base datasets <base-datasets>` for desired format;
* Paired dataset generated synthetically from ground-truth, see :ref:`generating datasets <generating-datasets>`;
* A public dataset, such as DIV2K or FastMRI, see :ref:`predefined datasets <predefined-datasets>`.

.. _base-datasets:

Base Datasets
-------------

Datasets can return optionally ground-truth images `x`, measurements `y`, or :ref:`physics parameters <parameter-dependent-operators>` `params`,
or any combination of these, in one of the following ways:

* `x` i.e a dataset that returns only ground truth;
* `(x, y)` i.e. a dataset that returns pairs of ground truth and measurement. `x` can be equal to `torch.nan` if your dataset is ground-truth-free.
* `(x, params)` i.e. a dataset of ground truth and dict of :ref:`physics parameters <physics_generators>`. Useful for training with online measurements.
* `(x, y, params)` i.e. a dataset that returns ground truth, measurements and dict of params.

.. tip::

  If you have a dataset of measurements only `(y)` or `(y, params)` you should modify it such that it returns `(torch.nan, y)` or `(torch.nan, y, params)`

If you have your own dataset (e.g. a PyTorch `Dataset`), check that it is compatible using the function :func:`deepinv.datasets.check_dataset` 
(e.g. to be used with :class:`deepinv.Trainer` or :class:`deepinv.test`).

.. seealso::

  See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py` for a simple example of how to use DeepInverse with your own dataset.

We provide dataset classes for you to easily load in your own data:

.. list-table:: Base Datasets Overview
   :header-rows: 1

   * - **Dataset**
     - **Description**
   * - :class:`deepinv.datasets.ImageDataset`
     - Base abstract dataset class
   * - :class:`deepinv.datasets.ImageFolder`
     - Dataset that loads images (ground-truth, measurements or both) from a folder
   * - :class:`deepinv.datasets.TensorDataset`
     - Dataset that returns tensor(s) passed in at input: either tensor(s) for a single observation or a whole dataset of them
   * - :class:`deepinv.datasets.HDF5Dataset`
     - Dataset of measurements generated using :func:`deepinv.datasets.generate_dataset`, see :ref:`below <generating-datasets>` for how to use.

.. _generating-datasets:

Generating Datasets
-------------------
You can generate a dataset associated with a certain forward operator using :func:`deepinv.datasets.generate_dataset`
using a base dataset.
Your base dataset can be any dataset that returns ground truth, i.e. either one of our :ref:`predefined datasets <predefined-datasets>`, 
your own data in the format provided by one of our :ref:`base datasets <base-datasets>`,
or other external datasets.

For example, here we generate a dataset of inpainting measurements from the :class:`deepinv.datasets.Set14HR` dataset:

.. note::

    We support all data types supported by ``h5py``, including complex numbers.

.. doctest::

    >>> import deepinv as dinv
    >>> from torchvision.transforms import ToTensor, Compose, CenterCrop
    >>> save_dir = dinv.utils.demo.get_data_home() / 'set14'
    >>> 
    >>> # Define base train dataset
    >>> dataset = dinv.datasets.Set14HR(save_dir, download=True, transform=Compose([CenterCrop(128), ToTensor()])) # doctest: +ELLIPSIS
    ...
    >>> 
    >>> # Define forward operator
    >>> physics = dinv.physics.Inpainting(img_size=(3, 128, 128), mask=0.8, noise_model=dinv.physics.GaussianNoise(sigma=.05))
    >>> 
    >>> # Generate paired dataset
    >>> path = dinv.datasets.generate_dataset(dataset, physics, save_dir=save_dir, verbose=False)

The datasets are saved in ``.h5`` (HDF5) format, and can be easily loaded to PyTorch's standard
:class:`torch.utils.data.DataLoader`:

.. doctest::

    >>> from torch.utils.data import DataLoader
    >>> 
    >>> train_dataset = dinv.datasets.HDF5Dataset(path)
    >>> dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    >>> x, y = next(iter(dataloader))
    >>> x.shape, y.shape
    (torch.Size([4, 3, 128, 128]), torch.Size([4, 3, 128, 128]))
    >>> train_dataset.close()

We can also use physics generators to randomly generate physics `params` for data,
and save and load the physics `params` into the dataset:

.. doctest::

    >>> physics_generator = dinv.physics.generator.SigmaGenerator()
    >>> path = dinv.datasets.generate_dataset(dataset, physics, physics_generator=physics_generator, save_dir=save_dir, verbose=False)
    >>> train_dataset = dinv.datasets.HDF5Dataset(path, load_physics_generator_params=True)
    >>> dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    >>> x, y, params = next(iter(dataloader))
    >>> print(params['sigma'].shape)
    torch.Size([4])


.. _predefined-datasets:

Predefined Datasets
-------------------
Multiple popular easy-to-download datasets are available, which all either return
ground-truth only `x`, paired data `(x, y)` or with params (e.g. mask) `(x, y, params)`.

All these datasets inherit from :class:`deepinv.datasets.ImageDataset`. 

.. list-table:: Datasets Overview
   :header-rows: 1

   * - **Dataset**
     - **Returns**
     - **Dataset Size**
     - **Tensor Sizes**
     - **Description**

   * - :class:`DIV2K <deepinv.datasets.DIV2K>`
     - `x`
     - 800 (train) + 100 (val) images
     - RGB, up to 2040x2040 pixels (variable)
     - A widely-used dataset for natural image restoration.

   * - :class:`Urban100HR <deepinv.datasets.Urban100HR>`
     - `x`
     - 100 images
     - up to 1200x1280 pixels (variable)
     - Contains diverse high-resolution urban scenes, typically used for testing super-resolution algorithms.

   * - :class:`Set14HR <deepinv.datasets.Set14HR>`
     - `x`
     - 14 high-resolution images
     - RGB, 248×248 to 512×768 pixels.
     - A small benchmark dataset for super-resolution tasks, containing a variety of natural images.

   * - :class:`CBSD68 <deepinv.datasets.CBSD68>`
     - `x`
     - 68 images
     - RGB, 481x321 pixels
     - A subset of the color Berkeley Segmentation Dataset.

   * - :class:`FastMRISliceDataset <deepinv.datasets.FastMRISliceDataset>`
     - `(x, y)` or `(x, y, {'mask': mask, 'coil_maps': coil_maps})`
     - Over 100,000 MRI slices
     - Complex, varying shape approx. 640x320
     - Raw MRI knee and brain fully-sampled or undersampled k-space data and optional RSS targets from the FastMRI dataset.

   * - :class:`SimpleFastMRISliceDataset <deepinv.datasets.SimpleFastMRISliceDataset>`
     - `x`
     - 973 (knee) and 455 (brain) images
     - 320x320 fully-sampled reconstructed slices
     - Easy-to-use in-memory prepared subset of 2D slices from the full FastMRI slice dataset for knees and brains, padded to standard size.

   * - :class:`CMRxReconSliceDataset <deepinv.datasets.CMRxReconSliceDataset>`
     - `(x, y)` or `(x, y, {'mask': mask})`
     - 300 patients, each with 8-13 slices
     - Padded to 512x256x12 time steps
     - Dynamic MRI sequences of cardiac cine from short axis (5-10 slices) and long axis (3 views) split by patient, from the CMRxRecon challenge.

   * - :class:`SKMTEASliceDataset <deepinv.datasets.SKMTEASliceDataset>`
     - `(x, y, {'mask': mask, 'coil_maps': coil_maps})`
     - 25,000 slices from 155 patients
     - Complex double-echo with 8 coils of shape 512x160.
     - Raw MRI knee multicoil undersampled k-space data and fully-sampled ground truth from the Stanford SKM-TEA dataset, with precomputed Poisson disc masks from 4x to 16x acceleration, and pre-estimated coil maps.

   * - :class:`LidcIdriSliceDataset <deepinv.datasets.LidcIdriSliceDataset>`
     - `x`
     - Over 200,000 CT scan slices
     - Slices 512x512 voxels
     - A comprehensive dataset of lung CT scans with annotations, used for medical image processing and lung cancer detection research.

   * - :class:`Flickr2kHR <deepinv.datasets.Flickr2kHR>`
     - `x`
     - 2,650 images
     - RGB, up to 2000x2000 pixels (variable)
     - A dataset from Flickr containing high-resolution images for tasks like super-resolution and image restoration.

   * - :class:`LsdirHR <deepinv.datasets.LsdirHR>`
     - `x`
     - 84499 (train) + 1000 (val) images
     - RGB, up to 2160x2160 pixels (variable)
     - A dataset with high-resolution images, often used for training large reconstruction models.

   * - :class:`FMD <deepinv.datasets.FMD>`
     - `x`
     - 12000 images
     - 512x512 pixels
     - The Fluorescence Microscopy Dataset (FMD) is a dataset of real fluorescence microscopy images.

   * - :class:`Kohler <deepinv.datasets.Kohler>`
     - `(x, y)`
     - 48 blurry + 9547 sharp images
     - 800x800 RGB
     - A blind-deblurring dataset consists of blurry shots and sharp frames, each blurry shot being associated with about 200 sharp frames.

   * - :class:`NBUDataset <deepinv.datasets.NBUDataset>`
     - `x` Tensor or TensorList
     - 510 images across 6 satellites
     - Cx256x256 multispectral (C=4 or 8) and 1x1024x1024 panchromatic
     - Multispectral satellite images of urban scenes from 6 different satellites.


.. _data-transforms:

Data Transforms
---------------

We provide some torchvision-style transforms for use when loading data:

.. list-table:: Data Transforms Overview
   :header-rows: 1

   * - **Transform**
     - **Description**
   * - :class:`deepinv.datasets.utils.Rescale`
     - Min-max or clip value rescaling.
   * - :class:`deepinv.datasets.utils.ToComplex`
     - Add empty imaginary dimension to image.
   * - :class:`deepinv.datasets.utils.Crop`
     - Crop image in corner or with arbitrary crop position and/or size.
   * - :class:`deepinv.datasets.MRISliceTransform`
     - Transform raw FastMRI data by simulating masks and estimating coil maps.
