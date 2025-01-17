.. _datasets:

Datasets
========

This subpackage can be used for generating reconstruction datasets from other base datasets (e.g. MNIST or CelebA).


.. _generating-datasets:

Generating Datasets
-------------------
Generating a dataset associated with a certain forward operator is done via :func:`deepinv.datasets.generate_dataset`
using a base PyTorch dataset (:class:`torch.utils.data.Dataset`).
Your base dataset can either be one of our predefined datasets (see :ref:`predefined-datasets`), 
your own data (which you can load using :class:`torchvision.datasets.ImageFolder` or a custom dataset),
or other external datasets (e.g. in this case, torchvision MNIST).

For example, here we generate a compressed sensing MNIST dataset:

.. note::

    We support all data types supported by ``h5py``, including complex numbers.

::

    import deepinv as dinv
    from torchvision import datasets, transforms

    # directory where the dataset will be saved.
    save_dir = dinv.utils.demo.get_data_home() / 'MNIST'

    # define base train dataset
    transform_data = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST(root=save_dir, train=True,
                                transform=transform_data, download=True)
    data_test = datasets.MNIST(root=save_dir, train=False, transform=transform_data)

    # define forward operator
    physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    physics.noise_model = dinv.physics.GaussianNoise(sigma=.05)

    # generate paired dataset
    generated_dataset_path = dinv.datasets.generate_dataset(train_dataset=data_train,
            test_dataset=data_test, physics=physics, save_dir=save_dir, verbose=False)

Similarly, we can generate a dataset from a local folder of images (other types of data can be loaded using the ``loader``
and ``is_valid_file`` arguments of :class:`torchvision.datasets.ImageFolder`):

::

    # Note that ImageFolder requires file structure to be '.../dir/train/xxx/yyy.ext'
    # where xxx is an arbitrary class label
    data_train = datasets.ImageFolder(f'{save_dir}/train', transform=transform_data)
    data_test  = datasets.ImageFolder(f'{save_dir}/test',  transform=transform_data)

    dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                                   physics=physics, save_dir=save_dir)


The datasets are saved in ``.h5`` (HDF5) format, and can be easily loaded to pytorch's standard
:class:`torch.utils.data.DataLoader`:

::

    from torch.utils.data import DataLoader

    dataset = dinv.datasets.HDF5Dataset(path=generated_dataset_path, train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

We can also use physics generators to randomly generate physics params for data,
and save and load the physics params into the dataset:

::

    physics_generator = dinv.physics.generator.SigmaGenerator()
    path = dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                                          physics=physics, physics_generator=physics_generator,
                                          save_dir=save_dir)
    dataset = dinv.datasets.HDF5Dataset(path=path, load_physics_generator_params=True, train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    x, y, params = next(iter(dataloader))
    print(params['sigma'].shape)


.. _predefined-datasets:

Predefined Datasets
-------------------
Multiple popular easy-to-download datasets are available:


.. list-table:: Datasets Overview
   :header-rows: 1

   * - **Dataset**
     - **Dataset Size**
     - **Tensor Sizes**
     - **Description**

   * - :class:`deepinv.datasets.DIV2K`
     - 800 (train) + 100 (val) images
     - RGB, up to 2040x2040 pixels (variable)
     - A widely-used dataset for natural image restoration.

   * - :class:`deepinv.datasets.Urban100HR`
     - 100 images
     - up to 1200x1280 pixels (variable)
     - Contains diverse high-resolution urban scenes, typically used for testing super-resolution algorithms.

   * - :class:`deepinv.datasets.Set14HR`
     - 14 high-resolution images
     - RGB, 248×248 to 512×768 pixels.
     - A small benchmark dataset for super-resolution tasks, containing a variety of natural images.

   * - :class:`deepinv.datasets.CBSD68`
     - 68 images
     - RGB, 481x321 pixels
     - A subset of the Berkeley Segmentation Dataset.

   * - :class:`deepinv.datasets.FastMRISliceDataset`
     - Over 100,000 MRI slices
     - Complex numbers, 320x320 pixels
     - A large-scale dataset of MRI knee and brain scans for training and evaluating MRI reconstruction methods.

   * - :class:`deepinv.datasets.SimpleFastMRISliceDataset`
     - 973 (knee) and 455 (brain) images
     - 320x320 fully-sampled reconstructed slices
     - Easy-to-use in-memory prepared subset of 2D slices from the full FastMRI slice dataset for knees and brains.

   * - :class:`deepinv.datasets.LidcIdriSliceDataset`
     - Over 200,000 CT scan slices
     - Slices 512x512 voxels
     - A comprehensive dataset of lung CT scans with annotations, used for medical image processing and lung cancer detection research.

   * - :class:`deepinv.datasets.Flickr2kHR`
     - 2,650 images
     - RGB, up to 2000x2000 pixels (variable)
     - A dataset from Flickr containing high-resolution images for tasks like super-resolution and image restoration.

   * - :class:`deepinv.datasets.LsdirHR`
     - 84499 (train) + 1000 (val) images
     - RGB, up to 2160x2160 pixels (variable)
     - A dataset with high-resolution images, often used for training large reconstruction models.

   * - :class:`deepinv.datasets.FMD`
     - 12000 images
     - 512x512 pixels
     - The Fluorescence Microscopy Dataset (FMD) is a dataset of real fluorescence microscopy images.

   * - :class:`deepinv.datasets.Kohler`
     - 48 blurry + 9547 sharp images
     - 800x800 RGB
     - A blind-deblurring dataset consists of blurry shots and sharp frames, each blurry shot being associated with about 200 sharp frames.

   * - :class:`deepinv.datasets.NBUDataset`
     - 510 images across 6 satellites
     - Cx256x256 multispectral (C=4 or 8) and 1x1024x1024 panchromatic
     - Multispectral satellite images of urban scenes from 6 different satellites.


.. _data-transforms:

Data Transforms
---------------

We provide some torchvision-style transforms for use when loading data:

.. list-table:: Datasets Overview
   :header-rows: 1

   * - **Transform**
     - **Description**
   * - :class:`deepinv.datasets.utils.Rescale`
     - Min-max or clip value rescaling.
   * - :class:`deepinv.datasets.utils.ToComplex`
     - Add empty imaginary dimension to image.