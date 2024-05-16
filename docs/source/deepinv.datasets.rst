.. _datasets:

Datasets
========

This subpackage can be used for generating reconstruction datasets from other base datasets (e.g. MNIST or CelebA).


HD5Dataset
----------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.HDF5Dataset
    deepinv.datasets.generate_dataset


Generating a dataset associated with a certain forward operator is done via :func:`deepinv.datasets.generate_dataset`
using a base dataset (in this case MNIST). For example, here we generate a compressed sensing MNIST dataset:

.. doctest::

    >>> import deepinv as dinv
    >>> from torchvision import datasets, transforms
    >>>
    >>> save_dir = '../datasets/MNIST/' # directory where the dataset will be saved.
    >>>
    >>> # define base train dataset
    >>> transform_data = transforms.Compose([transforms.ToTensor()])
    >>> data_train = datasets.MNIST(root='../datasets/', train=True,
    ...                             transform=transform_data, download=True)
    >>> data_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)
    >>>
    >>> # define forward operator
    >>> physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28))
    >>> physics.noise_model = dinv.physics.GaussianNoise(sigma=.05)
    >>>
    >>> # generate paired dataset
    >>> generated_dataset_path = dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
    ...                                physics=physics, save_dir=save_dir, verbose=False)


Similarly, we can generate a dataset from a local folder of images (other types of data can be loaded using the ``loader``
and ``is_valid_file`` arguments of :meth:`torchvision.datasets.ImageFolder``):

.. doctest::

    >>> # Note that ImageFolder requires file structure to be '.../dir/train/xxx/yyy.ext' where xxx is an arbitrary class label
    >>> data_train = datasets.ImageFolder(f'{save_dir}/train', transform=transform_data)
    >>> data_test  = datasets.ImageFolder(f'{save_dir}/test',  transform=transform_data)
    >>>
    >>> dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
    >>>                                physics=physics, device=dinv.device, save_dir=save_dir)

The datasets are saved in ``.h5`` (HDF5) format, and can be easily loaded to pytorch's standard
:class:`torch.utils.data.DataLoader`:

.. doctest::

    >>> from torch.utils.data import DataLoader
    >>>
    >>> dataset = dinv.datasets.HDF5Dataset(path=generated_dataset_path, train=True)
    >>> dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

PatchDataset
------------

Generate a dataset of all patches out of a tensor of images.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.PatchDataset

Image Datasets
--------------

Ready-made datasets available in the `deepinv.datasets` module.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.datasets.DIV2K