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

::

    import deepinv as dinv
    from torchvision import datasets, transforms

    dir = '../datasets/MNIST/' # directory where the dataset will be saved.

    # define base train dataset
    transform_data = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST(root='../datasets/', train=True, transform=transform_data, download=True)
    data_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)

    # define forward operator
    physics = dinv.physics.CompressedSensing(m=300, img_shape=(1, 28, 28), device=dinv.device)
    physics.noise_model = dinv.physics.GaussianNoise(sigma=.05)

    # generate paired dataset
    dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                                   physics=physics, device=dinv.device, save_dir=dir)


The datasets are saved in ``.h5`` (HDF5) format, and can be easily loaded to pytorch's standard
:class:`torch.utils.data.DataLoader`:

::

    from torch.utils.data import DataLoader

    dataset = dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset.h5', train=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
