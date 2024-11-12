import numpy as np
import torch

try:
    import odl
except:
    odl = ImportError("The odl package is not installed.")


def random_shapes(interior=False):
    """
    Generate random shape parameters.
    Taken from https://github.com/adler-j/adler/blob/master/adler/odl/phantom.py
    """
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return (
        (np.random.rand() - 0.5) * np.random.exponential(0.4),
        np.random.exponential() * 0.2,
        np.random.exponential() * 0.2,
        x_0,
        y_0,
        np.random.rand() * 2 * np.pi,
    )


def random_phantom(spc, n_ellipse=50, interior=False):
    """
    Generate a random ellipsoid phantom.
    Taken from https://github.com/adler-j/adler/blob/master/adler/odl/phantom.py
    """
    if isinstance(odl, ImportError):
        raise ImportError(
            "odl is needed to use generate random phantoms. "
            "It should be installed with `python3 -m pip install"
            " https://github.com/odlgroup/odl/archive/master.zip`"
        ) from odl
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes(interior=interior) for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, shapes)


class RandomPhantomDataset(torch.utils.data.Dataset):
    """
    Dataset of random ellipsoid phantoms. The phantoms are generated on the fly.
    The phantoms are generated using the odl library (https://odlgroup.github.io/odl/).

    :param int size: Size of the phantom (square) image.
    :param int n_data: Number of phantoms to generate per sample.
    :param transform: Transformation to apply to the output image.
    :param float length: Length of the dataset. Useful for iterating the data-loader for a certain nb of iterations.
    """

    def __init__(self, size=128, n_data=1, transform=None, length=np.inf):
        if isinstance(odl, ImportError):
            raise ImportError(
                "odl is needed to use generate random phantoms. "
                "It should be installed with `python3 -m pip install"
                " https://github.com/odlgroup/odl/archive/master.zip`"
            ) from odl
        self.space = odl.uniform_discr(
            [-64, -64], [64, 64], [size, size], dtype="float32"
        )
        self.transform = transform
        self.n_data = n_data
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        :return tuple : A tuple (phantom, 0) where phantom is a torch tensor of shape (n_data, size, size).
        """
        phantom_np = np.array([random_phantom(self.space) for i in range(self.n_data)])
        phantom = torch.from_numpy(phantom_np).float()
        if self.transform is not None:
            phantom = self.transform(phantom)
        return phantom, 0


class SheppLoganDataset(torch.utils.data.Dataset):
    """
    Dataset for the single Shepp-Logan phantom. The dataset has length 1.
    """

    def __init__(self, size=128, n_data=1, transform=None):
        if isinstance(odl, ImportError):
            raise ImportError(
                "odl is needed to use generate the Shepp Logan phantom. "
                "It should be installed with `python3 -m pip install"
                " https://github.com/odlgroup/odl/archive/master.zip`"
            ) from odl
        self.space = odl.uniform_discr(
            [-64, -64], [64, 64], [size, size], dtype="float32"
        )
        self.transform = transform
        self.n_data = n_data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if isinstance(odl, ImportError):
            raise ImportError(
                "odl is needed to use generate the Shepp Logan phantom. "
                "It should be installed with `python3 -m pip install"
                " https://github.com/odlgroup/odl/archive/master.zip`"
            ) from odl
        phantom_np = np.array(
            [odl.phantom.shepp_logan(self.space, True) for i in range(self.n_data)]
        )
        phantom = torch.from_numpy(phantom_np).float()
        if self.transform is not None:
            phantom = self.transform(phantom)
        return phantom, 0
