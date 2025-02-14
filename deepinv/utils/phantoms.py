import numpy as np
import torch
from torch.utils.data import Dataset


def random_shapes(interior=False):
    r"""
    Generate random shape parameters for an ellipse.
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


def generate_random_phantom(size, n_ellipse=50, interior=False):
    """
    Generate a random ellipsoid phantom directly using torch.
    """
    phantom = torch.zeros(size, size)
    n = np.random.poisson(n_ellipse)
    for _ in range(n):
        a, b, c, x_0, y_0, theta = random_shapes(interior)
        x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size))
        x_rot = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta)
        y_rot = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta)
        mask = ((x_rot / b) ** 2 + (y_rot / c) ** 2) <= 1
        phantom += a * mask.float()
    return phantom.clamp(0, 1)


class RandomPhantomDataset(Dataset):
    r"""
    Dataset of random ellipsoid phantoms generated on the fly.
    :param int size: Size of the phantom (square) image.
    :param int n_data: Number of phantoms to generate per sample.
    :param transform: Transformation to apply to the output image.
    :param float length: Length of the dataset.
    """

    def __init__(self, size=128, n_data=1, transform=None, length=np.inf):
        self.size = size
        self.n_data = n_data
        self.transform = transform
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        :return tuple : A tuple (phantom, 0) where phantom is a torch tensor of shape (n_data, size, size).
        """
        phantom_np = torch.stack(
            [generate_random_phantom(self.size) for _ in range(self.n_data)]
        )
        if self.transform is not None:
            phantom_np = self.transform(phantom_np)
        return phantom_np, 0


def generate_shepp_logan(size):
    """
    Generate a Shepp-Logan phantom approximation in PyTorch.
    """
    ellipses = [
        (1, 0.69, 0.92, 0, 0, 0),
        (-0.8, 0.6624, 0.874, 0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0, -18 * np.pi / 180),
        (-0.2, 0.16, 0.41, -0.22, 0, 18 * np.pi / 180),
        (0.1, 0.21, 0.25, 0, 0.35, 0),
        (0.1, 0.046, 0.046, 0, 0.1, 0),
        (0.1, 0.046, 0.046, 0, -0.1, 0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0),
        (0.1, 0.023, 0.023, 0, -0.606, 0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0),
    ]
    phantom = torch.zeros(size, size)
    x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size))
    for a, b, c, x_0, y_0, theta in ellipses:
        x_rot = (x - x_0) * np.cos(theta) + (y - y_0) * np.sin(theta)
        y_rot = -(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta)
        mask = ((x_rot / b) ** 2 + (y_rot / c) ** 2) <= 1
        phantom += a * mask.float()
    phantom = phantom.transpose(-2, -1).flip(-2)
    return phantom.clamp(0, 1)


class SheppLoganDataset(Dataset):
    """
    Dataset for the single Shepp-Logan phantom. The dataset has length 1.
    """

    def __init__(self, size=128, n_data=1, transform=None):
        self.size = size
        self.n_data = n_data
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        phantom_np = torch.stack(
            [generate_shepp_logan(self.size) for _ in range(self.n_data)]
        )
        if self.transform is not None:
            phantom_np = self.transform(phantom_np)
        return phantom_np, 0
