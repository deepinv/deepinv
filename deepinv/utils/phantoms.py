from __future__ import annotations
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

    :param float length: Length of the dataset.
    :param int size: Size of the phantom (square) image.
    :param int n_data: Number of phantoms to generate per sample.
    :param Callable transform: Transformation to apply to the output image.
    """

    def __init__(self, length: int, size: int = 128, n_data: int = 1, transform=None):
        self.size = size
        self.n_data = n_data
        self.transform = transform
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        :return: A torch.Tensor of shape (n_data, size, size).
        """
        x = torch.stack(
            [generate_random_phantom(self.size) for _ in range(self.n_data)]
        )

        if self.transform is not None:
            x = self.transform(x)

        return x


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

    :param int size: Size of the phantom (square) image.
    :param int n_data: Number of phantoms to generate per sample.
    :param Callable transform: Transformation to apply to the output image.
    """

    def __init__(self, size=128, n_data=1, transform=None):
        self.size = size
        self.n_data = n_data
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """
        :return: A torch.Tensor of shape (n_data, size, size).
        """
        x = torch.stack([generate_shepp_logan(self.size) for _ in range(self.n_data)])

        if self.transform is not None:
            x = self.transform(x)

        return x


# NEMA IEC body phantom geometry, in millimetres, following the specification
# used by OpenGATE (opengate/contrib/phantoms/nemaiec.py).
NEMA_IEC_SPHERE_DIAMETERS_MM = (10.0, 13.0, 17.0, 22.0, 28.0, 37.0)
NEMA_IEC_RING_RADIUS_MM = 57.27  # radius of the circle the sphere centres sit on
NEMA_IEC_LUNG_RADIUS_MM = 25.0  # central cold cylindrical insert
NEMA_IEC_START_ANGLE_DEG = 180.0  # angular position of the first (smallest) sphere
NEMA_IEC_BOX_HALF_X_MM = 100.0  # half-width of the surrounding box
NEMA_IEC_BOX_HALF_Y_MM = 90.0  # half-height of the surrounding box
NEMA_IEC_FOV_HALF_MM = 110.0  # physical half-extent mapped to the [-1, 1] grid


def generate_nema_iec_phantom(
    size: int,
    activities: list[float] | None = None,
    background: float = 0.25,
    lung: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    r"""
    Generate a 2D NEMA IEC body phantom.

    The NEMA IEC body phantom is a standard image-quality phantom commonly used
    in PET and SPECT. It consists of six spheres of increasing diameter
    (10, 13, 17, 22, 28 and 37 mm) arranged on a circle inside a uniform
    background, with a central cold cylindrical "lung" insert. This function
    returns the 2D transverse cross-section through the centre of the six
    spheres, with the body approximated by a rectangular box.

    The activity level of each sphere can be set individually via ``activities``.

    .. note::

        This is a synthetic phantom intended for demonstrations and testing of
        reconstruction algorithms (e.g. with :class:`deepinv.physics.Tomography`),
        not a physically accurate simulation of a PET/SPECT acquisition.

    |sep|

    :Example:

    >>> import deepinv as dinv
    >>> x = dinv.utils.phantoms.generate_nema_iec_phantom(64)
    >>> x.shape
    torch.Size([64, 64])

    :param int size: size of the (square) phantom image.
    :param list[float] activities: list of 6 activity levels, one per sphere ordered
        by increasing diameter. Defaults to ``[1.0] * 6`` (all spheres equally active).
    :param float background: activity level of the uniform background inside the box.
    :param float lung: activity level of the central cold cylindrical insert.
    :param bool normalize: if ``True``, the phantom is rescaled so that its maximum
        value is 1.
    :return: (:class:`torch.Tensor`) a phantom of shape ``(size, size)``.
    """
    n_spheres = len(NEMA_IEC_SPHERE_DIAMETERS_MM)
    if activities is None:
        activities = [1.0] * n_spheres
    if len(activities) != n_spheres:
        raise ValueError(
            f"activities must have length {n_spheres}, got {len(activities)}."
        )

    scale = 1.0 / NEMA_IEC_FOV_HALF_MM
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )

    phantom = torch.zeros(size, size)

    # body box filled with uniform background activity
    box = (x.abs() <= NEMA_IEC_BOX_HALF_X_MM * scale) & (
        y.abs() <= NEMA_IEC_BOX_HALF_Y_MM * scale
    )
    phantom[box] = background

    # six spheres arranged on a circle, ordered by increasing diameter
    ring_radius = NEMA_IEC_RING_RADIUS_MM * scale
    for i, diameter in enumerate(NEMA_IEC_SPHERE_DIAMETERS_MM):
        angle = np.deg2rad(NEMA_IEC_START_ANGLE_DEG + i * 360 / n_spheres)
        x_0 = ring_radius * np.cos(angle)
        y_0 = ring_radius * np.sin(angle)
        radius = (diameter / 2) * scale
        mask = ((x - x_0) ** 2 + (y - y_0) ** 2) <= radius**2
        phantom[mask] = activities[i]

    # central cold cylindrical "lung" insert
    lung_mask = (x**2 + y**2) <= (NEMA_IEC_LUNG_RADIUS_MM * scale) ** 2
    phantom[lung_mask] = lung

    if normalize:
        max_val = phantom.max()
        if max_val > 0:
            phantom = phantom / max_val

    return phantom


class NEMAIECPhantomDataset(Dataset):
    r"""
    Dataset for the NEMA IEC body phantom. The dataset has length 1.

    See :func:`deepinv.utils.phantoms.generate_nema_iec_phantom` for details on
    the phantom and its parameters.

    :param int size: Size of the phantom (square) image.
    :param int n_data: Number of phantoms to generate per sample.
    :param list[float] activities: list of 6 sphere activity levels, ordered by
        increasing diameter. Defaults to ``[1.0] * 6``.
    :param float background: activity level of the uniform background.
    :param float lung: activity level of the central cold insert.
    :param bool normalize: if ``True``, rescale so that the maximum value is 1.
    :param Callable transform: Transformation to apply to the output image.
    """

    def __init__(
        self,
        size: int = 128,
        n_data: int = 1,
        activities: list[float] | None = None,
        background: float = 0.25,
        lung: float = 0.0,
        normalize: bool = True,
        transform=None,
    ):
        self.size = size
        self.n_data = n_data
        self.activities = activities
        self.background = background
        self.lung = lung
        self.normalize = normalize
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        """
        :return: A torch.Tensor of shape (n_data, size, size).
        """
        x = torch.stack(
            [
                generate_nema_iec_phantom(
                    self.size,
                    activities=self.activities,
                    background=self.background,
                    lung=self.lung,
                    normalize=self.normalize,
                )
                for _ in range(self.n_data)
            ]
        )

        if self.transform is not None:
            x = self.transform(x)

        return x
