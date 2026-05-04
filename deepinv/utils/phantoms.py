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


def generate_pet_phantom(
    img_shape: tuple[int, int, int] | tuple[int, int],
    mu_value: float = 0.01,
    add_spheres: bool = True,
    add_inner_cylinder: bool = True,
    r0: float = 0.45,
    r1: float = 0.28,
    oversampling_factor=4,
    device: str = "cpu",
):
    r"""
    Generate a 2D or 3D PET-like phantom and its corresponding attenuation map.

    The phantom mimics a simplified PET scanner body phantom consisting of:

    - *Outer elliptical body*: a large elliptical cylinder (or ellipsoid in 3D)
      filled with uniform emission activity (value 1.0) and attenuation ``mu_value``.
      Its semi-axes are ``r0`` (along the first spatial dimension) and ``r1`` (along
      the second spatial dimension), expressed as fractions of the respective image size.
    - *Inner cold rod* (optional, ``add_inner_cylinder=True``): a small elliptical
      cylinder at the centre of the body with lower emission activity (value 0.25) and
      reduced attenuation (``mu_value / 3``), representing a low-activity insert.
    - *Hot and cold spheres* (optional, ``add_spheres=True``): four pairs of spheres
      placed at two axial offsets inside the body. *Hot* spheres (value 2.5) simulate
      lesion-like regions with elevated metabolic activity; *cold* spheres (value 0.25)
      simulate photopenic defects (e.g. cysts or necrotic tissue).

    The function generates the phantom at ``oversampling_factor`` times the requested
    resolution and then downsamples by averaging, reducing partial-volume effects at
    the boundaries.

    For 2D inputs (``img_shape = (H, W)``), the phantom is generated as a 3D volume
    and the central axial slice is returned.

    :param tuple img_shape: Shape of the output image ``(H, W)`` or volume ``(D, H, W)``.
    :param float mu_value: Attenuation coefficient of the outer body (default: 0.01).
    :param bool add_spheres: If ``True``, hot and cold spheres are added inside the body
        (default: ``True``).
    :param bool add_inner_cylinder: If ``True``, a low-activity inner rod is added at the
        centre of the body (default: ``True``).
    :param float r0: Fractional semi-axis of the outer ellipse along the *first* spatial
        dimension (D or H for 3D/2D inputs). Must satisfy ``0 < r0 <= 0.5`` so that the
        ellipse fits within the image. Default: 0.45.
    :param float r1: Fractional semi-axis of the outer ellipse along the *second* spatial
        dimension (H or W for 3D/2D inputs). Must satisfy ``0 < r1 <= 0.5`` so that the
        ellipse fits within the image. If ``add_spheres=True``, ``r1`` should be at least
        ~0.25 to ensure the off-centre spheres remain inside the body. Default: 0.28.
    :param int oversampling_factor: Upsampling factor used during phantom generation to
        reduce partial-volume artefacts (default: 4).
    :param str or torch.device device: Device on which tensors are allocated (default:
        ``"cpu"``).
    :returns: Tuple ``(x_em, x_att)`` where both tensors have shape ``(1, 1, H, W)``
        for 2D inputs or ``(1, 1, D, H, W)`` for 3D inputs. ``x_em`` is the emission
        activity map and ``x_att`` is the attenuation map.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    :raises ValueError: If ``r0`` or ``r1`` are outside the valid range ``(0, 0.5]``.

    |sep|

    :Example:

    >>> from deepinv.utils.phantoms import generate_pet_phantom
    >>> x_em, x_att = generate_pet_phantom(img_shape=(64, 64, 32))
    >>> print(x_em.shape, x_att.shape)
    torch.Size([1, 1, 64, 64, 32]) torch.Size([1, 1, 64, 64, 32])
    """
    if not (0 < r0 <= 0.5):
        raise ValueError(
            f"r0={r0} is out of range. r0 must satisfy 0 < r0 <= 0.5 so that the outer "
            f"ellipse fits within the image along the first spatial dimension. "
            f"Typical value: r0=0.45."
        )
    if not (0 < r1 <= 0.5):
        raise ValueError(
            f"r1={r1} is out of range. r1 must satisfy 0 < r1 <= 0.5 so that the outer "
            f"ellipse fits within the image along the second spatial dimension. "
            f"Typical value: r1=0.28."
        )
    if add_spheres and r1 < 0.25:
        raise ValueError(
            f"r1={r1} is too small when add_spheres=True. The off-centre spheres are "
            f"placed at 1.4 × (image_height / 2), which requires r1 >= 0.25 to remain "
            f"inside the outer ellipse. Either increase r1 or set add_spheres=False."
        )
    if add_spheres and r0 < 0.25:
        raise ValueError(
            f"r0={r0} is too small when add_spheres=True. The off-centre spheres are "
            f"placed at 1.3 × (image_depth / 2), which requires r0 >= 0.25 to remain "
            f"inside the outer ellipse. Either increase r0 or set add_spheres=False."
        )
    if len(img_shape) == 2:
        keep_center_slice = True
        img_shape = img_shape + (32,)
    else:
        keep_center_slice = False

    D, H, W = img_shape
    od, oh, ow = [oversampling_factor * x for x in img_shape]
    x_em = torch.zeros((od, oh, ow), dtype=torch.float32, device=device)
    x_att = torch.zeros_like(x_em)

    c0 = od / 2
    c1 = oh / 2
    c2 = ow / 2

    a = r0 * od
    b = r1 * oh

    rix = od / 25
    riy = oh / 25

    y, x = torch.meshgrid(
        torch.arange(od, device=device),
        torch.arange(oh, device=device),
        indexing="ij",
    )

    outer_mask = ((x - c0) / a) ** 2 + ((y - c1) / b) ** 2 <= 1
    inner_mask = ((x - c0) / rix) ** 2 + ((y - c1) / riy) ** 2 <= 1

    for z in range(ow):
        x_em[:, :, z][outer_mask] = 1.0
        x_att[:, :, z][outer_mask] = mu_value

        if add_inner_cylinder:
            x_em[:, :, z][inner_mask] = 0.25
            x_att[:, :, z][inner_mask] = mu_value / 3

    if add_spheres:
        x, y, z = torch.meshgrid(
            torch.arange(od, device=device),
            torch.arange(oh, device=device),
            torch.arange(ow, device=device),
            indexing="ij",
        )

        r_sp = [ow / 9] * 3
        r_sp2 = [ow / 17] * 3

        for z_offset in [c2, 0.45 * c2]:

            sp_mask = ((x - c0) / r_sp[0]) ** 2 + ((y - 1.4 * c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em[sp_mask] = 2.5

            sp_mask2 = ((x - 1.3 * c0) / r_sp[0]) ** 2 + ((y - c1) / r_sp[1]) ** 2 + (
                (z - z_offset) / r_sp[2]
            ) ** 2 <= 1
            x_em[sp_mask2] = 0.25

            sp_mask = ((x - c0) / r_sp2[0]) ** 2 + ((y - 0.6 * c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em[sp_mask] = 2.5

            sp_mask2 = ((x - 0.7 * c0) / r_sp2[0]) ** 2 + ((y - c1) / r_sp2[1]) ** 2 + (
                (z - z_offset) / r_sp2[2]
            ) ** 2 <= 1
            x_em[sp_mask2] = 0.25

    # downsample by averaging
    f = oversampling_factor

    def downsample(v):
        v = v.view(D, f, H, f, W, f)
        v = v.sum(dim=(1, 3, 5))
        return v / (f**3)

    x_em = downsample(x_em)
    x_att = downsample(x_att)

    x_em[:, :, :3] = 0
    x_em[:, :, -3:] = 0

    x_att[:, :, :2] = 0
    x_att[:, :, -2:] = 0

    if keep_center_slice:
        x_em = x_em[..., x_em.size(-1) // 2]
        x_att = x_att[..., x_att.size(-1) // 2]

    # add batch + channel
    x_em = x_em.unsqueeze(0).unsqueeze(0)
    x_att = x_att.unsqueeze(0).unsqueeze(0)

    return x_em, x_att
