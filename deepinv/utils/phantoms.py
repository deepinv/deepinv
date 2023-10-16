import numpy as np
import odl
import torch

# def _getshapes_2d(center, max_radius, shape):
#     """Calculate indices and slices for the bounding box of a disk."""
#     index_mean = shape * center
#     index_radius = max_radius / 2.0 * np.array(shape)

#     # Avoid negative indices
#     min_idx = np.maximum(np.floor(index_mean - index_radius), 0).astype(int)
#     max_idx = np.ceil(index_mean + index_radius).astype(int)
#     idx = [slice(minx, maxx) for minx, maxx in zip(min_idx, max_idx)]
#     shapes = [(idx[0], slice(None)),
#               (slice(None), idx[1])]
#     return tuple(idx), tuple(shapes)

# def _ellipse_phantom_2d(space, ellipses):
#     """Create a phantom of ellipses in 2d space.

#     Parameters
#     ----------
#     space : `DiscretizedSpace`
#         Uniformly discretized space in which the phantom should be generated.
#         If ``space.shape`` is 1 in an axis, a corresponding slice of the
#         phantom is created (instead of squashing the whole phantom into the
#         slice).
#     ellipses : list of lists
#         Each row should contain the entries ::

#             'value',
#             'axis_1', 'axis_2',
#             'center_x', 'center_y',
#             'rotation'

#         The provided ellipses need to be specified relative to the
#         reference rectangle ``[-1, -1] x [1, 1]``. Angles are to be given
#         in radians.

#     Returns
#     -------
#     phantom : ``space`` element
#         2D ellipse phantom in ``space``.
#     """
#     # Blank image
#     p = np.zeros(space.shape, dtype=space.dtype)

#     minp = space.grid.min_pt
#     maxp = space.grid.max_pt

#     # Create the pixel grid
#     grid_in = space.grid.meshgrid

#     # move points to [-1, 1]
#     grid = []
#     for i in range(2):
#         mean_i = (minp[i] + maxp[i]) / 2.0
#         # Where space.shape = 1, we have minp = maxp, so we set diff_i = 1
#         # to avoid division by zero. Effectively, this allows constructing
#         # a slice of a 2D phantom.
#         diff_i = (maxp[i] - minp[i]) / 2.0 or 1.0
#         grid.append((grid_in[i] - mean_i) / diff_i)

#     for ellip in ellipses:
#         assert len(ellip) == 6

#         intensity = ellip[0]
#         a_squared = ellip[1] ** 2
#         b_squared = ellip[2] ** 2
#         x0 = ellip[3]
#         y0 = ellip[4]
#         theta = ellip[5]

#         scales = [1 / a_squared, 1 / b_squared]
#         center = (np.array([x0, y0]) + 1.0) / 2.0

#         # Create the offset x,y and z values for the grid
#         if theta != 0:
#             # Rotate the points to the expected coordinate system.
#             ctheta = np.cos(theta)
#             stheta = np.sin(theta)

#             mat = np.array([[ctheta, stheta],
#                             [-stheta, ctheta]])

#             # Calculate the points that could possibly be inside the volume
#             # Since the points are rotated, we cannot do anything directional
#             # without more logic
#             max_radius = np.sqrt(
#                 np.abs(mat).dot([a_squared, b_squared]))
#             idx, shapes = _getshapes_2d(center, max_radius, space.shape)

#             subgrid = [g[idi] for g, idi in zip(grid, shapes)]
#             offset_points = [vec * (xi - x0i)[..., None]
#                              for xi, vec, x0i in zip(subgrid,
#                                                      mat.T,
#                                                      [x0, y0])]
#             rotated = offset_points[0] + offset_points[1]
#             np.square(rotated, out=rotated)
#             radius = np.dot(rotated, scales)
#         else:
#             # Calculate the points that could possibly be inside the volume
#             max_radius = np.sqrt([a_squared, b_squared])
#             idx, shapes = _getshapes_2d(center, max_radius, space.shape)

#             subgrid = [g[idi] for g, idi in zip(grid, shapes)]
#             squared_dist = [ai * (xi - x0i) ** 2
#                             for xi, ai, x0i in zip(subgrid,
#                                                    scales,
#                                                    [x0, y0])]

#             # Parentheses to get best order for broadcasting
#             radius = squared_dist[0] + squared_dist[1]

#         # Find the points within the ellipse
#         inside = radius <= 1

#         # Add the ellipse intensity to those points
#         p[idx][inside] += intensity

#     return space.element(p)

# def ellipsoid_phantom(space, ellipsoids, min_pt=None, max_pt=None):
#     """Return a phantom given by ellipsoids.

#     Code taken from the OLD library : https://github.com/odlgroup/odl/blob/master/odl/phantom/geometric.py#L580

#     Parameters
#     ----------
#     space : `DiscretizedSpace`
#         Space in which the phantom should be created, must be 2- or
#         3-dimensional. If ``space.shape`` is 1 in an axis, a corresponding
#         slice of the phantom is created (instead of squashing the whole
#         phantom into the slice).
#     ellipsoids : sequence of sequences
#         If ``space`` is 2-dimensional, each row should contain the entries ::

#             'value',
#             'axis_1', 'axis_2',
#             'center_x', 'center_y',
#             'rotation'

#         If ``space`` is 3-dimensional, each row should contain the entries ::

#             'value',
#             'axis_1', 'axis_2', 'axis_3',
#             'center_x', 'center_y', 'center_z',
#             'rotation_phi', 'rotation_theta', 'rotation_psi'

#         The provided ellipsoids need to be specified relative to the
#         reference rectangle ``[-1, -1] x [1, 1]``, or analogously in 3d.
#         The angles are to be given in radians.

#     min_pt, max_pt : array-like, optional
#         If provided, use these vectors to determine the bounding box of the
#         phantom instead of ``space.min_pt`` and ``space.max_pt``.
#         It is currently required that ``min_pt >= space.min_pt`` and
#         ``max_pt <= space.max_pt``, i.e., shifting or scaling outside the
#         original space is not allowed.

#         Providing one of them results in a shift, e.g., for ``min_pt``::

#             new_min_pt = min_pt
#             new_max_pt = space.max_pt + (min_pt - space.min_pt)

#         Providing both results in a scaled version of the phantom.
#     """
#     _phantom = _ellipse_phantom_2d

#     if min_pt is None and max_pt is None:
#         return _phantom(space, ellipsoids)

#     else:
#         # Generate a temporary space with given `min_pt` and `max_pt`
#         # (snapped to the cell grid), create the phantom in that space and
#         # resize to the target size for `space`.
#         # The snapped points are constructed by finding the index of
#         # `min/max_pt` in the space partition, indexing the partition with
#         # that index, yielding a single-cell partition, and then taking
#         # the lower-left/upper-right corner of that cell.
#         if min_pt is None:
#             snapped_min_pt = space.min_pt
#         else:
#             min_pt_cell = space.partition[space.partition.index(min_pt)]
#             snapped_min_pt = min_pt_cell.min_pt

#         if max_pt is None:
#             snapped_max_pt = space.max_pt
#         else:
#             max_pt_cell = space.partition[space.partition.index(max_pt)]
#             snapped_max_pt = max_pt_cell.max_pt
#             # Avoid snapping to the next cell where max_pt falls exactly on
#             # a boundary
#             for i in range(space.ndim):
#                 if max_pt[i] in space.partition.cell_boundary_vecs[i]:
#                     snapped_max_pt[i] = max_pt[i]

#         tmp_space = uniform_discr_fromdiscr(
#             space, min_pt=snapped_min_pt, max_pt=snapped_max_pt,
#             cell_sides=space.cell_sides)

#         tmp_phantom = _phantom(tmp_space, ellipsoids)
#         offset = space.partition.index(tmp_space.min_pt)
#         return space.element(
#             resize_array(tmp_phantom, space.shape, offset))



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

    return ((np.random.rand() - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc, n_ellipse=50, interior=False):
    """
    Generate a random ellipsoid phantom. 
    Taken from https://github.com/adler-j/adler/blob/master/adler/odl/phantom.py
    """
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes(interior=interior) for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, shapes)


class RandomPhantomDataset(torch.utils.data.Dataset):
    def __init__(self, size=128, transform=None, length=np.inf):
        self.space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        phantom_np = np.array(random_phantom(self.space))
        phantom = torch.from_numpy(phantom_np).float().unsqueeze(0)
        if self.transform is not None:
            phantom = self.transform(phantom)
        return phantom, 0
    
class SheppLoganDataset(torch.utils.data.Dataset):
    def __init__(self, size=128, transform=None):
        self.space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, index):
        phantom_np = odl.phantom.shepp_logan(self.space, True)
        phantom = torch.from_numpy(phantom_np).float()
        if self.transform is not None:
            phantom = self.transform(phantom)
        return phantom, 0