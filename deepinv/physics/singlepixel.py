from deepinv.physics.forward import DecomposablePhysics
import torch
import numpy as np
import warnings


def hadamard_1d(u, normalize=True):
    """
    Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.

    :param torch.Tensor u: Input tensor of shape (..., n), where n must be a power of 2.
    :param bool normalize: If True, divide the result by 2^{m/2} where m = log_2(n).
    :return: The product tensor of shape (..., n).
    :rtype: torch.Tensor
    """
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, "n must be a power of 2"
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat(
            (x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1
        )
    return x.squeeze(-2) / 2 ** (m / 2) if normalize else x.squeeze(-2)


def hadamard_2d(x):
    """
    Applies the 2-dimensional Hadamard transform to the input tensor.

    This function computes the 2D Hadamard transform by applying the 1D Hadamard transform
    along one axis, transposing the result, applying the 1D Hadamard transform again, and
    then transposing back to the original axes order.

    :param torch.Tensor x: Input tensor of shape (..., n, m), where n and m are powers of 2.
    :return: The Hadamard-transformed tensor of the same shape as the input.
    :rtype: torch.Tensor
    """
    out = hadamard_1d(hadamard_1d(x).transpose(-1, -2)).transpose(-1, -2)
    return out


def hadamard_shift(x, dim):
    """
    Rearranges the Hadamard transform to sequency order along a specified dimension.

    This function reorders the elements of the Hadamard transform along the given dimension
    to follow the sequency order, which is a specific ordering of Walsh functions.

    :param torch.Tensor x: Input tensor of shape (..., n, ...), where `n` is the size along the specified dimension.
    :param int dim: The dimension along which to rearrange the Hadamard transform.
    :return: The tensor with the Hadamard transform rearranged in sequency order along the specified dimension.
    :rtype: torch.Tensor
    """
    n = x.shape[dim]
    indexs = sequency_order(n)
    x = x.index_select(dim, torch.tensor(indexs, device=x.device))
    return x


def hadamard_ishift(x, dim):
    """
    Reverses the arrangement of the Hadamard transform from sequency order along a specified dimension.

    This function undoes the sequency ordering of the Hadamard transform along the given dimension,
    restoring the original order.

    :param torch.Tensor x: Input tensor of shape (..., n, ...), where `n` is the size along the specified dimension.
    :param int dim: The dimension along which to reverse the sequency order.
    :return: The tensor with the Hadamard transform restored to its original order along the specified dimension.
    :rtype: torch.Tensor
    """
    n = x.shape[dim]
    indexs = sequency_order(n)
    indexs = np.argsort(indexs)
    x = x.index_select(dim, torch.tensor(indexs, device=x.device))
    return x


def hadamard_2d_shift(x):
    """
    Rearranges the 2D Hadamard transform to sequency order.

    This function applies the `hadamard_shift` function along both the last two dimensions
    of the input tensor to reorder the 2D Hadamard transform in sequency order.

    :param torch.Tensor x: Input tensor of shape (..., n, m), where `n` and `m` are the sizes of the last two dimensions.
    :return: The tensor with the 2D Hadamard transform rearranged in sequency order.
    :rtype: torch.Tensor
    """
    x = hadamard_shift(x, -2)
    x = hadamard_shift(x, -1)
    return x


def hadamard_2d_ishift(x):
    """
    Reverses the arrangement of the 2D Hadamard transform from sequency order.

    This function applies the `hadamard_ishift` function along both the last two dimensions
    of the input tensor to restore the original order of the 2D Hadamard transform.

    :param torch.Tensor x: Input tensor of shape (..., n, m), where `n` and `m` are the sizes of the last two dimensions.
    :return: The tensor with the 2D Hadamard transform restored to its original order.
    :rtype: torch.Tensor
    """
    x = hadamard_ishift(x, -1)
    x = hadamard_ishift(x, -2)
    return x


def sequency_mask(img_shape, m):

    _, H, W = img_shape
    n = H * W

    indexes = sequency_order(n)[:m]
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    i = i.flatten(order="F")
    j = j.flatten(order="F")

    mask = torch.zeros((1, *img_shape))
    mask[:, :, i[indexes], j[indexes]] = 1.0

    return mask


def old_sequency_mask(img_shape, m):

    _, H, W = img_shape
    n = H * W

    indexes = get_permutation_list(n)[:m]
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    i = i.flatten(order="F")
    j = j.flatten(order="F")

    mask = torch.zeros((1, *img_shape))
    mask[:, :, i[indexes], j[indexes]] = 1.0

    return mask


def cake_cutting_seq(i, p):
    """Sequence of i-th"""
    step = -i * (-1) ** (np.mod(i, 2))

    seq = None
    # if i is odd
    if np.mod(i, 2) == 1:
        seq = list(range(i, i * p + 1, step))
    else:
        seq = list(range(i * p, i - 1, step))

    return seq


def cake_cutting_order(n):
    """Cake cutting order"""
    p = int(np.sqrt(n))
    seq = [cake_cutting_seq(i, p) for i in range(1, p + 1)]
    seq = [item for sublist in seq for item in sublist]
    return np.argsort(seq)


def cake_cutting_mask(img_shape, m):

    _, H, W = img_shape

    if H != W:
        warnings.warn("Image height and width must be equal for cake cutting mask.")

    n = H * W

    indexes = sequency_order(n)
    cake_order = cake_cutting_order(n)
    indexes = indexes[cake_order][:m]
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    i = i.flatten(order="F")
    j = j.flatten(order="F")

    mask = torch.zeros((1, *img_shape))
    mask[:, :, i[indexes], j[indexes]] = 1.0

    return mask


def diagonal_index_matrix(H: int, W: int) -> torch.Tensor:
    # Grid of row (I) and column (J) indices
    I, J = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    S = I + J
    flat_I = I.flatten()
    flat_S = S.flatten()
    total = H * W

    # Sort keys: primary S ascending, secondary I descending
    order = torch.argsort(flat_S * total - flat_I, stable=True)

    # Place 0..total-1 according to that order
    flat_A = torch.empty(total, dtype=torch.long, device=order.device)
    flat_A[order] = torch.arange(total, dtype=torch.long, device=order.device)

    return flat_A.view(H, W)


def zig_zag_mask(img_shape, m):

    _, H, W = img_shape
    mask = diagonal_index_matrix(H, W)
    mask = mask < m
    mask = mask.float()
    mask = mask.unsqueeze(0).repeat(1, img_shape[0], 1, 1)
    mask = hadamard_2d_ishift(mask)

    return mask


def xy_mask(img_shape, m):

    _, H, W = img_shape

    X, Y = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    index_matrix = X * Y + (X**2 + Y**2) / 4
    index_matrix /= index_matrix.max()

    indx = torch.argsort(index_matrix.view(-1))

    mask = torch.arange(1, H * W + 1)
    mask[indx] = mask.clone()
    mask = mask.view(H, W) <= m
    mask = mask.float()

    mask = mask.unsqueeze(0).repeat(1, img_shape[0], 1, 1)
    mask = hadamard_2d_ishift(mask)

    return mask


class SinglePixelCamera(DecomposablePhysics):
    r"""
    Single pixel imaging camera.

    Linear imaging operator with binary entries.

    If ``fast=True``, the operator uses a 2D subsampled Hadamard transform, which keeps the first :math:`m` modes
    according to the ``ordering`` parameter, set by default to `sequency ordering <https://en.wikipedia.org/wiki/Walsh_matrix#Sequency_ordering>`_.
    In this case, the images should have a size which is a power of 2.

    If ``fast=False``, the operator is a random iid binary matrix with equal probability of :math:`1/\sqrt{m}` or
    :math:`-1/\sqrt{m}`.

    Both options allow for an efficient singular value decomposition (see :class:`deepinv.physics.DecomposablePhysics`)
    The operator is always applied independently across channels.

    It is recommended to use ``fast=True`` for image sizes bigger than 32 x 32, since the forward computation with
    ``fast=False`` has an :math:`O(mn)` complexity, whereas with ``fast=True`` it has an :math:`O(n \log n)` complexity.

    An existing operator can be loaded from a saved ``.pth`` file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    .. warning::

        Since version 0.3.1, a small bug in the sequency ordering has been fixed. However, it is possible to use the old sequency ordering by setting `ordering='old_sequency'`.

    :param int m: number of single pixel measurements per acquisition (m).
    :param tuple img_shape: shape (C, H, W) of images.
    :param bool fast: The operator is iid binary if false, otherwise A is a 2D subsampled hadamard transform.
    :param str ordering: The ordering of selecting the first m measurements, available options are: `'sequency'`, `'cake_cutting'`, `'zig_zag'`, `'xy'`, `'old_sequency'`.
    :param torch.Generator rng: (optional) a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        SinglePixelCamera operators with 16 binary patterns for 32x32 image:

        >>> from deepinv.physics import SinglePixelCamera
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 32, 32)) # Define random 32x32 image
        >>> physics = SinglePixelCamera(m=16, img_shape=(1, 32, 32), fast=True)
        >>> torch.sum(physics.mask).item() # Number of measurements
        16.0
        >>> torch.round(physics(x)[:, :, :3, :3]).abs() # Compute measurements
        tensor([[[[1., 0., 1.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])

    """

    def __init__(
        self,
        m,
        img_shape,
        fast=True,
        ordering="sequency",
        device="cpu",
        dtype=torch.float32,
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = f"spcamera_m{m}"
        self.img_shape = img_shape
        self.fast = fast
        self.device = device
        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {self.device}."
            self.rng = rng
        self.initial_random_state = self.rng.get_state()

        if self.fast:

            _, H, W = img_shape

            assert H == 1 << int(np.log2(H)), "image height must be a power of 2"
            assert W == 1 << int(np.log2(W)), "image width must be a power of 2"

            if ordering == "sequency":
                mask = sequency_mask(img_shape, m)
            elif ordering == "cake_cutting":
                mask = cake_cutting_mask(img_shape, m)
            elif ordering == "zig_zag":
                mask = zig_zag_mask(img_shape, m)
            elif ordering == "xy":
                mask = xy_mask(img_shape, m)
            elif ordering == "old_sequency":
                # Raise warning if the old sequency mask is used
                print(
                    "Warning: The old sequency mask is deprecated. Plase, use sequency mask instead."
                )
                mask = old_sequency_mask(img_shape, m)
            else:
                raise ValueError(
                    f"Unknown ordering {ordering}. Available options are: `sequency`, `cake_cutting`, `zig_zag`, `xy`."
                )

            mask = mask.to(device)

        else:
            n = int(np.prod(img_shape[1:]))
            A = torch.where(
                torch.randn((m, n), device=device, dtype=dtype, generator=self.rng)
                > 0.5,
                -1.0,
                1.0,
            )
            A /= np.sqrt(m)  # normalize
            u, mask, vh = torch.linalg.svd(A, full_matrices=False)

            mask = mask.to(device).unsqueeze(0).type(dtype)
            self.vh = vh.to(device).type(dtype)
            self.u = u.to(device).type(dtype)

            self.u = torch.nn.Parameter(self.u, requires_grad=False)
            self.vh = torch.nn.Parameter(self.vh, requires_grad=False)

        self.mask = torch.nn.Parameter(mask, requires_grad=False)

    def V_adjoint(self, x):
        if self.fast:
            y = hadamard_2d(x)
        else:
            N, C = x.shape[0], self.img_shape[0]
            x = x.reshape(N, C, -1)
            y = torch.einsum("ijk, mk->ijm", x, self.vh)
        return y

    def V(self, y):
        if self.fast:
            x = hadamard_2d(y)
        else:
            N = y.shape[0]
            C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]
            x = torch.einsum("ijk, km->ijm", y, self.vh)
            x = x.reshape(N, C, H, W)
        return x

    def U_adjoint(self, x):
        if self.fast:
            out = x
        else:
            out = torch.einsum("ijk, km->ijm", x, self.u)
        return out

    def U(self, x):
        if self.fast:
            out = x
        else:
            out = torch.einsum("ijk, mk->ijm", x, self.u)
        return out


def gray_code(n):
    """
    Generate a Gray code sequence for n elements.

    Gray code is a binary numeral system where two successive values differ in only one bit.

    :param int n: Number of elements in the Gray code sequence.
    :return: A 2D array where each row represents a Gray code in binary form.
    :rtype: np.ndarray
    """
    g0 = np.array([[0], [1]])
    g = g0

    while g.shape[0] < n:
        g = np.hstack(
            [np.kron(g0, np.ones((g.shape[0], 1))), np.vstack([g, g[::-1, :]])]
        )
    return g


def gray_decode(n):
    """
    Decode a Gray code into its binary equivalent.

    This function converts a given Gray code integer into its corresponding binary number.

    :param int n: The Gray code to decode.
    :return: The decoded binary number.
    :rtype: int
    """
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n


def reverse(n, numbits):
    return sum(1 << (numbits - 1 - i) for i in range(numbits) if n >> i & 1)


def get_permutation_list(n):
    rev = np.zeros((n), dtype=int)
    for l in range(n):
        rev[l] = reverse(l, np.log2(n).astype(int))

    rev2 = np.zeros_like(rev)
    for l in range(n):
        rev2[l] = rev[gray_decode(l)]

    return rev2


def sequency_order(n):
    G = gray_code(n)
    G = G[:, ::-1]
    G = np.dot(G, 2 ** np.arange(G.shape[1] - 1, -1, -1)).astype(np.int32)
    return G


# test code
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import deepinv as dinv
#     import torchvision
#
#     device = "cuda:0"
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(device) / 255
#     x = torchvision.transforms.Resize((16, 8))(x)
#
#     m = 20
#     physics = SinglePixelCamera(m, (3, 16, 8), fast=False, device=device)
#
#     y = physics(x)
#
#     xhat = physics.A_adjoint(y)
#
#     dinv.utils.plot([x, xhat])
#
#     print(physics.adjointness_test(x))
#     print(physics.compute_norm(x))
#     # mi = min(int(np.sqrt(m)), x.shape[-2])
#     # mj = min(m - mi, x.shape[-2])
#     #
#     # revi = get_permutation_list(x.shape[-2])[:mi]
#     # revj = get_permutation_list(x.shape[-1])[:mj]
#     #
#     # mask = torch.zeros_like(x)
#     # for i in range(len(revi)):
#     #     for j in range(len(revj)):
#     #         mask[0, :, revi[i], revj[j]] = 1
#     #
#     # # generate low pass hadamard mask
#     # f = hadamard_2d(x)
#     # f = f * mask
#     # out = hadamard_2d(f)
#     #
#     # dinv.utils.plot_batch([x, out, f])
#     #
#     # rev = get_permutation_list(8)
#     # imgs = []
#     # for i in range(8):
#     #     y = torch.zeros((8, 1, 8, 8), device=dinv.device)
#     #     for j in range(8):
#     #         x = torch.zeros((8, 8), device=dinv.device)
#     #         x[rev[i], rev[j]] = 1
#     #         x = hadamard_2d(x)
#     #         y[j, 0, :, :] = x
#     #
#     #     imgs.append(y)
#     #
#     # dinv.utils.plot_batch(imgs)
