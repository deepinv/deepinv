from deepinv.physics.forward import DecomposablePhysics
import torch
import numpy as np


def hadamard_1d(u, normalize=True):
    """
    Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.

    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
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
    Computes 2 dimensional Hadamard transform using 1 dimensional transform.
    """
    out = hadamard_1d(hadamard_1d(x).transpose(-1, -2)).transpose(-1, -2)
    return out


class SinglePixelCamera(DecomposablePhysics):
    r"""
    Single pixel imaging camera.

    Linear imaging operator with binary entries.

    If ``fast=False``, the operator uses a 2D subsampled hadamard transform, which keeps the first :math:`m` modes
    according to the `sequency ordering <https://en.wikipedia.org/wiki/Walsh_matrix#Sequency_ordering>`_.
    In this case, the images should have a size which is a power of 2.

    If ``fast=False``, the operator is a random iid binary matrix with equal probability of :math:`1/\sqrt{m}` or
    :math:`-1/\sqrt{m}`.

    Both options allow for an efficient singular value decomposition (see :class:`deepinv.physics.DecomposablePhysics`)
    The operator is always applied independently across channels.

    It is recommended to use ``fast=True`` for image sizes bigger than 32 x 32, since the forward computation with
    ``fast=False`` has an :math:`O(mn)` complexity, whereas with ``fast=True`` it has an :math:`O(n \log n)` complexity.

    An existing operator can be loaded from a saved ``.pth`` file via ``self.load_state_dict(save_path)``,
    in a similar fashion to :class:`torch.nn.Module`.

    :param int m: number of single pixel measurements per acquisition (m).
    :param tuple img_shape: shape (C, H, W) of images.
    :param bool fast: The operator is iid binary if false, otherwise A is a 2D subsampled hadamard transform.
    :param str device: Device to store the forward matrix.
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
        48.0
        >>> torch.round(physics(x)[:, :, :3, :3]).abs() # Compute measurements
        tensor([[[[1., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]]]])

    """

    def __init__(
        self,
        m,
        img_shape,
        fast=True,
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
            C, H, W = img_shape
            mi = min(int(np.sqrt(m)), H)
            mj = min(m - mi, W)

            revi = get_permutation_list(H)[:mi]
            revj = get_permutation_list(W)[:mj]

            assert H == 1 << int(np.log2(H)), "image height must be a power of 2"
            assert W == 1 << int(np.log2(W)), "image width must be a power of 2"

            mask = torch.zeros(img_shape).unsqueeze(0)
            for i in range(len(revi)):
                for j in range(len(revj)):
                    mask[0, :, revi[i], revj[j]] = 1

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


def gray_decode(n):
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
