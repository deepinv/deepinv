import numpy as np
import torch
from deepinv.physics.forward import LinearPhysics

try:
    from mrinufft import get_operator
    MRINUFFT_AVAILABLE = True
except ImportError:
    MRINUFFT_AVAILABLE = False


class MRI_NC(LinearPhysics):
    r"""MRI Non-Cartesian Multicoil operator.

    Implementation of the MRI Non-Cartesian Multicoil operator relying on the non-uniform fft implementation from
    mrinufft (https://github.com/mind-inria/mri-nufft).

    :param kspace_trajectory: the k-space trajectory;
    :param shape: image shape;
    :param n_coils: number of coils to be used;
    :param smaps: sensitivity maps;
    :param density: density compensation;
    :param backend: which backend to use. Could be either cufinufft (nufft on CPU) or ...
    """

    def __init__(self, kspace_trajectory, shape, n_coils, smaps=None, density=True, backend="cufinufft", **kwargs):
        super().__init__(**kwargs)
        if MRINUFFT_AVAILABLE is False:
            raise RuntimeError("mri-nufft is not installed.")

        self.backend = backend

        opKlass = get_operator(backend)
        self._operator = opKlass(kspace_trajectory, shape, density=density, n_coils=n_coils, smaps=smaps, keep_dims=True)

    def A(self, x):
        r"""
        Forward operator.
        """
        x_np = np.complex64(x.cpu().numpy())
        y_np = self._operator.op(x_np)
        return torch.from_numpy(y_np).type(x.type())

    def A_adjoint(self, y):
        r"""
        Adjoint operator.
        """
        y_np = np.complex64(y.squeeze().cpu().numpy())
        x_np = self._operator.adj_op(y_np)
        return torch.from_numpy(x_np).type(y.type())
