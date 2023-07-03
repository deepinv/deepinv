import torch
from deepinv.physics.forward import LinearPhysics

MRINUFFT_AVAILABLE = True
try:
    from mrinufft import get_operator
except ImportError:
    MRINUFFT_AVAILABLE = False

class MRI_NC(LinearPhysics):
    """MRI Non-Cartesian Multicoil operator.

    """

    def __init__(self, kspace_trajectory, shape, n_coils, smaps=None, density=True, backend="cufinufft", **kwargs):
        super().__init__(**kwargs)
        if MRINUFFT_AVAILABLE is False:
            raise RuntimeError("mri-nufft is not installed.")

        self.backend = backend

        opKlass = get_operator(backend)
        self._operator = opKlass(kspace_trajectory, shape, density=density, n_coils=n_coils, smaps=smaps)

        # self.A = self._operator.op
        # self.A_adjoint = self._operator.adj_op

    def A(self, x):
        x_np = x.cpu().numpy()
        y_np = self._operator.op(x_np)
        return torch.from_numpy(y_np).to(x.type())

    def A_adjoint(self, y):
        y_np = y.cpu().numpy()
        x_np = self._operator.op(y_np)
        return torch.from_numpy(x_np).to(y.type())
