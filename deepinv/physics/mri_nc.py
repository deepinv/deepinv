from deepinv.physics.forward import LinearPhysics

MRINUFFT_AVAILABLE = True
try:
    from mrinufft import get_operator
except ImportError:
    MRINUFFT_AVAILABLE = False

class MRI_NC(LinearPhysics):
    """MRI Non-Cartesian Multicoil operator.

    """

    def __init__(self, kspace_trajectory, shape, n_coils, smaps,density=True, backend="cufinufft",**kwargs):
        super().__init__(**kwargs)
        if MRINUFFT_AVAILABLE is False:
            raise RuntimeError("mri-nufft is not installed.")

        self.backend = backend

        opKlass = get_operator(backend)
        self._operator = opKlass(kspace_trajectory, shape, density=density, n_coils=n_coils, smaps=smaps)

        self.A = self._operator.op
        self.A_adjoint = self._operator.adj_op
