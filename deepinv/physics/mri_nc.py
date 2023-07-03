from deepinv.physics.forward import LinearPhysics

from mrinufft import get_operator

class MRI_NC(LinearPhysics):
    """MRI Non-Cartesian Multicoil operator.

    """

    def __init__(self, kspace_trajectory, shape, n_coils, smaps,density=True, backend="cufinufft",**kwargs):
        super().__init__(**kwargs)

        self.backend = backend

        opKlass = get_operator(backend)
        self._operator = opKlass(kspace_trajectory, shape, density=density, n_coils=n_coils, smaps=smaps)

        self.A = self._operator.op
        self.A_adjoint = self._operator.adj_op
