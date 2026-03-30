from deepinv.physics.forward import LinearPhysics
from deepinv.transform.base import Transform


# A virtual operator is an operator of the form A' = A T where A is a linear
# operator and T is an invertible linear operator. The operator T is to be
# thought of as a specific transform, e.g., a rotation with a specific angle,
# or a shift with a specific displacement.
# Unlike general composition of linear operators, the invertibility of T allows
# to compute the pseudo-inverse of A' in a computationally efficient closed
# form, i.e., A'^\dagger = T^{-1} A^\dagger.
class VirtualLinearPhysics(LinearPhysics):
    def __init__(self, *, physics: LinearPhysics, transform: Transform, g_params: dict):
        super().__init__()
        self.physics = physics
        self.T = lambda x: transform.transform(x, **g_params)
        self.T_inv = lambda x: transform.inverse(x, **g_params)

    def A(self, x, **kwargs):
        Tx = self.T(x)  # T
        return self.physics.A(Tx)  # A

    def A_adjoint(self, y, **kwargs):
        x = self.physics.A_adjoint(y)  # A^*
        return self.T_inv(x)  # T^{-1}

    def A_dagger(self, y, **kwargs):
        x = self.physics.A_dagger(y)  # A^\dagger
        return self.T_inv(x)  # T^{-1}
