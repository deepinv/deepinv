from deepinv.physics.forward import LinearPhysics
from typing import Callable

# A virtual operator is an operator of the form A' = A T where A is a linear
# operator and T is an invertible linear operator. The operator T is to be
# thought of as a specific transform, e.g., a rotation with a specific angle,
# or a shift with a specific displacement.
# Unlike general composition of linear operators, the invertibility of T allows
# to compute the pseudo-inverse of A' in a computationally efficient closed
# form, i.e., A'^\dagger = T^{-1} A^\dagger.
class VirtualOperator(LinearPhysics):
    def __init__(self, *, physics: LinearPhysics, T: Callable, T_inv: Callable):
        super().__init__()
        self.physics = physics
        self.T = T
        self.T_inv = T_inv

    def A(self, x, **kwargs):
        Tx = self.T(x)  # T
        return self.physics.A(Tx, **kwargs)  # A

    def A_adjoint(self, y, **kwargs):
        x = self.physics.A_adjoint(y, **kwargs)  # A^*
        return self.T_inv(x)  # T^{-1}

    def A_dagger(self, y, **kwargs):
        x = self.physics.A_dagger(y, **kwargs)  # A^\dagger
        return self.T_inv(x)  # T^{-1}
