import torch
import numpy as np
from deepinv.optim.utils import conjugate_gradient


class Physics(torch.nn.Module):  # parent class for forward models
    def __init__(self, A=lambda x: x, A_adjoint=lambda x: x,
                 noise_model=lambda x: x, sensor_model=lambda x: x,
                 max_iter=50, tol=1e-3):
        '''
        Parent function for forward operators
        TODO
        :param A: linear
        :param A_adjoint:
        :param noise_model:
        :param sensor_model:
        :param max_iter:
        :param tol:
        '''
        super().__init__()
        self.noise_model = noise_model
        self.sensor_model = sensor_model
        self.forw = A
        self.SVD = False  # flag indicating SVD available
        self.adjoint = A_adjoint
        self.max_iter = max_iter
        self.tol = tol

    def __add__(self, other): #  physics3 = physics1 + physics2
        A = lambda x: self.A(other.A(x)) # (A' = A_1 A_2)
        A_adjoint = lambda x: other.A_adjoint(self.A_adjoint(x)) #(A'^{T} = A_2^{T} A_1^{T})
        noise = self.noise_model
        sensor = self.sensor_model
        return Physics(A, A_adjoint, noise, sensor)

    def forward(self, x):  # degrades signal
        return self.sensor(self.noise(self.A(x)))

    def A(self, x):
        return self.forw(x)

    def sensor(self, x):
        return self.sensor_model(x)

    def noise(self, x):
        return self.noise_model(x)

    def A_adjoint(self, x):
        return self.adjoint(x)

    def prox_l2(self, z, y, gamma):
        '''
        Computes proximal operator of f(x) = 1/2*||Ax-y||^2
        i.e. argmin_x 1/2*||Ax-y||^2+gamma/2*||x-z||^2

        :param y: measurements tensor
        :param z: signal tensor
        :param gamma: hyperparameter of the proximal operator
        :return: estimated signal tensor
        '''
        b = self.A_adjoint(y) + gamma*z
        H = lambda x: self.A_adjoint(self.A(x))+gamma*x
        x = conjugate_gradient(H, b, self.max_iter, self.tol)
        return x

    def prox_l2_norm(self, z, y, gamma):
        '''
        computes the proximal operator of \frac{1}{2}*gamma*||x-y||_2^2 in z

        :param y: measurements tensor
        :param z: signal tensor
        :param gamma: hyperparameter of the proximal operator
        :return: estimated signal tensor
        '''
        return (z + gamma * y) / (1 + gamma)

    def A_dagger(self, y):
        """
        Computes A^{\dagger}y = x using conjugate gradient method.

        If the size of y is larger than x (overcomplete problem), it computes (A^t A)^{-1} A^t y
        otherwise (incomplete problem) it computes  A^t (A A^t)^{-1} y

        This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist

        :param y : The right hand side (RHS) vector of the system.
        """


        Aty = self.A_adjoint(y)

        overcomplete = np.prod(Aty.shape) < np.prod(y.shape)

        if not overcomplete:
            A = lambda x: self.A(self.A_adjoint(x))
            b = y
        else:
            A = lambda x: self.A_adjoint(self.A(x))
            b = Aty

        x = conjugate_gradient(A=A, b=b, max_iter=self.max_iter, tol=self.tol)

        if not overcomplete:
            x = self.A_adjoint(x)

        return x

    def power_method(self, x0, max_iter=100, tol=1e-3, verbose=True):
        '''
        Computes the spectral (l2) norm (Lipschitz constant) of the operator At*A, i.e. ||At*A||.
        Args:
            x0: initialisation point of the algorithm
            A: forward operator A
            At: adjoint (backward) operator of A
            max_iter: maximum number of iterations
            tol: relative variation criterion for convergence
            verbose: print information

        Returns:
            z: spectral norm of At*A, i.e. z = ||At*A||
        '''
        x = torch.randn_like(x0)
        x /= torch.norm(x)
        zold = torch.zeros_like(x)
        for it in range(max_iter):
            y = self.A(x)
            y = self.A_adjoint(y)
            z = torch.matmul(x.reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2

            rel_var = torch.norm(z - zold)
            if rel_var < tol and verbose:
                print(f"Power iteration converged at iteration {it}, value={z.item():.2f}")
                break
            zold = z
            x = y / torch.norm(y)

        return z

    def adjointness_test(self, u):
        '''
        Numerically check that A_adj is indeed the adjoint of A.

        Args:
            u: initialisation point of the adjointness test method
        Returns:
            s1-s2: a quantity that should be theoretically 0. In practice, it should be of the order of the
            chosen dtype precision (i.e. single or double).
        '''
        u_in = u #.type(self.dtype)
        Au = self.A(u_in)

        v = torch.randn_like(Au)
        Atv = self.A_adjoint(v)

        s1 = (v*Au).flatten().sum()
        s2 = (Atv*u_in).flatten().sum()

        return s1-s2


class DecomposablePhysics(Physics):
    def __init__(self, U=lambda x: x, U_adjoint=lambda x: x, V=lambda x: x, V_adjoint=lambda x: x, mask = 1., **kwargs):
        r'''
        Parent function for linear forward operators which have a simple singular value decomposition (SVD) of the form

        .. math::
            A = U\diag(s)V^{T} \in \mathbb{R}^{m\times n}

        where :math:`U\in\mathbb{C}^{n\times n}` and :math:`V\in\mathbb{C}^{m\times m}`
        are orthonormal linear transformations and `U\in\mathbb{C}^{n\times n}`

        :param U: (callable) orthonormal transformation
        :param U_adjoint: (callable) transpose of U
        :param V: (callable) orthonormal transformation
        :param V_adjoint: (callable) transpose of V
        :param mask: (Torch.Tensor or float) Singular values of the transform
        '''
        super().__init__(**kwargs)
        self._V = V
        self._U = U
        self._U_adjoint = U_adjoint
        self._V_adjoint = V_adjoint
        self.mask = mask

    def A(self, x):
        return self.U(self.mask*self.V_adjoint(x))

    def U(self, x):
        return self._U(x)

    def V(self, x):
        return self._U(x)

    def U_adjoint(self, x):
        return self._U_adjoint(x)

    def V_adjoint(self, x):
        return self._V_adjoint(x)

    def A_adjoint(self, y):

        if isinstance(self.mask, float):
            mask = self.mask
        else:
            mask = torch.conj(self.mask)

        return self.V(mask*self.V_adjoint(y))

    def prox_l2(self, z, y, gamma):
        '''
        Computes proximal operator of f(x) = 1/2*||Ax-y||^2
        i.e. argmin_x 1/2*||Ax-y||^2+gamma/2*||x-z||^2

        :param y: (Torch.Tensor) measurements tensor
        :param z: (Torch.Tensor or float) signal tensor
        :param gamma: (positive float) hyperparameter of the proximal operator
        :return: estimated signal tensor
        '''
        b = self.A_adjoint(y) + gamma*z

        # scaling = self.mask.pow(2) + gamma
        if not isinstance(self.mask, float):
            scaling = self.mask.pow(2) + gamma
        else:
            scaling = self.mask ** 2 + gamma

        x = self.V(self.V_adjoint(b)/scaling)

        return x

    def A_dagger(self, y):
        """
        Computes
        .. math::
            A^{\dagger}y = x

        This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist

        :param y : The right hand side (RHS) vector of the system.
        """

        # avoid division by singular value = 0

        mask = self.mask

        if not isinstance(self.mask, float):
            mask[mask == 0] = 1

        return self.V(self.U_adjoint(y)/mask)
