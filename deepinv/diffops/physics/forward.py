import torch
import numpy as np

class GaussianNoise(torch.nn.Module): # parent class for forward models
    def __init__(self, std=.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x)*self.std


class Forward(torch.nn.Module):  # parent class for forward models
    def __init__(self, A=lambda x: x, A_adjoint=lambda x: x,
                 noise_model=lambda x: x, sensor_model=lambda x: x,
                 max_iter=50, tol=1e-3):
        super().__init__()
        self.noise_model = noise_model
        self.sensor_model = sensor_model
        self.forw = A
        self.adjoint = A_adjoint
        self.max_iter = max_iter
        self.tol = tol

    def __add__(self, other): #  physics3 = physics1 + physics2
        A = lambda x: self.A(other.A(x)) # (A' = A_1 A_2)
        A_adjoint = lambda x: other.A_adjoint(self.A_adjoint(x)) #(A'^{T} = A_2^{T} A_1^{T})
        noise = self.noise_model
        sensor = self.sensor_model
        return Forward(A, A_adjoint, noise, sensor)

    def forward(self, x):# degrades signal
        return self.sensor(self.noise(self.A(x)))

    def A(self, x):
        return self.forw(x)

    def sensor(self, x):
        return self.sensor_model(x)

    def noise(self, x):
        return self.noise_model(x)

    def A_adjoint(self, x):
        return self.adjoint(x)

    def A_dagger(self, y):
        """
        Computes A^{\dagger}y = x using conjugate gradient method.
        see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

        If the size of y is larger than x (overcomplete problem), it computes (A^t A)^{-1} A^t y
        otherwise (incomplete problem) it computes  A^t (A A^t)^{-1} y

        This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist

        @:param y : The right hand side (RHS) vector of the system.
        """

        def dot(s1, s2):
            return (s1*s2).flatten().sum()

        yr = self.A_adjoint(y)
        incomplete = np.prod(yr.shape) > np.prod(y.shape)
        x = torch.zeros_like(y)
        if not incomplete:
            y = yr

        r = y
        p = r
        rsold = dot(r, r)

        for i in range(self.max_iter):
            if incomplete:
                Ap = self.A(self.A_adjoint(p))
            else:
                Ap = self.A_adjoint(self.A(p))

            alpha = rsold / dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = dot(r, r)
            #print(rsnew.sqrt())
            if rsnew.sqrt() < self.tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        if incomplete:
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
                print("Power iteration converged at iteration: ", it, ", val: ", z)
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

        s1 = v.flatten().T @ Au.flatten()
        s2 = Atv.flatten().T @ u_in.flatten()

        return s1-s2

class Denoising(Forward):
    def __init__(self, sigma=.1):
        super().__init__()
        self.name = 'denoising'
        self.noise_model = GaussianNoise(sigma)

    def A(self, x):
        return x

    def A_dagger(self, x):
        return x

    def A_adjoint(self, x):
        return x