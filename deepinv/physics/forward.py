import torch
from deepinv.optim.utils import conjugate_gradient
from .noise import GaussianNoise
from deepinv.utils import randn_like, TensorList


class Physics(torch.nn.Module):  # parent class for forward models
    r"""
    Parent class for forward operators

    It describes the general forward measurement process

    .. math::

        y = N(A(x))

    where :math:`x` is an image of :math:`n` pixels, :math:`y` is the measurements of size :math:`m`,
    :math:`A:\xset\mapsto \yset` is a deterministic mapping capturing the physics of the acquisition
    and :math:`N:\yset\mapsto \yset` is a stochastic mapping which characterizes the noise affecting
    the measurements.

    :param callable A: forward operator function which maps an image to the observed measurements :math:`x\mapsto y`.
    :param callable noise_model: function that adds noise to the measurements :math:`N(z)`.
        See the noise module for some predefined functions.
    :param callable sensor_model: function that incorporates any sensor non-linearities to the sensing process,
        such as quantization or saturation, defined as a function :math:`\eta(z)`, such that
        :math:`y=\eta\left(N(A(x))\right)`. By default, the sensor_model is set to the identity :math:`\eta(z)=z`.
    :param int max_iter: If the operator does not have a closed form pseudoinverse, the gradient descent algorithm
        is used for computing it, and this parameter fixes the maximum number of gradient descent iterations.
    :param float tol: If the operator does not have a closed form pseudoinverse, the gradient descent algorithm
        is used for computing it, and this parameter fixes the absolute tolerance of the gradient descent algorithm.

    """

    def __init__(
        self,
        A=lambda x: x,
        noise_model=lambda x: x,
        sensor_model=lambda x: x,
        max_iter=50,
        tol=1e-3,
    ):
        super().__init__()
        self.noise_model = noise_model
        self.sensor_model = sensor_model
        self.forw = A
        self.SVD = False  # flag indicating SVD available
        self.max_iter = max_iter
        self.tol = tol

    def __mul__(self, other):  #  physics3 = physics1 \circ physics2
        r"""
        Concatenates two forward operators :math:`A = A_1\circ A_2` via the mul operation

        The resulting operator keeps the noise and sensor models of :math:`A_1`.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (deepinv.physics.Physics) concantenated operator

        """
        A = lambda x: self.A(other.A(x))  # (A' = A_1 A_2)
        noise = self.noise_model
        sensor = self.sensor_model
        return Physics(
            A=A,
            noise_model=noise,
            sensor_model=sensor,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def __add__(self, other):
        r"""
        Stacks two linear forward operators :math:`A(x) = \begin{bmatrix} A_1(x) \\ A_2(x) \end{bmatrix}`
        via the add operation.

        The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
        each entry corresponds to the measurements of the corresponding operator.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (deepinv.physics.Physics) stacked operator

        """
        A = lambda x: TensorList(self.A(x)).append(TensorList(other.A(x)))

        class noise(torch.nn.Module):
            def __init__(self, noise1, noise2):
                super().__init__()
                self.noise1 = noise1
                self.noise2 = noise2

            def forward(self, x):
                return TensorList(self.noise1(x[:-1])).append(self.noise2(x[-1]))

        class sensor(torch.nn.Module):
            def __init__(self, sensor1, sensor2):
                super().__init__()
                self.sensor1 = sensor1
                self.sensor2 = sensor2

            def forward(self, x):
                return TensorList(self.sensor1(x[:-1])).append(self.sensor2(x[-1]))

        return Physics(
            A=A,
            noise_model=noise(self.noise_model, other.noise_model),
            sensor_model=sensor(self.sensor_model, other.sensor_model),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def reset(self, **kwargs):
        if isinstance(self.noise_model, torch.nn.Module):
            self.noise_model.__init__(**kwargs)

    def forward(self, x):
        r"""
        Computes forward operator :math:`y = N(A(x))` (with noise and/or sensor non-linearities)

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) noisy measurements

        """
        return self.sensor(self.noise(self.A(x)))

    def A(self, x):
        r"""
        Computes forward operator :math:`y = A(x)` (without noise and/or sensor non-linearities)

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) clean measurements

        """
        return self.forw(x)

    def sensor(self, x):
        r"""
        Computes sensor non-linearities :math:`y = \eta(y)`

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) clean measurements
        """
        return self.sensor_model(x)

    def noise(self, x):
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :return torch.Tensor: noisy measurements

        """
        return self.noise_model(x)

    def A_dagger(self, y, x_init=None):
        r"""
        Computes an inverse of :math:`y = Ax` via gradient descent.

        This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :param torch.Tensor x_init: initial guess for the reconstruction.
        :return: (torch.Tensor) The reconstructed image :math:`x`.

        """

        if x_init is None:
            x_init = self.A_adjoint(y)

        x = torch.nn.Parameter(x_init, requires_grad=True)

        optimizer = torch.optim.SGD([x], lr=1e-1)
        loss = torch.nn.MSELoss()
        for i in range(self.max_iter):
            err = loss(self.A(x), y)
            optimizer.zero_grad()
            err.backward(retain_graph=True)
            optimizer.step()
            if err < self.tol:
                break

        return x.clone()


class LinearPhysics(Physics):
    r"""
    Parent class for linear operators.

    It describes the linear forward measurement process of the form

    .. math::

        y = N(A(x))

    where :math:`x` is an image of :math:`n` pixels, :math:`y` is the measurements of size :math:`m`,
    :math:`A:\xset\mapsto \yset` is a deterministic linear mapping capturing the physics of the acquisition
    and :math:`N:\yset\mapsto \yset` is a stochastic mapping which characterizes the noise affecting
    the measurements.

    :param callable A: forward operator function which maps an image to the observed measurements :math:`x\mapsto y`.
        It is recommended to normalize it to have unit norm.
    :param callable A_adjoint: transpose of the forward operator, which should verify the adjointness test.
    :param callable noise_model: function that adds noise to the measurements :math:`N(z)`.
        See the noise module for some predefined functions.
    :param callable sensor_model: function that incorporates any sensor non-linearities to the sensing process,
        such as quantization or saturation, defined as a function :math:`\eta(z)`, such that
        :math:`y=\eta\left(N(A(x))\right)`. By default, the sensor_model is set to the identity :math:`\eta(z)=z`.
    :param int max_iter: If the operator does not have a closed form pseudoinverse, the conjugate gradient algorithm
        is used for computing it, and this parameter fixes the maximum number of conjugate gradient iterations.
    :param float tol: If the operator does not have a closed form pseudoinverse, the conjugate gradient algorithm
        is used for computing it, and this parameter fixes the absolute tolerance of the conjugate gradient algorithm.

    """

    def __init__(
        self,
        A=lambda x: x,
        A_adjoint=lambda x: x,
        noise_model=lambda x: x,
        sensor_model=lambda x: x,
        max_iter=50,
        tol=1e-3,
        **kwargs,
    ):
        super().__init__(
            A=A,
            noise_model=noise_model,
            sensor_model=sensor_model,
            max_iter=max_iter,
            tol=tol,
        )

        self.adjoint = A_adjoint

    def A_adjoint(self, y):
        r"""
        Computes transpose of the forward operator :math:`\tilde{x} = A^{\top}y`.
        If :math:`A` is linear, it should be the exact transpose of the forward matrix.

        .. note:

            If problem is non-linear, there is not a well-defined transpose operation,
            but defining one can be useful for some reconstruction networks, such as ``deepinv.models.ArtifactRemoval``.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) linear reconstruction :math:`\tilde{x} = A^{\top}y`.

        """
        return self.adjoint(y)

    def __mul__(self, other):
        r"""
        Concatenates two linear forward operators :math:`A = A_1\circ A_2` via the * operation

        The resulting linear operator keeps the noise and sensor models of :math:`A_1`.

        :param deepinv.physics.LinearPhysics other: Physics operator :math:`A_2`
        :return: (deepinv.physics.LinearPhysics) concantenated operator

        """
        A = lambda x: self.A(other.A(x))  # (A' = A_1 A_2)
        A_adjoint = lambda x: other.A_adjoint(self.A_adjoint(x))
        noise = self.noise_model
        sensor = self.sensor_model
        return LinearPhysics(
            A=A,
            A_adjoint=A_adjoint,
            noise_model=noise,
            sensor_model=sensor,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def __add__(self, other):
        r"""
        Stacks two linear forward operators :math:`A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}` via the add operation.

        The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
        each entry corresponds to the measurements of the corresponding operator.

        :param deepinv.physics.LinearPhysics other: Physics operator :math:`A_2`
        :return: (deepinv.physics.LinearPhysics) stacked operator

        """
        A = lambda x: TensorList(self.A(x)).append(TensorList(other.A(x)))

        def A_adjoint(y):
            at1 = self.A_adjoint(y[:-1]) if len(y) > 2 else self.A_adjoint(y[0])
            return at1 + other.A_adjoint(y[-1])

        class noise(torch.nn.Module):
            def __init__(self, noise1, noise2):
                super().__init__()
                self.noise1 = noise1
                self.noise2 = noise2

            def forward(self, x):
                return TensorList(self.noise1(x[:-1])).append(self.noise2(x[-1]))

        class sensor(torch.nn.Module):
            def __init__(self, sensor1, sensor2):
                super().__init__()
                self.sensor1 = sensor1
                self.sensor2 = sensor2

            def forward(self, x):
                return TensorList(self.sensor1(x[:-1])).append(self.sensor2(x[-1]))

        return LinearPhysics(
            A=A,
            A_adjoint=A_adjoint,
            noise_model=noise(self.noise_model, other.noise_model),
            sensor_model=sensor(self.sensor_model, other.sensor_model),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def compute_norm(self, x0, max_iter=100, tol=1e-3, verbose=True):
        r"""
        Computes the spectral :math:`\ell_2` norm (Lipschitz constant) of the operator

        :math:`A^{\top}A`, i.e., :math:`\|A^{\top}A\|`.

        using the `power method <https://en.wikipedia.org/wiki/Power_iteration>`_.

        :param torch.Tensor x0: initialisation point of the algorithm
        :param int max_iter: maximum number of iterations
        :param float tol: relative variation criterion for convergence
        :param bool verbose: print information

        :returns z: (float) spectral norm of :math:`A^{\top}A`, i.e., :math:`\|A^{\top}A\|`.
        """
        x = torch.randn_like(x0)
        x /= torch.norm(x)
        zold = torch.zeros_like(x)
        for it in range(max_iter):
            y = self.A(x)
            y = self.A_adjoint(y)
            z = torch.matmul(x.reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2

            rel_var = torch.norm(z - zold)
            if rel_var < tol and verbose:
                print(
                    f"Power iteration converged at iteration {it}, value={z.item():.2f}"
                )
                break
            zold = z
            x = y / torch.norm(y)

        return z

    def adjointness_test(self, u):
        r"""
        Numerically check that :math:`A^{\top}` is indeed the adjoint of :math:`A`.

        :param torch.Tensor u: initialisation point of the adjointness test method

        :return: (float) a quantity that should be theoretically 0. In practice, it should be of the order of the chosen dtype precision (i.e. single or double).

        """
        u_in = u  # .type(self.dtype)
        Au = self.A(u_in)

        if isinstance(Au, tuple) or isinstance(Au, list):
            V = [randn_like(au) for au in Au]
            Atv = self.A_adjoint(V)
            s1 = 0
            for au, v in zip(Au, V):
                s1 += (v * au).flatten().sum()

        else:
            v = randn_like(Au)
            Atv = self.A_adjoint(v)

            s1 = (v * Au).flatten().sum()

        s2 = (Atv * u_in).flatten().sum()

        return s1 - s2

    def prox_l2(self, z, y, gamma):
        r"""
        Computes proximal operator of :math:`f(x) = \frac{1}{2}\|Ax-y\|^2`, i.e.,

        .. math::

            \underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2

        :param torch.Tensor y: measurements tensor
        :param torch.Tensor z: signal tensor
        :param float gamma: hyperparameter of the proximal operator
        :return: (torch.Tensor) estimated signal tensor

        """
        b = self.A_adjoint(y) + 1 / gamma * z
        H = lambda x: self.A_adjoint(self.A(x)) + 1 / gamma * x
        x = conjugate_gradient(H, b, self.max_iter, self.tol)
        return x

    def A_dagger(self, y):
        r"""
        Computes the solution in :math:`x` to :math:`y = Ax` using the
        ` conjugate gradient method <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.

        If the size of :math:`y` is larger than :math:`x` (overcomplete problem), it computes :math:`(A^{\top} A)^{-1} A^{\top} y`,
        otherwise (incomplete problem) it computes :math:`A^{\top} (A A^{\top})^{-1} y`.

        This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :return: (torch.Tensor) The reconstructed image :math:`x`.

        """
        Aty = self.A_adjoint(y)

        overcomplete = Aty.flatten().shape[0] < y.flatten().shape[0]

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


class DecomposablePhysics(LinearPhysics):
    r"""
    Parent class for linear operators with SVD decomposition.


    The singular value decomposition is expressed as

    .. math::

        A = U\text{diag}(s)V^{\top} \in \mathbb{R}^{m\times n}

    where :math:`U\in\mathbb{C}^{n\times n}` and :math:`V\in\mathbb{C}^{m\times m}`
    are orthonormal linear transformations and :math:`s\in\mathbb{R}_{+}^{n}` are the singular values.

    :param callable U: orthonormal transformation
    :param callable U_adjoint: transpose of U
    :param callable V: orthonormal transformation
    :param callable V_adjoint: transpose of V
    :param torch.Tensor, float mask: Singular values of the transform

    """

    def __init__(
        self,
        U=lambda x: x,
        U_adjoint=lambda x: x,
        V=lambda x: x,
        V_adjoint=lambda x: x,
        mask=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._V = V
        self._U = U
        self._U_adjoint = U_adjoint
        self._V_adjoint = V_adjoint
        self.mask = mask

    def A(self, x):
        return self.U(self.mask * self.V_adjoint(x))

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

        return self.V(mask * self.U_adjoint(y))

    def prox_l2(self, z, y, gamma):
        r"""
        Computes proximal operator of :math:`f(x)=\frac{\gamma}{2}\|Ax-y\|^2`
        in an efficient manner leveraging the singular vector decomposition.

        :param torch.Tensor y: measurements tensor
        :param torch.Tensor, float z: signal tensor
        :param float gamma: hyperparameter :math:`\gamma` of the proximal operator
        :return: (torch.Tensor) estimated signal tensor

        """
        b = self.A_adjoint(y) + 1 / gamma * z
        if isinstance(self.mask, float):
            scaling = self.mask**2 + 1 / gamma
        else:
            scaling = torch.conj(self.mask) * self.mask + 1 / gamma
        x = self.V(self.V_adjoint(b) / scaling)
        return x

    def A_dagger(self, y):
        r"""
        Computes :math:`A^{\dagger}y = x` in an efficient manner leveraging the singular vector decomposition.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :return: (torch.Tensor) The reconstructed image :math:`x`.

        """

        # avoid division by singular value = 0

        if not isinstance(self.mask, float):
            mask = torch.zeros_like(self.mask)
            mask[self.mask > 1e-5] = 1 / self.mask[self.mask > 1e-5]
        else:
            mask = 1 / self.mask

        return self.V(self.U_adjoint(y) * mask)


class Denoising(DecomposablePhysics):
    r"""

    Forward operator for denoising problems.

    The linear operator is just the identity mapping :math:`A(x)=x`

    :param torch.nn.Module noise: noise distribution, e.g., ``deepinv.physics.GaussianNoise``, or a user-defined torch.nn.Module.
    """

    def __init__(self, noise=GaussianNoise(sigma=0.1), **kwargs):
        super().__init__(**kwargs)
        if noise is None:
            noise = GaussianNoise(sigma=0.0)
        self.noise_model = noise
