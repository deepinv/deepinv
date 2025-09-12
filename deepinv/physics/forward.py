from __future__ import annotations
from typing import Union, Optional, Callable, List, Tuple
import warnings
import copy
import inspect
import collections.abc

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn
from deepinv.physics.noise import NoiseModel, GaussianNoise, ZeroNoise
from deepinv.utils.tensorlist import randn_like, TensorList
from deepinv.optim.utils import least_squares, lsqr
from deepinv.utils.compat import zip_strict


class Physics(torch.nn.Module):  # parent class for forward models
    r"""
    Parent class for forward operators

    It describes the general forward measurement process

    .. math::

        y = \noise{\forw{x}}

    where :math:`x` is an image of :math:`n` pixels, :math:`y` is the measurements of size :math:`m`,
    :math:`A:\xset\mapsto \yset` is a deterministic mapping capturing the physics of the acquisition
    and :math:`N:\yset\mapsto \yset` is a stochastic mapping which characterizes the noise affecting
    the measurements.

    :param Callable A: forward operator function which maps an image to the observed measurements :math:`x\mapsto y`.
    :param deepinv.physics.NoiseModel, Callable noise_model: function that adds noise to the measurements :math:`\noise{z}`.
        See the noise module for some predefined functions.
    :param Callable sensor_model: function that incorporates any sensor non-linearities to the sensing process,
        such as quantization or saturation, defined as a function :math:`\sensor{z}`, such that
        :math:`y=\sensor{\noise{\forw{x}}}`. By default, the `sensor_model` is set to the identity :math:`\sensor{z}=z`.
    :param int max_iter: If the operator does not have a closed form pseudoinverse, the gradient descent algorithm
        is used for computing it, and this parameter fixes the maximum number of gradient descent iterations.
    :param float tol: If the operator does not have a closed form pseudoinverse, the gradient descent algorithm
        is used for computing it, and this parameter fixes the absolute tolerance of the gradient descent algorithm.
    :param str solver: least squares solver to use. Only gradient descent is available for non-linear operators.
    """

    def __init__(
        self,
        A: Callable = lambda x, **kwargs: x,
        noise_model: Optional[NoiseModel] = None,
        sensor_model: Callable = lambda x: x,
        solver: str = "gradient_descent",
        max_iter: int = 50,
        tol: float = 1e-4,
        **kwargs,
    ):
        if noise_model is None:
            noise_model = ZeroNoise()
        super().__init__()
        self.noise_model = noise_model
        self.sensor_model = sensor_model
        self.forw = A
        self.SVD = False  # flag indicating SVD available
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

        if len(kwargs) > 0:
            warnings.warn(
                f"Arguments {kwargs} are passed to {self.__class__.__name__} but are ignored."
            )

    def __mul__(self, other):
        r"""
        Concatenates two forward operators :math:`A = A_1\circ A_2` via the mul operation

        The resulting operator keeps the noise and sensor models of :math:`A_1`.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.Physics`) concatenated operator

        """

        warnings.warn(
            "You are composing two physics objects. The resulting physics will not retain the original attributes. "
            "You may instead retrieve attributes of the original physics by indexing the resulting physics."
        )
        return compose(other, self, max_iter=self.max_iter, tol=self.tol)

    def stack(self, other):
        r"""
        Stacks two forward operators :math:`A(x) = \begin{bmatrix} A_1(x) \\ A_2(x) \end{bmatrix}`

        The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
        each entry corresponds to the measurements of the corresponding operator.

        Returns a :class:`deepinv.physics.StackedPhysics` object.

        See :ref:`physics_combining` for more information.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.StackedPhysics`) stacked operator

        """
        return stack(self, other)

    def forward(self, x, **kwargs):
        r"""
        Computes forward operator

        .. math::

                y = N(A(x), \sigma)


        :param torch.Tensor, list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) noisy measurements

        """
        return self.sensor(self.noise(self.A(x, **kwargs), **kwargs))

    def A(self, x, **kwargs):
        r"""
        Computes forward operator :math:`y = A(x)` (without noise and/or sensor non-linearities)

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) clean measurements

        """
        return self.forw(x, **kwargs)

    def sensor(self, x):
        r"""
        Computes sensor non-linearities :math:`y = \eta(y)`

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (:class:`torch.Tensor`) clean measurements
        """
        return self.sensor_model(x)

    def set_noise_model(self, noise_model, **kwargs):
        r"""
        Sets the noise model

        :param Callable noise_model: noise model
        """
        self.noise_model = noise_model

    def noise(self, x, **kwargs) -> Tensor:
        r"""
        Incorporates noise into the measurements :math:`\tilde{y} = N(y)`

        :param torch.Tensor x:  clean measurements
        :param None, float noise_level: optional noise level parameter
        :return: noisy measurements

        """

        return self.noise_model(x, **kwargs)

    def A_dagger(self, y, x_init=None):
        r"""
        Computes an inverse as:

        .. math::

            x^* \in \underset{x}{\arg\min} \quad \|\forw{x}-y\|^2.

        This function uses gradient descent to find the inverse. It can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :param None, torch.Tensor x_init: initial guess for the reconstruction. If `None` (default) it is set to the adjoint of the forward operator (it it exists) applied to the measurements :math:`y`, i.e., :math:`x_0 = A^{\top}y`.
        :return: (:class:`torch.Tensor`) The reconstructed image :math:`x`.

        """
        if self.solver == "gradient_descent":
            if x_init is None:
                if hasattr(self, "A_adjoint"):
                    x_init = self.A_adjoint(y)
                else:
                    raise ValueError(
                        "x_init must be provided for gradient descent solver if the physics does not have"
                        " A_adjoint defined."
                    )

            x = x_init

            lr = 1e-1
            loss = torch.nn.MSELoss()
            for _ in range(self.max_iter):
                x = x - lr * self.A_vjp(x, self.A(x) - y)
                err = loss(self.A(x), y)
                if err < self.tol:
                    break
        else:
            raise NotImplementedError(
                f"Solver {self.solver} not implemented for A_dagger"
            )

        return x.clone()

    def set_ls_solver(self, solver, max_iter=None, tol=None):
        r"""
        Change default solver for computing the least squares solution:

        .. math::

            x^* \in \underset{x}{\arg\min} \quad \|\forw{x}-y\|^2.

        :param str solver: solver to use. If the physics are non-linear, the only available solver is `'gradient_descent'`.
            For linear operators, the options are `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'` (see :func:`deepinv.optim.utils.least_squares` for more details).
        :param int max_iter: maximum number of iterations for the solver.
        :param float tol: relative tolerance for the solver, stopping when :math:`\|A(x) - y\| < \text{tol} \|y\|`.
        """

        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        self.solver = solver

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` evaluated at :math:`x`, defined as:

        .. math::

            A_{vjp}(x, v) = \left. \frac{\partial A}{\partial x}  \right|_x^\top  v.

        By default, the Jacobian is computed using automatic differentiation.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (:class:`torch.Tensor`) the VJP product between :math:`v` and the Jacobian.
        """
        _, vjpfunc = torch.func.vjp(self.A, x)
        return vjpfunc(v)[0]

    def update(self, **kwargs):
        r"""
        Update the parameters of the physics: forward operator and noise model.

        :param dict kwargs: dictionary of parameters to update.
        """
        self.update_parameters(**kwargs)
        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)

    def update_parameters(self, **kwargs):
        r"""

        Update the parameters of the forward operator.

        :param dict kwargs: dictionary of parameters to update.
        """
        if kwargs:
            for key, value in kwargs.items():
                if (
                    value is not None
                    and hasattr(self, key)
                    and isinstance(value, torch.Tensor)
                ):
                    self.register_buffer(key, value)

    # NOTE: Physics instances can hold instances of torch.Generator as
    # (possibly nested) attributes and they cannot be copied using deepcopy
    # natively. For this reason, we manually copy them beforehand and populate
    # the copies using the memo parameter of deepcopy. For more details, see:
    # https://github.com/pytorch/pytorch/issues/43672
    # https://github.com/pytorch/pytorch/pull/49840
    # https://discuss.pytorch.org/t/deepcopy-typeerror-cant-pickle-torch-c-generator-objects/104464
    def clone(self):
        r"""
        Clone the forward operator by performing deepcopy to copy all attributes to new memory.

        This method should be favored to `copy.deepcopy` as it works even in
        the presence of `torch.Generator` objects. See `this issue
        <https://github.com/pytorch/pytorch/issues/43672>` for more details.
        """
        memo = {}

        # Traverse the object hierarchy graph of the forward operator
        traversal_queue = [self]
        seen = set()
        while traversal_queue:
            # 1. Get the next node to process
            node = traversal_queue.pop()

            # 2. Process the node
            if isinstance(node, torch.Generator):
                generator_device = node.device
                obj_clone = torch.Generator(generator_device)
                obj_clone.set_state(node.get_state())
                node_id = id(node)
                memo[node_id] = obj_clone

            # 3. Compute its neighbors (attributes and values for mapping objects)
            neighbors = []

            # NOTE: Attribute resolution can be dynamic and return new objects,
            # preventing the algorithm from terminating. To avoid that, we use
            # insepct.getattr_static. We don't use inspect.getmembers_static
            # because it was only introduced in Python 3.11 and we currently
            # support Python 3.9 onwards.
            for attr in dir(node):
                value = inspect.getattr_static(node, attr, default=None)
                if value is not None:
                    neighbors.append(value)

            # NOTE: It is necessary to include values for mapping objects for
            # the case of submodules which are stored as entries in a
            # dictionary instead of directly as attributes.
            if isinstance(node, collections.abc.Mapping):
                neighbors += list(node.values())

            # 4. Queue the unseen neighbors
            for neighbor in neighbors:
                child_id = id(neighbor)
                if child_id not in seen:
                    seen.add(child_id)
                    traversal_queue.append(neighbor)

        return copy.deepcopy(self, memo=memo)


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

    :param Callable A: forward operator function which maps an image to the observed measurements :math:`x\mapsto y`.
        It is recommended to normalize it to have unit norm.
    :param None | Callable A_adjoint: transpose of the forward operator, which should verify the adjointness test.
        By default, it is set to `None`, which means that the adjoint is computed automatically using :func:`deepinv.physics.adjoint_function`.
        This automatic adjoint is computed using automatic differentiation, which is slower than a closed form adjoint, and can
        have a larger memory footprint. If you want to use the automatic adjoint, you should set the `img_size` parameter
        If you have a closed form for the adjoint, you can pass it as a callable function or rewrite the class method.
    :param tuple img_size: (optional, only required if A_adjoint is not provided) Size of the signal/image `x`, e.g. `(C, ...)` where `C` is the number of channels and `...` are the spatial dimensions,
        used for the automatic adjoint computation.
    :param Callable noise_model: function that adds noise to the measurements :math:`N(z)`.
        See the noise module for some predefined functions.
    :param Callable sensor_model: function that incorporates any sensor non-linearities to the sensing process,
        such as quantization or saturation, defined as a function :math:`\eta(z)`, such that
        :math:`y=\eta\left(N(A(x))\right)`. By default, the sensor_model is set to the identity :math:`\eta(z)=z`.
    :param int max_iter: If the operator does not have a closed form pseudoinverse, the conjugate gradient algorithm
        is used for computing it, and this parameter fixes the maximum number of conjugate gradient iterations.
    :param float tol: If the operator does not have a closed form pseudoinverse, a least squares algorithm
        is used for computing it, and this parameter fixes the relative tolerance of the least squares algorithm.
    :param str solver: least squares solver to use. Choose between `'CG'`, `'lsqr'`, `'BiCGStab'` and `'minres'`. See :func:`deepinv.optim.utils.least_squares` for more details.

    |sep|

    :Examples:

        Blur operator with a basic averaging filter applied to a 32x32 black image with
        a single white pixel in the center:

        >>> from deepinv.physics.blur import Blur, Downsampling
        >>> x = torch.zeros((1, 1, 32, 32)) # Define black image of size 32x32
        >>> x[:, :, 16, 16] = 1 # Define one white pixel in the middle
        >>> w = torch.ones((1, 1, 3, 3)) / 9 # Basic 3x3 averaging filter
        >>> physics = Blur(filter=w)
        >>> y = physics(x)

        Linear operators can also be stacked. The measurements produced by the resulting
        model are :class:`deepinv.utils.TensorList` objects, where each entry corresponds to the
        measurements of the corresponding operator (see :ref:`physics_combining` for more information):

        >>> physics1 = Blur(filter=w)
        >>> physics2 = Downsampling(img_size=((1, 32, 32)), filter="gaussian", factor=4)
        >>> physics = physics1.stack(physics2)
        >>> y = physics(x)

        Linear operators can also be composed by multiplying them:

        >>> physics = physics1 * physics2
        >>> y = physics(x)

        Linear operators also come with an adjoint, a pseudoinverse, and proximal operators in a given norm:

        >>> from deepinv.loss.metric import PSNR
        >>> physics = Blur(filter=w, padding='circular')
        >>> y = physics(x) # Compute measurements
        >>> x_dagger = physics.A_dagger(y) # Compute linear pseudoinverse
        >>> x_prox = physics.prox_l2(torch.zeros_like(x), y, 1.) # Compute prox at x=0
        >>> PSNR()(x, x_prox) > PSNR()(x, y) # Should be closer to the original
        tensor([True])

        The adjoint can be generated automatically using the :func:`deepinv.physics.adjoint_function` method
        which relies on automatic differentiation, at the cost of a few extra computations per adjoint call:

        >>> from deepinv.physics import LinearPhysics, adjoint_function
        >>> A = lambda x: torch.roll(x, shifts=(1,1), dims=(2,3)) # Shift image by one pixel
        >>> physics = LinearPhysics(A=A, A_adjoint=adjoint_function(A, (4, 1, 5, 5)))
        >>> x = torch.randn((4, 1, 5, 5))
        >>> y = physics(x)
        >>> torch.allclose(physics.A_adjoint(y), x) # We have A^T(A(x)) = x
        True

    """

    def __init__(
        self,
        A=lambda x, **kwargs: x,
        A_adjoint=None,
        img_size=None,
        noise_model=ZeroNoise(),
        sensor_model=lambda x: x,
        max_iter=50,
        tol=1e-4,
        solver="lsqr",
        **kwargs,
    ):
        super().__init__(
            A=A,
            noise_model=noise_model,
            sensor_model=sensor_model,
            max_iter=max_iter,
            solver=solver,
            tol=tol,
            **kwargs,
        )
        self.A_adj = A_adjoint
        self.img_size = img_size

    def A_adjoint(self, y, **kwargs):
        r"""
        Computes transpose of the forward operator :math:`\tilde{x} = A^{\top}y`.
        If :math:`A` is linear, it should be the exact transpose of the forward matrix.

        .. note::

            If the problem is non-linear, there is not a well-defined transpose operation,
            but defining one can be useful for some reconstruction networks, such as :class:`deepinv.models.ArtifactRemoval`.

        :param torch.Tensor y: measurements.
        :param None, torch.Tensor params: optional additional parameters for the adjoint operator.
        :return: (:class:`torch.Tensor`) linear reconstruction :math:`\tilde{x} = A^{\top}y`.

        """
        if self.A_adj is None:
            if self.img_size is None:
                raise ValueError(
                    "img_size must be set for using the automatic A_adjoint implementation."
                    "Set img_size in the constructor of the LinearPhyics class or pass it as a keyword argument."
                )
            else:
                tensor_size = (y.shape[0],) + self.img_size
            return adjoint_function(self.A, tensor_size, device=y.device)(y, **kwargs)
        else:
            return self.A_adj(y, **kwargs)

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` evaluated at :math:`x`, defined as:

        .. math::

            A_{vjp}(x, v) = \left. \frac{\partial A}{\partial x}  \right|_x^\top  v = \conj{A} v.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (:class:`torch.Tensor`) the VJP product between :math:`v` and the Jacobian.
        """
        return self.A_adjoint(v)

    def A_A_adjoint(self, y, **kwargs):
        r"""
        A helper function that computes :math:`A A^{\top}y`.

        This function can speed up computation when :math:`A A^{\top}` is available in closed form.
        Otherwise it just calls :func:`deepinv.physics.Physics.A` and :func:`deepinv.physics.LinearPhysics.A_adjoint`.

        :param torch.Tensor y: measurement.
        :return: (:class:`torch.Tensor`) the product :math:`AA^{\top}y`.
        """
        return self.A(self.A_adjoint(y, **kwargs), **kwargs)

    def A_adjoint_A(self, x, **kwargs):
        r"""
        A helper function that computes :math:`A^{\top}Ax`.

        This function can speed up computation when :math:`A^{\top}A` is available in closed form.
        Otherwise it just cals :func:`deepinv.physics.Physics.A` and :func:`deepinv.physics.LinearPhysics.A_adjoint`.

        :param torch.Tensor x: signal/image.
        :return: (:class:`torch.Tensor`) the product :math:`A^{\top}Ax`.
        """
        return self.A_adjoint(self.A(x, **kwargs), **kwargs)

    def __mul__(self, other):
        r"""
        Concatenates two linear forward operators :math:`A = A_1 \circ A_2` via the * operation

        The resulting linear operator keeps the noise and sensor models of :math:`A_1`.

        :param deepinv.physics.LinearPhysics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.LinearPhysics`) concatenated operator

        """
        return compose(other, self, max_iter=self.max_iter, tol=self.tol)

    def stack(self, other):
        r"""
        Stacks forward operators :math:`A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}`.

        The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
        each entry corresponds to the measurements of the corresponding operator.

        .. note::

            When using the ``stack`` operator between two noise objects, the operation will retain only the second
            noise.

        See :ref:`physics_combining` for more information.

        :param deepinv.physics.Physics other: Physics operator :math:`A_2`
        :return: (:class:`deepinv.physics.StackedPhysics`) stacked operator

        """
        return stack(self, other)

    def compute_norm(self, x0, max_iter=100, tol=1e-3, verbose=True, **kwargs):
        r"""
        Computes the spectral :math:`\ell_2` norm (Lipschitz constant) of the operator

        :math:`A^{\top}A`, i.e., :math:`\|A^{\top}A\|_2`,

        using the `power method <https://en.wikipedia.org/wiki/Power_iteration>`_.

        :param torch.Tensor x0: initialisation point of the algorithm
        :param int max_iter: maximum number of iterations
        :param float tol: relative variation criterion for convergence
        :param bool verbose: print information

        :returns z: (float) spectral norm of :math:`\conj{A} A`, i.e., :math:`\|\conj{A} A\|`.
        """
        x = torch.randn_like(x0)
        x /= torch.norm(x)
        zold = torch.zeros_like(x)
        for it in range(max_iter):
            y = self.A(x, **kwargs)
            y = self.A_adjoint(y, **kwargs)
            z = torch.matmul(x.conj().reshape(-1), y.reshape(-1)) / torch.norm(x) ** 2

            rel_var = torch.norm(z - zold)
            if rel_var < tol:
                if verbose:
                    print(
                        f"Power iteration converged at iteration {it}, value={z.item():.2f}"
                    )
                break
            zold = z
            x = y / torch.norm(y)
        else:
            warnings.warn("Power iteration: convergence not reached")

        return z.real

    def adjointness_test(self, u, **kwargs):
        r"""
        Numerically check that :math:`A^{\top}` is indeed the adjoint of :math:`A`.

        :param torch.Tensor u: initialisation point of the adjointness test method

        :return: (float) a quantity that should be theoretically 0. In practice, it should be of the order of the chosen dtype precision (i.e. single or double).

        """
        u_in = u  # .type(self.dtype)
        Au = self.A(u_in, **kwargs)

        if isinstance(Au, tuple) or isinstance(Au, list):
            V = [randn_like(au) for au in Au]
            Atv = self.A_adjoint(V, **kwargs)
            s1 = 0
            for au, v in zip_strict(Au, V):
                s1 += (v.conj() * au).flatten().sum()

        else:
            v = randn_like(Au)
            Atv = self.A_adjoint(v, **kwargs)

            s1 = (v.conj() * Au).flatten().sum()

        s2 = (Atv * u_in.conj()).flatten().sum()

        return s1.conj() - s2

    def condition_number(self, x, max_iter=500, tol=1e-6, verbose=False, **kwargs):
        r"""
        Computes an approximation of the condition number of the linear operator :math:`A`.

        Uses the LSQR algorithm, see :func:`deepinv.optim.utils.lsqr` for more details.

        :param torch.Tensor x: Any input tensor (e.g. random)
        :param int max_iter: maximum number of iterations
        :param float tol: relative variation criterion for convergence
        :param bool verbose: print information
        :return: (:class:`torch.Tensor`) condition number of the operator
        """
        y = self.A(x, **kwargs)
        _, cond = lsqr(
            self.A,
            self.A_adjoint,
            y,
            max_iter=max_iter,
            verbose=verbose,
            tol=tol,
            parallel_dim=None,
            **kwargs,
        )

        return cond

    def prox_l2(
        self, z, y, gamma, solver="CG", max_iter=None, tol=None, verbose=False, **kwargs
    ):
        r"""
        Computes proximal operator of :math:`f(x) = \frac{1}{2}\|Ax-y\|^2`, i.e.,

        .. math::

            \underset{x}{\arg\min} \; \frac{\gamma}{2}\|Ax-y\|^2 + \frac{1}{2}\|x-z\|^2

        :param torch.Tensor y: measurements tensor
        :param torch.Tensor z: signal tensor
        :param float gamma: hyperparameter of the proximal operator
        :return: (:class:`torch.Tensor`) estimated signal tensor

        """
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        if solver is not None:
            self.solver = solver

        return least_squares(
            self.A,
            self.A_adjoint,
            y,
            solver=solver,
            gamma=gamma,
            verbose=verbose,
            init=z,
            z=z,
            parallel_dim=[0],
            ATA=self.A_adjoint_A,
            AAT=self.A_A_adjoint,
            max_iter=self.max_iter,
            tol=self.tol,
            **kwargs,
        )

    def A_dagger(
        self, y, solver="CG", max_iter=None, tol=None, verbose=False, **kwargs
    ):
        r"""
        Computes the solution in :math:`x` to :math:`y = Ax` using a least squares solver.

        This function can be overwritten by a more efficient pseudoinverse in cases where closed form formulas exist.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :param str solver: least squares solver to use. Choose between 'CG', 'lsqr' and 'BiCGStab'. See :func:`deepinv.optim.utils.least_squares` for more details.
        :return: (:class:`torch.Tensor`) The reconstructed image :math:`x`.

        """
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        if solver is not None:
            self.solver = solver

        return least_squares(
            self.A,
            self.A_adjoint,
            y,
            parallel_dim=[0],
            AAT=self.A_A_adjoint,
            verbose=verbose,
            ATA=self.A_adjoint_A,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
            **kwargs,
        )


class ComposedPhysics(Physics):
    r"""
    Composes multiple physics operators into a single operator.

    The measurements produced by the resulting model are defined as

    .. math::

        \noise{\forw{x}} = N_k(A_k(\dots A_2(A_1(x))))

    where :math:`A_i(\cdot)` is the ith physics operator and :math:`N_k(\cdot)` is the noise of the last operator.

    :param list[deepinv.physics.Physics] *physics: list of physics to compose.
    """

    def __init__(self, *physics: Physics, device=None, **kwargs):
        super().__init__()

        self.physics_list = nn.ModuleList([])
        for physics_item in physics:
            self.physics_list.extend(
                [physics_item]
                if not isinstance(physics_item, ComposedPhysics)
                else physics_item.physics_list
            )
        self.noise_model = physics[-1].noise_model
        self.sensor_model = physics[-1].sensor_model
        self.to(device)

    def A(self, x: Tensor, **kwargs) -> Tensor:
        r"""
        Computes forward of composed operator

        .. math::

            y = A_k(\dots(A_1(x)))

        :param torch.Tensor x: signal/image
        :return: measurements
        """
        for physics in self.physics_list:
            x = physics.A(x, **kwargs)
        return x

    def update_parameters(self, **kwargs):
        r"""
        Updates the parameters of each operator in the composed operator.

        :param dict kwargs: dictionary of parameters to update.
        """
        for physics in self.physics_list:
            physics.update_parameters(**kwargs)

    def __str__(self):
        return (
            "ComposedPhysics("
            + "\n".join([f"{p}" for p in reversed(self.physics_list)])
            + ")"
        )

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        r"""
        Returns the physics operator at index `item`.

        :param int item: index of the physics operator
        """
        return self.physics_list[item]


class ComposedLinearPhysics(ComposedPhysics, LinearPhysics):
    r"""
    Composes multiple linear physics operators into a single operator.

    The measurements produced by the resulting model are defined as

    .. math::

        \noise{\forw{x}} = N_k(A_k \dots A_2(A_1(x)))

    where :math:`A_i(\cdot)` is the i-th physics operator and :math:`N_k(\cdot)` is the noise of the last operator.

    :param list[deepinv.physics.Physics] *physics: list of physics operators to compose.
    """

    def __init__(self, *physics: Physics, **kwargs):
        super().__init__(*physics, **kwargs)

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        r"""
        Computes adjoint of composed operator

        .. math::

            x = A_1^{\top} A_2^{\top} \dots A_k^{\top} y

        :param torch.Tensor y: measurements
        :return: signal/image
        """
        for physics in reversed(self.physics_list):
            y = physics.A_adjoint(y, **kwargs)
        return y


def compose(*physics: Union[Physics, LinearPhysics], **kwargs):
    r"""
    Composes multiple forward operators :math:`A = A_1\circ A_2\circ \dots \circ A_n`.

    The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
    each entry corresponds to the measurements of the corresponding operator.

    :param deepinv.physics.Physics physics: Physics operators :math:`A_i` to be composed.
    """
    if any(isinstance(phys, DecomposablePhysics) for phys in physics):
        warnings.warn(
            "At least one input physics is a DecomposablePhysics, but resulting physics will not be decomposable. `A_dagger` and `prox_l2` will fall back to approximate methods, which may impact performance."
        )

    if all(isinstance(phys, LinearPhysics) for phys in physics):
        return ComposedLinearPhysics(*physics, **kwargs)
    else:
        return ComposedPhysics(*physics, **kwargs)


class DecomposablePhysics(LinearPhysics):
    r"""
    Parent class for linear operators with SVD decomposition.

    The singular value decomposition is expressed as

    .. math::

        A = U\text{diag}(s)V^{\top} \in \mathbb{R}^{m\times n}

    where :math:`U\in\mathbb{C}^{n\times n}` and :math:`V\in\mathbb{C}^{m\times m}`
    are orthonormal linear transformations and :math:`s\in\mathbb{R}_{+}^{n}` are the singular values.

    :param None | Callable U: orthonormal transformation. If `None` (default), it is set to the identity function.
    :param None | Callable V_adjoint: transpose of V. If `None` (default), it is set to the identity function.
    :param tuple img_size: (optional, only required if V and/or U_adjoint are not provided) size of the signal/image `x`, e.g. `(C, ...)` where `C` is the number of channels and `...` are the spatial dimensions,
        used for the automatic adjoint computation.
    :param None | Callable U_adjoint: transpose of U. If `None` (default), it is computed automatically using :func:`deepinv.physics.adjoint_function`
        from the `U` function and the `img_size` parameter.
        This automatic adjoint is computed using automatic differentiation, which is slower than a closed form adjoint, and can
        have a larger memory footprint. If you want to use the automatic adjoint, you should set the `img_size` parameter.
    :param None | Callable V: If `None` (default), it is computed automatically using :func:`deepinv.physics.adjoint_function`
        from the `V_adjoint` function and the `img_size` parameter.
        This automatic adjoint is computed using automatic differentiation, which is slower than a closed form adjoint, and can
        have a larger memory footprint. If you want to use the automatic adjoint, you should set the `img_size` parameter.
    :param torch.nn.parameter.Parameter, float params: Singular values of the transform

    |sep|

    :Examples:

        Recreation of the Inpainting operator using the DecomposablePhysics class:

        >>> from deepinv.physics import DecomposablePhysics
        >>> seed = torch.manual_seed(0)  # Random seed for reproducibility
        >>> img_size = (1, 1, 3, 3)  # Input size
        >>> mask = torch.tensor([[1, 0, 1], [1, 0, 1], [1, 0, 1]])  # Binary mask
        >>> U = lambda x: x  # U is the identity operation
        >>> U_adjoint = lambda x: x  # U_adjoint is the identity operation
        >>> V = lambda x: x  # V is the identity operation
        >>> V_adjoint = lambda x: x  # V_adjoint is the identity operation
        >>> mask_svd = mask.float().unsqueeze(0).unsqueeze(0)  # Convert the mask to torch.Tensor and adjust its dimensions
        >>> physics = DecomposablePhysics(U=U, U_adjoint=U_adjoint, V=V, V_adjoint=V_adjoint, mask=mask_svd)

        Apply the operator to a random tensor:

        >>> x = torch.randn(img_size)
        >>> with torch.no_grad():
        ...     physics.A(x)  # Apply the masking
        tensor([[[[ 1.5410, -0.0000, -2.1788],
                  [ 0.5684, -0.0000, -1.3986],
                  [ 0.4033,  0.0000, -0.7193]]]])

    """

    def __init__(
        self,
        U=None,
        V_adjoint=None,
        img_size=None,
        U_adjoint=None,
        V=None,
        mask=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert not (
            U is None and not (U_adjoint is None)
        ), "U must be provided if U_adjoint is provided."
        assert not (
            V_adjoint is None and not (V is None)
        ), "V_adjoint must be provided if V is provided."

        # set to identity if not provided
        self._V_adjoint = (lambda x: x) if V_adjoint is None else V_adjoint
        self._U = (lambda x: x) if U is None else U

        # if U is the identity, we set U_adjoint as the identity as well
        self._U_adjoint = (lambda x: x) if U is None else U_adjoint

        # if V_adjoint is the identity, we set V as the identity as well
        self._V = (lambda x: x) if V_adjoint is None else V

        mask = torch.tensor(mask) if not isinstance(mask, torch.Tensor) else mask
        self.img_size = img_size
        self.register_buffer("mask", mask)

    def A(self, x, mask=None, **kwargs) -> Tensor:
        r"""
        Applies the forward operator :math:`y = A(x)`.

        If a mask/singular values is provided, it is used to apply the forward operator,
        and also stored as the current mask/singular values.

        :param torch.Tensor x: input tensor
        :param torch.nn.parameter.Parameter, float mask: singular values.
        :return: output tensor

        """

        self.update_parameters(mask=mask, **kwargs)

        return self.U(self.mask * self.V_adjoint(x))

    def A_adjoint(self, y, mask=None, **kwargs) -> Tensor:
        r"""
        Computes the adjoint of the forward operator :math:`\tilde{x} = A^{\top}y`.

        If a mask/singular values is provided, it is used to apply the adjoint operator,
        and also stored as the current mask/singular values.

        :param torch.Tensor y: input tensor
        :param torch.nn.parameter.Parameter, float mask: singular values.
        :return: output tensor
        """

        self.update_parameters(mask=mask, **kwargs)

        if isinstance(self.mask, float):
            mask = self.mask
        else:
            mask = torch.conj(self.mask)

        return self.V(mask * self.U_adjoint(y))

    def A_A_adjoint(self, y, mask=None, **kwargs):
        r"""
        A helper function that computes :math:`A A^{\top}y`.

        Using the SVD decomposition, we have :math:`A A^{\top} = U\text{diag}(s^2)U^{\top}`.

        :param torch.Tensor y: measurement.
        :return: (:class:`torch.Tensor`) the product :math:`AA^{\top}y`.
        """
        self.update_parameters(mask=mask, **kwargs)
        return self.U(self.mask.conj() * self.mask * self.U_adjoint(y))

    def A_adjoint_A(self, x, mask=None, **kwargs):
        r"""
        A helper function that computes :math:`A^{\top} A x`.

        Using the SVD decomposition, we have :math:`A^{\top}A = V\text{diag}(s^2)V^{\top}`.

        :param torch.Tensor x: signal/image.
        :return: (:class:`torch.Tensor`) the product :math:`A^{\top}Ax`.
        """
        self.update_parameters(mask=mask, **kwargs)
        return self.V(self.mask.conj() * self.mask * self.V_adjoint(x))

    def U(self, x):
        r"""
        Applies the :math:`U` operator of the SVD decomposition.

        .. note::

            This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

        :param torch.Tensor x: input tensor
        """
        return self._U(x)

    def V(self, x, **kwargs):
        r"""
        Applies the :math:`V` operator of the SVD decomposition.

        .. note::

            This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

        :param torch.Tensor x: input tensor
        """
        if self._V is None:
            if self.img_size is None:
                raise ValueError(
                    "img_size must be set for using the automatic V implementation. "
                    "Set img_size in the constructor of the DecomposablePhysics class or pass it as a keyword argument."
                )
            else:
                tensor_size = (x.shape[0],) + self.img_size
            return adjoint_function(self.V_adjoint, tensor_size, device=x.device)(
                x, **kwargs
            )
        else:
            return self._V(x)

    def U_adjoint(self, x, **kwargs):
        r"""
        Applies the :math:`U^{\top}` operator of the SVD decomposition.

        .. note::

            This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

        :param torch.Tensor x: input tensor
        """
        if self._U_adjoint is None:
            if self.img_size is None:
                raise ValueError(
                    "img_size must be set for using the automatic U_adjoint implementation. "
                    "Set img_size in the constructor of the DecomposablePhysics class or pass it as a keyword argument."
                )
            else:
                tensor_size = (x.shape[0],) + self.img_size
            return adjoint_function(self.U, tensor_size, device=x.device)(x, **kwargs)
        else:
            return self._U_adjoint(x)

    def V_adjoint(self, x):
        r"""
        Applies the :math:`V^{\top}` operator of the SVD decomposition.

        .. note::

            This method should be overwritten by the user to define its custom `DecomposablePhysics` operator.

        :param torch.Tensor x: input tensor
        """
        return self._V_adjoint(x)

    def prox_l2(self, z, y, gamma, **kwargs):
        r"""
        Computes proximal operator of :math:`f(x)=\frac{\gamma}{2}\|Ax-y\|^2`
        in an efficient manner leveraging the singular vector decomposition.

        :param torch.Tensor, float z: signal tensor
        :param torch.Tensor y: measurements tensor
        :param float gamma: hyperparameter :math:`\gamma` of the proximal operator
        :return: (:class:`torch.Tensor`) estimated signal tensor

        """
        b = self.A_adjoint(y) + 1 / gamma * z
        if isinstance(self.mask, float):
            scaling = self.mask**2 + 1 / gamma
        else:
            if (
                isinstance(gamma, torch.Tensor) and gamma.dim() < self.mask.dim()
            ):  # may be the case when mask is fft related
                gamma = gamma[(...,) + (None,) * (self.mask.dim() - gamma.dim())]
            scaling = torch.conj(self.mask) * self.mask + 1 / gamma
        x = self.V(self.V_adjoint(b) / scaling)
        return x

    def A_dagger(self, y, mask=None, **kwargs):
        r"""
        Computes :math:`A^{\dagger}y = x` in an efficient manner leveraging the singular vector decomposition.

        :param torch.Tensor y: a measurement :math:`y` to reconstruct via the pseudoinverse.
        :return: (:class:`torch.Tensor`) The reconstructed image :math:`x`.

        """
        self.update_parameters(mask=mask, **kwargs)

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

    :param torch.nn.Module noise: noise distribution, e.g., :class:`deepinv.physics.GaussianNoise`, or a user-defined torch.nn.Module. By default, it is set to Gaussian noise with a standard deviation of 0.1.

    |sep|

    :Examples:

        Denoising operator with Gaussian noise with standard deviation 0.1:

        >>> from deepinv.physics import Denoising, GaussianNoise
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = 0.5*torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> physics = Denoising(GaussianNoise(sigma=0.1))
        >>> with torch.no_grad():
        ...     physics(x)
        tensor([[[[ 0.7302, -0.2064, -1.0712],
                  [ 0.1985, -0.4322, -0.8064],
                  [ 0.2139,  0.3624, -0.3223]]]])

    """

    def __init__(self, noise_model: Optional[NoiseModel] = None, **kwargs):
        if noise_model is None:
            noise_model = GaussianNoise(sigma=0.1)
        super().__init__(noise_model=noise_model, **kwargs)


def adjoint_function(A, input_size, device="cpu", dtype=torch.float):
    r"""
    Provides the adjoint function of a linear operator :math:`A`, i.e., :math:`A^{\top}`.


    The generated function can be simply called as ``A_adjoint(y)``, for example:

    >>> import torch
    >>> from deepinv.physics.forward import adjoint_function
    >>> A = lambda x: torch.roll(x, shifts=(1,1), dims=(2,3)) # shift image by one pixel
    >>> x = torch.randn((4, 1, 5, 5))
    >>> y = A(x)
    >>> A_adjoint = adjoint_function(A, (4, 1, 5, 5))
    >>> torch.allclose(A_adjoint(y), x) # we have A^T(A(x)) = x
    True


    :param Callable A: linear operator :math:`A`.
    :param tuple input_size: size of the input tensor e.g. (B, C, H, W).
        The first dimension, i.e. batch size, should be equal or lower than the batch size B
        of the input tensor to the adjoint operator.
    :param str device: device where the adjoint operator is computed.
    :return: (Callable) function that computes the adjoint of :math:`A`.

    """
    x = torch.ones(input_size, device=device, dtype=dtype)
    (_, vjpfunc) = torch.func.vjp(A, x)
    batches = x.size()[0]

    # NOTE: In certain cases A(x) can't be automatically differentiated
    # infinitely many times wrt x. In that case, the adjoint of A computed
    # using automatic differentiation might not be automatically differentiable
    # either. We avoid this problem by using the involutive property of the
    # adjoint operator, i.e.,  (A^\top)^\top = A, and we specifically define
    # the adjoint of the adjoint as the original linear operator A. For more
    # details, see https://github.com/deepinv/deepinv/issues/511
    class Adjoint(torch.autograd.Function):
        @staticmethod
        def forward(y):
            if y.size()[0] < batches:
                y2 = torch.zeros(
                    (batches,) + y.shape[1:], device=y.device, dtype=y.dtype
                )
                y2[: y.size()[0], ...] = y
                return vjpfunc(y2)[0][: y.size()[0], ...]
            elif y.size()[0] > batches:
                raise ValueError(
                    "Batch size of A_adjoint input is larger than expected"
                )
            else:
                return vjpfunc(y)[0]

        @staticmethod
        def setup_context(ctx, inputs, outputs):
            pass

        @staticmethod
        def backward(ctx, grad_output):
            return A(grad_output)

    return Adjoint.apply


def stack(*physics: Union[Physics, LinearPhysics]):
    r"""
    Stacks multiple forward operators :math:`A = \begin{bmatrix} A_1(x) \\ A_2(x) \\ \vdots \\ A_n(x) \end{bmatrix}`.

    The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
    each entry corresponds to the measurements of the corresponding operator.

    :param deepinv.physics.Physics physics: Physics operators :math:`A_i` to be stacked.
    """
    if all(isinstance(phys, LinearPhysics) for phys in physics):
        return StackedLinearPhysics(physics)
    else:
        return StackedPhysics(physics)


class StackedPhysics(Physics):
    r"""
    Stacks multiple physics operators into a single operator.

    The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
    each entry corresponds to the measurements of the corresponding operator.

    See :ref:`physics_combining` for more information.

    :param list[deepinv.physics.Physics] physics_list: list of physics operators to stack.
    """

    def __init__(self, physics_list: list[Physics], **kwargs):
        super(StackedPhysics, self).__init__()

        self.physics_list = []
        for physics in physics_list:
            self.physics_list.extend(
                [physics]
                if not isinstance(physics, StackedPhysics)
                else physics.physics_list
            )

    def A(self, x: Tensor, **kwargs) -> TensorList:
        r"""
        Computes forward of stacked operator

        .. math::

            y = \begin{bmatrix} A_1(x) \\ A_2(x) \\ \vdots \\ A_n(x) \end{bmatrix}

        :param torch.Tensor x: signal/image
        :return: measurements
        """
        return TensorList([physics.A(x, **kwargs) for physics in self.physics_list])

    def __str__(self):
        return "StackedPhysics(" + "\n".join([f"{p}" for p in self.physics_list]) + ")"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        r"""
        Returns the physics operator at index `item`.

        :param int item: index of the physics operator
        """
        return self.physics_list[item]

    def sensor(self, y: TensorList, **kwargs) -> TensorList:
        r"""
        Applies sensor non-linearities to the measurements per physics operator
        in the stacked operator.

        :param deepinv.utils.TensorList y: measurements
        :return: measurements
        """
        for i, physics in enumerate(self.physics_list):
            y[i] = physics.sensor(y[i], **kwargs)
        return y

    def __len__(self):
        r"""
        Returns the number of physics operators in the stacked operator

        """
        return len(self.physics_list)

    def noise(self, y: TensorList, **kwargs) -> TensorList:
        r"""
        Applies noise to the measurements per physics operator in the stacked operator.

        :param deepinv.utils.TensorList y: measurements
        :return: noisy measurements
        """
        for i, physics in enumerate(self.physics_list):
            y[i] = physics.noise(y[i], **kwargs)
        return y

    def set_noise_model(self, noise_model, item=0):
        r"""
        Sets the noise model for the physics operator at index `item`.

        :param Callable, deepinv.physics.NoiseModel noise_model: noise model for the physics operator.
        :param int item: index of the physics operator
        """
        self.physics_list[item].set_noise_model(noise_model)

    def update_parameters(self, **kwargs):
        r"""
        Updates the parameters of the stacked operator.

        :param dict kwargs: dictionary of parameters to update.
        """
        for physics in self.physics_list:
            physics.update_parameters(**kwargs)


class StackedLinearPhysics(StackedPhysics, LinearPhysics):
    r"""
    Stacks multiple linear physics operators into a single operator.

    The measurements produced by the resulting model are :class:`deepinv.utils.TensorList` objects, where
    each entry corresponds to the measurements of the corresponding operator.

    See :ref:`physics_combining` for more information.

    :param list[deepinv.physics.Physics] physics_list: list of physics operators to stack.
    :param str reduction: how to combine tensorlist outputs of adjoint operators into single
        adjoint output. Choose between ``sum``, ``mean`` or ``None``.
    """

    def __init__(self, physics_list, reduction="sum", **kwargs):
        super(StackedLinearPhysics, self).__init__(physics_list, **kwargs)
        if reduction == "sum":
            self.reduction = sum
        elif reduction == "mean":
            self.reduction = lambda x: sum(x) / len(x)
        elif reduction in ("none", None):
            self.reduction = lambda x: x
        else:
            raise ValueError("reduction must be either sum, mean or none.")

    def A_adjoint(self, y: TensorList, **kwargs) -> torch.Tensor:
        r"""
        Computes the adjoint of the stacked operator, defined as

        .. math::

            A^{\top}y = \sum_{i=1}^{n} A_i^{\top}y_i.

        :param deepinv.utils.TensorList y: measurements
        """
        return self.reduction(
            [
                physics.A_adjoint(y[i], **kwargs)
                for i, physics in enumerate(self.physics_list)
            ]
        )


class DistributedPhysics(Physics):
    """
    Distributed non-linear physics.

    Users provide:
      - factory(i, device, shared) -> Physics
      - num_ops (total number of Physics to build)

    We shard indices across ranks, build only local operators, and
    gather per-op outputs into a TensorList in global order on out_device/CPU.
    """
    def __init__(
        self,
        factory: Callable[[int, torch.device, Optional[dict]], Physics],
        num_ops: int,
        shared: Optional[dict] = None,
        device_list: Optional[List[torch.device]] = None,
        out_device: Optional[torch.device] = None,
        sharding: str = "round_robin",
        **kwargs
    ):
        
        super().__init__(**kwargs)

        # Check parameters
        self.factory = factory
        self.num_ops = num_ops
        assert isinstance(self.num_ops, int), "num_ops must be an integer."
        assert self.num_ops > 0, "num_ops must be positive."

        # Dist info
        self.is_dist = dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_dist else 1
        self.rank = dist.get_rank() if self.is_dist else 0

        print(f"Rank {self.rank}/{self.world_size} building DistributedPhysics with {self.num_ops} total ops, sharding={sharding}")

        # device_list default: one GPU per process (safe for torchrun/srun)
        if device_list is None:
            if torch.cuda.is_available():
                device_list = [torch.device(f"cuda:{torch.cuda.current_device()}")]
            else:
                device_list = [torch.device("cpu")]
        self.device_list = device_list

        # Which operator indices we own on this rank
        self.local_indices = self._default_local_indices(self.num_ops, self.world_size, self.rank, sharding)

        print(f"Rank {self.rank} owns indices {self.local_indices}")

        # Prepare shared dict (broadcast from rank 0 if distributed)
        self.shared = self._broadcast_shared(shared)

        print(f"Rank {self.rank} received shared keys: {list(self.shared.keys()) if self.shared is not None else None}")

        # Build local physics
        self.physics = {}
        self.physics_device_map = {}
        for k, i in enumerate(self.local_indices):
            dev = self.device_list[k % len(self.device_list)]
            self.physics[i] = self.factory(i, dev, self.shared)
            self.physics_device_map[i] = dev
            self._move_physics_to_device(self.physics[i], dev)

        print(f"Rank {self.rank} built {len(self.physics)} local physics.")

        # out_device default: current device for NCCL; CPU otherwise
        if out_device is None:
            if torch.cuda.is_available():
                out_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                out_device = torch.device("cpu")

        # Enforce NCCL sanity
        if self.is_dist and dist.get_backend().lower() == "nccl":
            if out_device.type != "cuda":
                raise RuntimeError("NCCL backend requires CUDA tensors for collectives.")
            current_device = torch.cuda.current_device()
            if out_device.index != current_device:
                if self.rank == 0:
                    print(f"[WARN] out_device={out_device} != current cuda:{current_device}; overriding.")
                out_device = torch.device(f"cuda:{current_device}")

        self.out_device = out_device


        print(f"Rank {self.rank} will gather outputs on device {self.out_device}")

    def _default_local_indices(self, num_ops: int, world_size: int, rank: int, sharding: str) -> List[int]:
        if sharding == "round_robin":
            return [i for i in range(num_ops) if (i % world_size) == rank]
        elif sharding == "block":
            ops_per_rank = (num_ops + world_size - 1) // world_size
            start = rank * ops_per_rank
            end = min(start + ops_per_rank, num_ops)
            return list(range(start, end))
        else:
            raise ValueError("sharding must be either 'round_robin' or 'block'.")
    
    def _broadcast_shared(self, shared: Optional[dict]) -> Optional[dict]:
        if not self.is_dist:
            return shared
        obj = [shared if self.rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def _move_physics_to_device(self, physics: Physics, device: torch.device | str):
        physics.to(device)
        # Try to move forward callable if it carries state
        forw = getattr(physics, "forw", None)
        if isinstance(forw, nn.Module):
            forw.to(device)
        elif hasattr(forw, "to") and callable(forw.to):
            forw.to(device)
    
    def _gather_pairs_all(self, local_pairs: List[Tuple[int, torch.Tensor]]) -> List[Tuple[int, torch.Tensor]]:
        if not self.is_dist:
            return local_pairs
        bucket = [None] * self.world_size
        dist.all_gather_object(bucket, local_pairs)
        merged = []
        for part in bucket:
            merged.extend(part)
        return merged

    def _order_by_index(self, pairs: List[Tuple[int, torch.Tensor]], n: int) -> List[torch.Tensor]:
        out = [None] * n
        for i, t in pairs:
            out[i] = t
        assert all(x is not None for x in out), "Missing pieces after gather."
        return [t for t in out]

    def _broadcast_input_tensor(self, x: torch.Tensor, *, src: int = 0) -> torch.Tensor:
        """
        Ensure all ranks have the same input tensor x on self.out_device,
        without the user having to do any broadcast.

        - Rank 'src' provides x; others allocate an empty tensor of the same (dtype, shape).
        - Broadcast happens on self.out_device (CUDA for NCCL fast path).
        """
        if not self.is_dist:
            return x

        # Move source tensor to the target device (others will allocate on that device too)
        if self.rank == src:
            x_dev = x.to(self.out_device, non_blocking=True)
            metadata = (x_dev.dtype, tuple(x_dev.shape))
        else:
            x_dev, metadata = None, None

        # Share (dtype, shape) so non-src can allocate correctly
        metadatas = [None] * self.world_size

        print(f"Rank {self.rank} broadcasting input tensor metadata from rank {src}")

        dist.all_gather_object(metadatas, metadata)

        print(f"Rank {self.rank} received input tensor metadata from all ranks")

        chosen = next((m for m in metadatas if m is not None), None)
        if chosen is None:
            raise RuntimeError("Broadcast failed: could not infer input tensor metadata.")
        dtype, shape = chosen

        if self.rank != src:
            x_dev = torch.empty(shape, dtype=dtype, device=self.out_device)

        # Now broadcast from src into every rank's preallocated buffer
        dist.broadcast(x_dev, src=src)

        print(f"Rank {self.rank} completed input tensor broadcast from rank {src}")

        return x_dev

    def _broadcast_tensorlist(self, tl: TensorList, *, src: int = 0) -> TensorList:
        """
        Ensure every rank has tl[i] for all i=0..num_ops-1 on self.out_device.
        Safe to call even if tl is already present everywhere.
        """
        if not self.is_dist:
            # move to out_device for consistency
            return TensorList([t.to(self.out_device, non_blocking=True) for t in tl])

        out = []
        for i in range(self.num_ops):
            if self.rank == src:
                t_src = tl[i].to(self.out_device, non_blocking=True)
                meta = (t_src.dtype, tuple(t_src.shape))
            else:
                t_src, meta = None, None

            metas = [None] * self.world_size
            dist.all_gather_object(metas, meta)
            chosen = next((m for m in metas if m is not None), None)
            if chosen is None:
                raise RuntimeError(f"Could not infer metadata for tensorlist element {i}.")
            dtype, shape = chosen

            if self.rank != src:
                t_src = torch.empty(shape, dtype=dtype, device=self.out_device)
            dist.broadcast(t_src, src=src)
            out.append(t_src)
        return TensorList(out)

    def A(self, x: torch.Tensor, **kwargs) -> TensorList:
        """Distributed forward. Each rank computes its share, then we gather to build a TensorList

        Args:
            x (torch.Tensor): _input tensor_

        Raises:
            ValueError: _description_
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            TensorList: _description_
        """

        x_dev_all = self._broadcast_input_tensor(x, src=0)

        local_pairs = []
        for i in self.local_indices:
            dev = self.physics_device_map[i]
            xi = x_dev_all.to(dev, non_blocking=True)
            yi = self.physics[i].A(xi, **kwargs)
            yi = yi.to(self.out_device, non_blocking=True)
            local_pairs.append((i, yi))

        print(f"Rank {self.rank} completed local computations, gathering results...")
        
        all_pairs = self._gather_pairs_all(local_pairs)

        print(f"Rank {self.rank} gathered all results, ordering...")

        ordered = self._order_by_index(all_pairs, self.num_ops)

        print(f"Rank {self.rank} completed ordering of results.")
        return TensorList(ordered)


class DistributedLinearPhysics(DistributedPhysics, LinearPhysics):
    """Distributed linear physics with reduction for adjoint/VJP.

    The factory should return LinearPhysics instances.

    reduction: 'sum' | 'mean' | 'none'/None
    - 'sum' (default): sum over i of A_i^T y_i (or VJP)
    - 'mean': average of the sum
    - 'none': return a TensorList of per-operator adjoints
    """
    def __init__(
        self,
        factory: Callable[[int, torch.device, Optional[dict]], LinearPhysics],
        num_ops: int,
        shared: Optional[dict] = None,
        reduction: str = "sum",
        **kwargs
    ):
        LinearPhysics.__init__(self)
        super().__init__(factory=factory, num_ops=num_ops, shared=shared, **kwargs)
        if reduction not in ("sum", "mean", "none", None):
            raise ValueError("reduction must be either 'sum', 'mean' or 'none'.")
        self.reduction_mode = "none" if reduction in ("none", None) else reduction

        for i, p in self.physics.items():
            if not isinstance(p, LinearPhysics):
                raise ValueError("factory must return LinearPhysics instances.")
    
    def _sum_tensor_list(self, ts: List[torch.Tensor]) -> torch.Tensor:
        out = ts[0].clone()
        for t in ts[1:]:
            out.add_(t)
        return out

    def _linear_reduce(
        self,
        build_local_fn: Callable[[int, torch.device], torch.Tensor],
        op_name: str,
        **kwargs,
    ) -> torch.Tensor | TensorList:
        """
        Shared implementation for adjoint / vjp reductions.

        - Builds per-op local tensors on each owned device via `build_local_fn(i, dev)`.
        - If reduction_mode == "none": gathers a TensorList ordered by global index and returns it.
        - Else: sums locals, discovers (dtype, shape) globally, synthesizes zeros on empty ranks,
                and performs a distributed reduction.
        We use `dist.all_reduce` so that the result is available on all ranks.
        """
        # 1) Compute local contributions
        local_pairs = []
        for i in self.local_indices:
            dev = self.physics_device_map[i]
            assert dev is not None
            yi = build_local_fn(i, dev)
            yi = yi.to(self.out_device, non_blocking=True)
            local_pairs.append((i, yi))

        # 2) No-reduction mode → gather ordered TensorList
        if self.reduction_mode == "none":
            all_pairs = self._gather_pairs_all(local_pairs)
            ordered = self._order_by_index(all_pairs, self.num_ops)
            return TensorList(ordered)

        # 3) Operator-wise reduction path (sum/mean)
        local_tensors = [t for _, t in local_pairs]
        has_local = len(local_tensors) > 0

        if has_local:
            local_sum = self._sum_tensor_list(local_tensors)
            local_meta = (local_sum.dtype, tuple(local_sum.shape))
        else:
            local_sum = None
            local_meta = None

        if self.is_dist:
            # 3a) Discover dtype/shape from ANY rank that has it
            metas = [None] * self.world_size
            dist.all_gather_object(metas, local_meta)
            chosen = next((m for m in metas if m is not None), None)
            if chosen is None:
                raise RuntimeError(f"No partials produced; cannot reduce {op_name}.")
            dtype, shape = chosen

            # 3b) Synthesize zero on ranks without locals
            if not has_local:
                local_sum = torch.zeros(shape, dtype=dtype, device=self.out_device)

            # 3c) All reduction sum
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)

        else:
            # Single-process path
            if not has_local:
                raise RuntimeError(f"No local tensors and not distributed — nothing to reduce in {op_name}.")

        if self.reduction_mode == "mean":
            local_sum = local_sum / self.num_ops
        return local_sum

    def A_adjoint(self, y: TensorList, **kwargs) -> torch.Tensor | TensorList:
        """
        Computes the adjoint (Hermitian transpose) of the forward operator for a list of input tensors.
        This method applies the adjoint operator `A_adjoint` of each physics module in the `self.physics` list
        to the corresponding element in the input tensor list `y`. The results are then reduced using the
        `_linear_reduce` method.
        Args:
            y (TensorList): A list of input tensors, where each tensor corresponds to the output of a forward operator.
            **kwargs: Additional keyword arguments passed to each physics module's `A_adjoint` method.
        Returns:
            torch.Tensor | TensorList: The result of applying the adjoint operator to the input tensors.
        """

        y_all = self._broadcast_tensorlist(y, src=0)

        return self._linear_reduce(
            lambda i, dev: self.physics[i].A_adjoint(y_all[i].to(dev, non_blocking=True), **kwargs),
            op_name="A_adjoint",
            **kwargs,
        )

    def A_vjp(self, x: torch.Tensor, v: TensorList, **kwargs) -> torch.Tensor | TensorList:
        """
        Computes the vector-Jacobian product (VJP) of the forward operator A with respect to the input tensor `x` and vector `v`.
        This method applies the VJP operation for each physics operator in a possibly batched or distributed setting, reducing the results as needed.
        Args:
            x (torch.Tensor): The input tensor with respect to which the VJP is computed.
            v (TensorList): A list of tensors representing the vectors to be multiplied with the Jacobian of the forward operator.
            **kwargs: Additional keyword arguments passed to the underlying physics operators.
        Returns:
            torch.Tensor | TensorList: The result of the VJP operation, which may be a single tensor, or a list of tensors.
        """

        x_all = self._broadcast_input_tensor(x, src=0)
        v_all = self._broadcast_tensorlist(v, src=0)

        return self._linear_reduce(
            lambda i, dev: self.physics[i].A_vjp(x_all.to(dev, non_blocking=True),
                                                v_all[i].to(dev, non_blocking=True), **kwargs),
            op_name="A_vjp",
            **kwargs,
        )
