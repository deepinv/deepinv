from __future__ import annotations
from deepinv.utils import zeros_like
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable
from deepinv.utils.tensorlist import TensorList
from deepinv.utils.compat import zip_strict
import warnings
from typing import Callable
from .bicgstab import bicgstab
from .conjugate_gradient import conjugate_gradient
from .lsqr import lsqr
from .minres import minres


def least_squares(
    A: Callable,
    AT: Callable,
    y: Tensor,
    z: Tensor | float | None = 0.0,
    init: Tensor | None = None,
    gamma: float | Tensor | None = None,
    parallel_dim: int = 0,
    AAT: Callable | None = None,
    ATA: Callable | None = None,
    solver: str = "CG",
    max_iter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> Tensor:
    r"""
    Solves :math:`\min_x \|Ax-y\|^2 + \frac{1}{\gamma}\|x-z\|^2` using the specified solver.

    The solvers are stopped either when :math:`\|Ax-y\| \leq \text{tol} \times \|y\|` or
    when the maximum number of iterations is reached.

    The solution depends on the regularization parameter :math:`\gamma`:

    - If `gamma=None` (:math:`\gamma = \infty`), it solves the unregularized least squares problem :math:`\min_x \|Ax-y\|^2`.
        - If :math:`A` is overcomplete (rows>=columns), it computes the minimum norm solution :math:`x = A^{\top}(AA^{\top})^{-1}y`.
        - If :math:`A` is undercomplete (columns>rows), it computes the least squares solution :math:`x = (A^{\top}A)^{-1}A^{\top}y`.
    - If :math:`0 < \gamma < \infty`, it computes the least squares solution :math:`x = (A^{\top}A + \frac{1}{\gamma}I)^{-1}(A^{\top}y + \frac{1}{\gamma}z)`.

    .. warning::

        If :math:`\gamma \leq 0`, the problem can become non-convex and the solvers are not designed for that.
        A warning is raised, but solvers continue anyway (except for LSQR, which cannot be used for negative :math:`\gamma`).

    Available solvers are:

    - `'CG'`: `Conjugate Gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_.
    - `'BiCGStab'`: `Biconjugate Gradient Stabilized method <https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method>`_
    - `'lsqr'`: `Least Squares QR <https://www-leland.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf>`_
    - `'minres'`: `Minimal Residual Method <https://en.wikipedia.org/wiki/Minimal_residual_method>`_

    .. note::

        Both `'CG'` and `'BiCGStab'` are used for squared linear systems, while `'lsqr'` is used for rectangular systems.

        If the chosen solver requires a squared system, we map to the problem to the normal equations:
        If the size of :math:`y` is larger than :math:`x` (overcomplete problem), it computes :math:`(A^{\top} A)^{-1} A^{\top} y`,
        otherwise (incomplete problem) it computes :math:`A^{\top} (A A^{\top})^{-1} y`.


    :param Callable A: Linear operator :math:`A` as a callable function.
    :param Callable AT: Adjoint operator :math:`A^{\top}` as a callable function.
    :param torch.Tensor y: input tensor of shape (B, ...)
    :param torch.Tensor z: input tensor of shape (B, ...) or scalar.
    :param torch.Tensor init: (Optional) initial guess for the solver. If None, it is set to a tensor of zeros.
    :param None, float, torch.Tensor gamma: (Optional) inverse regularization parameter. Can be batched (shape (B, ...)) or a scalar.
        If multi-dimensional tensor, then its shape must match that of :math:`A^{\top} y`.
        If None, it is set to :math:`\infty` (no regularization).
    :param str solver: solver to be used, options are `'CG'`, `'BiCGStab'`, `'lsqr'` and `'minres'`.
    :param Callable AAT: (Optional) Efficient implementation of :math:`A(A^{\top}(x))`. If not provided, it is computed as :math:`A(A^{\top}(x))`.
    :param Callable ATA: (Optional) Efficient implementation of :math:`A^{\top}(A(x))`. If not provided, it is computed as :math:`A^{\top}(A(x))`.
    :param int max_iter: maximum number of iterations.
    :param float tol: relative tolerance for stopping the algorithm.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param kwargs: Keyword arguments to be passed to the solver.
    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...).
    """
    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]

    if gamma is None:
        gamma = torch.tensor(0.0, device=y.device)
        gamma_provided = False
    else:
        gamma_provided = True

        if not isinstance(gamma, Tensor):
            gamma = torch.tensor(gamma, device=y.device)

        if torch.any(gamma <= 0):
            warnings.warn(
                "Regularization parameter of least squares problem (gamma) should be positive."
                "Otherwise, the problem can become non-convex and the solvers are not designed for that."
                "Continuing anyway..."
            )

    Aty = AT(y)

    if gamma.ndim > 0:  # if batched gamma
        if isinstance(Aty, TensorList):
            batch_size = Aty[0].size(0)
            ndim = Aty[0].ndim
        else:
            batch_size = Aty.size(0)
            ndim = Aty.ndim

        if gamma.size(0) != batch_size:
            raise ValueError(
                "If gamma is batched, its batch size must match the one of y."
            )
        elif gamma.ndim == 1:  # expand gamma to ATy
            gamma = gamma.view([gamma.size(0)] + [1] * (ndim - 1))
        elif gamma.ndim != ndim:
            raise ValueError(
                f"gamma should either be 0D, 1D, or match same number of dimensions as ATy, but got ndims {gamma.ndim} and {ndim}"
            )

    if solver == "lsqr":  # rectangular solver
        eta = 1 / gamma if gamma_provided else None
        x, _ = lsqr(
            A,
            AT,
            y,
            x0=z,
            eta=eta,
            max_iter=max_iter,
            tol=tol,
            parallel_dim=parallel_dim,
            **kwargs,
        )

    else:
        complete = Aty.shape == y.shape
        overcomplete = Aty.numel() < y.numel()

        if complete and (solver == "BiCGStab" or solver == "minres"):
            H = lambda x: A(x)
            b = y
        else:
            if AAT is None:
                AAT = lambda x: A(AT(x))
            if ATA is None:
                ATA = lambda x: AT(A(x))

            if gamma_provided:
                b = AT(y) + 1 / gamma * z
                H = lambda x: ATA(x) + 1 / gamma * x
                overcomplete = False
            else:
                if not overcomplete:
                    H = lambda x: AAT(x)
                    b = y
                else:
                    H = lambda x: ATA(x)
                    b = Aty

        if solver == "CG":
            x = conjugate_gradient(
                A=H,
                b=b,
                init=init,
                max_iter=max_iter,
                tol=tol,
                parallel_dim=parallel_dim,
                **kwargs,
            )
        elif solver == "BiCGStab":
            x = bicgstab(
                A=H,
                b=b,
                init=init,
                max_iter=max_iter,
                tol=tol,
                parallel_dim=parallel_dim,
                **kwargs,
            )
        elif solver == "minres":
            x = minres(
                A=H,
                b=b,
                init=init,
                max_iter=max_iter,
                tol=tol,
                parallel_dim=parallel_dim,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Solver {solver} not recognized. Choose between 'CG', 'lsqr' and 'BiCGStab'."
            )

        if not gamma_provided and not overcomplete and not complete:
            x = AT(x)
    return x


class LeastSquaresSolver(torch.autograd.Function):
    r"""
    Custom autograd function for the least squares solver to enable O(1) memory backward propagation using implicit differentiation.

    The forward pass solves the following problem using :func:`deepinv.optim.linear.least_squares`:

    .. math::

        \min_x \|A_\theta x - y \|^2 + \frac{1}{\gamma} \|x - z\|^2

    where :math:`A_\theta` is a linear operator :class:`deepinv.physics.LinearPhysics` parameterized by :math:`\theta`.

    .. note::

        This function uses a :func:`least squares <deepinv.optim.utils.least_squares>` solver under the hood, which supports various solvers such as Conjugate Gradient (CG), BiCGStab, LSQR, and MinRes (see :func:`deepinv.optim.linear.least_squares` for more details).

    The backward pass computes the gradients with respect to the inputs using implicit differentiation.
    """

    # NOTE: the physics parameters are handled as side-effects (not inputs/outputs of the autograd function)
    # we add a dummy input tensor `trigger` to trigger backward when needed (i.e. when physics parameters require grad)
    @staticmethod
    def forward(
        ctx,
        physics,
        y: Tensor,
        z: Tensor,
        init: Tensor,
        gamma: float | Tensor,
        trigger: Tensor = None,
        extra_kwargs: dict = None,
    ):

        kwargs = extra_kwargs if extra_kwargs is not None else {}

        with torch.no_grad():
            solution = least_squares(
                A=physics.A,
                AT=physics.A_adjoint,
                y=y,
                z=z,
                init=init,
                gamma=gamma,
                AAT=physics.A_A_adjoint,
                ATA=physics.A_adjoint_A,
                **kwargs,
            )

        # Save tensors only
        gamma_orig_shape = gamma.shape
        # For broadcasting with other tensors. Note we already have checked gamma shapes
        # in forward, so the following is just for gamma batched but not shaped.
        if gamma.ndim == 1:
            if isinstance(solution, TensorList):
                ndim = solution[0].ndim
            else:
                ndim = solution.ndim

            gamma = gamma.view([gamma.size(0)] + [1] * (ndim - 1))

        ctx.save_for_backward(solution, y, z, gamma)
        # Save other non-tensor contexts
        ctx.physics = physics
        ctx.kwargs = kwargs
        ctx.gamma_orig_shape = gamma_orig_shape

        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, ...]:
        h, y, z, gamma = ctx.saved_tensors
        physics = ctx.physics

        # Solve (A^T A + I/gamma) mv = grad_output
        with torch.no_grad():
            mv = least_squares(
                A=physics.A,
                AT=physics.A_adjoint,
                y=torch.zeros_like(y),
                z=grad_output * gamma,
                gamma=gamma,
                AAT=physics.A_A_adjoint,
                ATA=physics.A_adjoint_A,
                **ctx.kwargs,
            )

        # Build grads aligned with inputs: (physics, y, z, gamma, solver, max_iter, tol, ..., **kwargs)
        needs = ctx.needs_input_grad
        grads = [None] * len(needs)

        # dL/dy = A mv
        if needs[1]:
            grads[1] = physics.A(mv)

        # dL/dz = (1/gamma) mv
        if needs[2]:
            grads[2] = mv / gamma

        # dL/dgamma = <mv, h - z> / gamma^2
        if needs[4]:
            diff = h - z
            # vdot gives correct conjugation for complex; .real keeps real scalar
            grad_gamma = torch.sum(
                mv.conj() * diff, dim=list(range(1, mv.ndim)), keepdim=True
            ).real / (gamma**2)

            # If gamma was batched in the forward, we return a batched grad
            if len(ctx.gamma_orig_shape) > 0:
                grad_gamma = grad_gamma.view(ctx.gamma_orig_shape)
            # gamma was a scalar in the forward, we accumulate all grads to return a scalar, similar to
            # torch autograd behavior for broadcasted inputs
            else:
                grad_gamma = torch.sum(grad_gamma).view(())
            grads[4] = grad_gamma

        # Optional: implicit grads w.r.t physics parameters (side-effect accumulation)
        params = [
            p for p in getattr(physics, "buffers", lambda: [])() if p.requires_grad
        ]
        if params:
            # pseudo-loss = - <mv, A^T (A h - y)>
            h_det = h.detach()
            y_det = y.detach()
            with torch.enable_grad():
                pseudo = -torch.vdot(
                    mv.flatten(), physics.A_adjoint(physics.A(h_det) - y_det).flatten()
                ).real
            g_params = torch.autograd.grad(
                pseudo, params, retain_graph=False, allow_unused=True
            )
            for p, g in zip_strict(
                params,
                g_params,
            ):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.detach()
                    else:
                        p.grad = p.grad + g.detach()

        return tuple(grads)


# wrapper of the autograd function for easier use
def least_squares_implicit_backward(
    physics,
    y: Tensor,
    z: Tensor = None,
    init: Tensor = None,
    gamma: float | Tensor = None,
    **kwargs,
) -> Tensor:
    r"""
    Least squares solver with O(1) memory backward propagation using implicit differentiation.
    The function is similar to :func:`deepinv.optim.linear.least_squares` for the forward pass, but uses implicit differentiation for the backward pass, which reduces memory consumption to O(1) in the number of iterations.

    This function supports backpropagation with respect to the inputs :math:`y`, :math:`z` and :math:`\gamma` and also with respect to the parameters of the physics operator :math:`A_\theta` if they require gradients. See :ref:`sphx_glr_auto_examples_unfolded_demo_unfolded_constant_memory.py` and the notes below for more details.

    Let :math:`h(z, y, \theta, \gamma)` denote the output of the least squares solver, i.e. the solution of the following problem:

    .. math::

        h(z, y, \theta, \gamma) = \underset{x}{\arg\min} \; \frac{\gamma}{2}\|A_\theta x-y\|^2 + \frac{1}{2}\|x-z\|^2

    When the forward least-squares solver converges to the exact minimizer, we have the following closed-form expressions for :math:`h(z, y, \theta, \gamma)`:

    .. math::

        h(z, y, \theta, \gamma) = \left( A_\theta^{\top} A_\theta + \frac{1}{\gamma} I \right)^{-1} \left( A_\theta^{\top} y + \frac{1}{\gamma} z \right)

    Let :math:`M` denote the inverse :math:`\left( A_\theta^T A_\theta + \frac{1}{\gamma} I \right)^{-1}`. In the forward, we need to compute the vector-Jacobian products (VJPs), which can be computed as follows:

    .. math::

        \left( \frac{\partial h}{\partial z} \right)^{\top} v               &= \frac{1}{\gamma} M v \\
        \left( \frac{\partial h}{\partial y} \right)^{\top} v               &= A_\theta M v \\
        \left( \frac{\partial h}{\partial \gamma} \right)^{\top} v          &=   (h - z)^\top M  v / \gamma^2 \\
        \left( \frac{\partial h}{\partial \theta} \right)^{\top} v          &= \frac{\partial p}{\partial \theta}

    where :math:`p =  (y - A_\theta h)^{\top} A_\theta M v` and :math:`\frac{\partial p}{\partial \theta}` can be computed using the standard backpropagation mechanism (autograd).

    .. note::

        This function only supports first-order gradients. Higher-order gradients are not supported. If you need higher-order gradients, please use :func:`deepinv.optim.linear.least_squares` instead but be aware that it requires storing all intermediate iterates, which can be memory-intensive.

    .. note::

        This function also supports implicit gradients with respect to the parameters of the physics operator :math:`A_\theta` if they require gradients. This is useful for learning the physics parameters in an end-to-end fashion. The gradients are accumulated in-place in the `.grad` attribute of the parameters of the physics operator. To make this work, the function takes as input the physics operator itself (not just its matmul functions) and checks if any of its parameters require gradients. If so, it triggers the backward pass accordingly.

    .. warning::

        Implicit gradients can be incorrect if the least squares solver does not converge sufficiently. Make sure to set the `max_iter` and `tol` parameters of the least squares solver appropriately to ensure convergence. You can monitor the convergence by setting `verbose=True` in the least squares solver via `kwargs`. If the solver does not converge, the implicit gradients can be very inaccurate and lead to divergence of the training.

    .. warning::

        This function does not support :class:`deepinv.utils.TensorList` inputs yet. If you use :class:`deepinv.utils.TensorList` as inputs, the function will fall back to standard least squares with full backpropagation.

    .. tip::

        If you do not need gradients with respect to the physics parameters, you can set `requires_grad=False` for all parameters of the physics operator to avoid the additional backward pass. This can save some computation time.

    .. tip::

        Training unfolded network with implicit differentiation can reduce memory consumption significantly, especially when using many iterations. On GPU, we can expect a memory reduction factor of about 2x-3x compared to standard backpropagation and a speed-up of about 1.2x-1.5x. The exact numbers depend on the problem and the number of iterations.

    :param deepinv.physics.LinearPhysics physics: physics operator :class:`deepinv.physics.LinearPhysics`.
    :param torch.Tensor y: input tensor of shape (B, ...)
    :param torch.Tensor, float, int, None  z: input tensor of shape (B, ...). Default is `None`, which corresponds to a zero tensor.
    :param None, torch.Tensor init: Optional initial guess, only used for the forward pass. Default is `None`, which corresponds to a zero initialization.
    :param None, float, torch.Tensor gamma: regularization parameter :math:`\gamma > 0`. Default is `None`. Can be batched (shape (B, ...)) or a scalar.
    :param kwargs: additional arguments to be passed to the least squares solver.

    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...), the solution of the least squares problem.
    """

    if z is None:
        # To get correct shape
        z = zeros_like(physics.A_adjoint(y))
    elif isinstance(z, float) or isinstance(z, int):
        z = torch.full_like(physics.A_adjoint(y), fill_value=float(z))
    if init is None:
        init = zeros_like(z)

    # NOTE: TensorList not supported by autograd function, we fall back to standard least_squares in this case for now
    if isinstance(y, TensorList):
        warnings.warn(
            "Warning: least_squares_implicit_backward does not support TensorList inputs. Falling back to standard least_squares with full backpropagation."
        )
        return least_squares(
            A=physics.A,
            AT=physics.A_adjoint,
            y=y,
            z=z,
            init=init,
            gamma=gamma,
            AAT=physics.A_A_adjoint,
            ATA=physics.A_adjoint_A,
            **kwargs,
        )

    physics_requires_grad_params = any(
        p.requires_grad for p in getattr(physics, "buffers", lambda: [])()
    )
    # NOTE: backward is triggered if any of the inputs require grad
    # When input tensors (y, z, gamma) do not require grad and we want to do backward w.r.t physics parameters, we need to trigger it manually by a dummy tensor.
    trigger_backward = (
        y.requires_grad
        or z.requires_grad
        or (isinstance(gamma, Tensor) and gamma.requires_grad)
        or physics_requires_grad_params
    )
    if trigger_backward:
        trigger = torch.ones(1, device=y.device, dtype=y.dtype).requires_grad_(
            True
        )  # Dummy tensor to trigger backward
    else:
        trigger = torch.ones(1, device=y.device, dtype=y.dtype)
    extra_kwargs = kwargs if kwargs else None
    dtype = y.dtype if not torch.is_complex(y) else y.real.dtype
    if gamma is None:
        gamma = torch.zeros((), device=y.device, dtype=dtype)
    if isinstance(gamma, Tensor) and gamma.ndim > 0:
        if gamma.size(0) != y.size(0):
            raise ValueError(
                "If gamma is batched, its batch size must match the one of y."
            )
    if not isinstance(gamma, Tensor):
        gamma = torch.as_tensor(gamma, device=y.device, dtype=dtype)
    return LeastSquaresSolver.apply(physics, y, z, init, gamma, trigger, extra_kwargs)
