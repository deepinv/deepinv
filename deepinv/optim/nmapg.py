import torch
import numpy as np
import warnings


def nonmonotone_accelerated_proximal_gradient(
    x0,
    f,
    y=None,
    nabla_f=None,
    f_and_nabla=None,
    g=None,
    prox_g=None,
    weighting=1.0,  # minimize f + weighting*g
    max_iter=2000,
    L_init=1,
    tol=1e-4,
    rho=0.9,
    delta=0.1,
    eta=0.8,
    verbose=False,
):
    """
    Implements the nonmonotone accelerated proximal gradient algorithm (nmAPG).
    It serves as a general purpose algorithm to minimize non-convex functions of the form

    .. math::
        \begin{equation*}
        F(x)=f(x)+\lambda g(x)
        \end{equation*}

    for differentiable :math:`f` (with known locally Lipschitz gradient) and possibly non-smooth :math:`g` (with known proximal operator).
    The algorithm admits convergence guarantees towards a stationary point also in the case of non-convex functions
    :math:`f` and :math:`g`.
    It combines the proximal gradient algorithm with momentum and a line search.

    In our implementation, the functions :math:`F`, :math:`f` and :math:`g` depends on some additional parameter :math:`y\in\mathbb{R}^d`.
    Moreover, the implementation uses batching over the parameters :math:`y` such that several problems of the form

    .. math::
        \begin{equation*}
        \min_y F(x,y_i)=f(x,y_i)+\lambda g(x,y_i),\quad i=1,...,\mathrm{batch size}
        \end{equation*}

    can be solved simultanously.
    To this end, the function takes a tensor `y` containing the paramters, where the first dimension is the batch dimension
    and an initialization tensor `x0` with the same first dimension as batch dimension.



    Reference:

    Huan Li, Zhouchen Lin
    Accelerated Proximal Gradient Methods for Nonconvex Programming
    NeurIPS 2015

    We use the variant with Barzilai-Borwein step sizes as summmarized in Algorithm 4 in the
    supplementary material of the paper.

    :param torch.Tensor x0: Initialization :math:`x_0` of the algorithm. The first dimension is a batch dimension (with `y.shape[0]==x0.shape[0]` if `y is not None`).
    :param Callable f: Differentiable part :math:`f(x,y)` of the objective function. It takes two inputs: the argument `x` (we are minimizing over `x`)
        and the parmeters `y` (which remains fixed over the optimization).
    :param torch.Tensor y: Parameter `y` from the objective function. The first dimension is a batch dimension with `y.shape[0]==x0.shape[0]`. Can be set to `None`, in which case :math:`f`, :math:`g`, :math:`\nabla f`
        and :math:`\mathrm{prox}_g` should not take `y` as an input argument.
    :param Callable nabla_f: gradient :math:`\nabla_x f(x,y)` of :math:`f`. `None` for computing the gradient via autodiff. Default: `None`
    :param Callable f_and_nabla: A function computing both, the function value and the gradient of :math:`f`. Set to `None` to call `f` and `nabla_f` separately
        (which might be inefficient). Default: `None`
    :param Callable g: Possibly non-smooth part :math:`g(x,y)` of the objective function It takes two inputs: the argument `x` (we are minimizing over `x`)
        and the parmeters `y` (which remains fixed over the optimization). Default: `lambda x, y: 0` (i.e. choose :math:`g(x,y)=0`).
    :param Callable prox_g: Proximal operator :math:`\mathrm{prox}_{\gamma g(\cdot,y)}(x)` of :math:`g`. It takes three inputs: the argument `x` (we are minimizing over `x`),
        the parmeters `y` (which remains fixed over the optimization) and the step size :math:`gamma` of the proximal operator. `None` for using the identity.
        Note that `prox_g` must not be `None` if `g is not None`. Default: `None`.
    :param float weighting: Parameter :math:`\lambda` from the objective value. Default: `1.0`
    :param int max_iter: Maximal number of iterations. Default: `2000`
    :param float L_init: Initial guess of the (local) Lipschitz constant of :math:`\nabla_x f(x,y)`. Default: `1.0`
    :param float tol: Convergence crieterion. The algorithm is stopped if the relative residual between two iterates is smaller than `tol`. Default: `1e-4`
    :param float rho: Decrease ratio of the step size if the line search fails. Default: `0.9`
    :param float delta: Quadratic decay parameter for the line search. Should be :math:`>0`. Default: `0.1`
    :param float eta: Non-monotonicity parameter for the line search. Should be in :math:`[0,1)`. Default: `0.8`
    :param bool verbose: Set to `True` to print the number of iterations used in the algorithm. Default: `False`.
    :return: Tuple out of the approximated minimizer `x` of :math:`F`, the estimated local Lipschitz constant `L`, the number `i` of used iterations and
        a bool `converged` indicating whether the algorithm reached the convergence criterion or not.
    """
    if y is None:
        y = x.clone()
        f_ = f
        f = lambda x, y: f_(x)
        if nabla_f is not None:
            nabla_f_ = nabla_f
            nabla_f = lambda x, y: nabla_f_(x, y)
        if g is not None:
            g_ = g
            g = lambda x, y: g_(x)
        if prox_g is not None:
            prox_g_ = prox_g
            prox_g = lambda x, y, param: prox_g(x, param)
    if f_and_nabla is None:
        if nabla_f is None:

            def f_and_nabla(x, y):
                with torch.enable_grad():
                    x_ = x.clone()
                    x_.requires_grad_(True)
                    z = torch.sum(f(x_, y))
                    grad = torch.autograd.grad(z, x_)[0]
                return z.detach(), grad.detach()

        else:
            f_and_nabla = lambda x, y: (f(x, y), nabla_f(x, y))
    if g is None:
        g = lambda x, y: 0
        prox_g = lambda x, y, param: x
    elif prox_g is None:
        raise ValueError("If g is used, prox_g has to be defined (given: prox_g=None)!")

    # initialize variables
    x = x0.clone()  # Noation of the paper: x1
    x_old = x.clone()  # x0
    z = x0.clone()  # z1
    t = 1.0  # t1
    t_old = 0.0  # t0
    q = 1.0  # q1
    c = f(x, y) + weighting * g(x, y)  # c1
    L = torch.full((x.shape[0],), L_init, dtype=torch.float32, device=x.device)
    while len(L.shape) < len(x0.shape):
        L = L.unsqueeze(-1)
    L_old = L.clone()
    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)
    grad_f = torch.zeros_like(x)  # nabla F(x)
    x_bar = torch.zeros_like(x)
    x_bar_old = x_bar.clone()
    grad_f_old = grad_f.clone()

    # Main loop
    for i in range(max_iter):
        assert not torch.any(
            torch.isnan(x)
        ), "Numerical errors! Some values became NaN!"
        x_bar[idx] = (
            x[idx]
            + t_old / t * (z[idx] - x[idx])
            + (t_old - 1) / t * (x[idx] - x_old[idx])
        )  # Eq 148, x_bar = yk
        x_old.copy_(x)
        energy_f, grad_f[idx] = f_and_nabla(x_bar[idx], y[idx])
        energy_g = g(x_bar[idx], y[idx])
        energy = energy_f + weighting * energy_g

        # Lipschitz Update (Barzilai-Borwein style step)
        if i > 0:
            dx = grad_f[idx] - grad_f_old[idx]  # r in the paper
            s = (dx * dx).sum(list(range(1, len(x0.shape))), keepdim=True)  # r^Tr
            L[idx] = torch.clip(
                s
                / (dx * (x_bar[idx] - x_bar_old[idx]))
                .sum(list(range(1, len(x0.shape))), keepdim=True)
                .abs()
                .clip(min=0.0, max=None),  # alpha_y = <s,r>/<r,r> in paper, Eq 150
                min=1.0,
                max=None,
            )  # clips for stability --> on a long term we can adjust min-clip based on the spectral norm of physics.A
        # line search on z (Eq 151 and 152)
        idx_search = idx
        idx_sub = torch.arange(0, idx.shape[0], device=x.device)
        energy_new = energy.clone()
        dx = z[idx] - x_bar[idx]
        for ii in range(150):
            z[idx_search] = prox_g(
                x_bar[idx_search] - grad_f[idx_search] / L[idx_search],
                y[idx_search],
                weighting / L[idx_search],
            )  # Eq 151, 1/L = alpha_y
            dx[idx_sub] = z[idx_search] - x_bar[idx_search]
            bound = torch.max(
                energy[idx_sub, None, None, None], c[idx_search, None, None, None]
            ) - delta * (dx[idx_sub] * dx[idx_sub]).sum(
                list(range(1, len(x0.shape))), keepdim=True
            )

            energy_new_ = f(z[idx_search], y[idx_search]) + weighting * g(
                z[idx_search], y[idx_search]
            )
            if torch.all(energy_new_ <= bound.view(-1)):
                energy_new[idx_sub] = energy_new_
                break
            energy_new[idx_sub] = energy_new_
            idx_sub = idx_sub[energy_new_ > bound.squeeze()]
            idx_search = idx[idx_sub]
            L[idx_search] = L[idx_search] / rho
        # If for Eq 153-158
        idx2 = (
            (
                energy_new[:]
                >= (c[idx] - delta * (dx * dx).sum(list(range(1, len(x0.shape)))))
            )
            .nonzero()
            .view(-1)
        )
        if idx2.nelement() > 0:
            idx_idx2 = idx[idx2]
            grad_fx = nabla_f(x[idx_idx2], y[idx_idx2])  # nabla f(xk)

            if i > 0:
                dx = grad_fx - grad_f_old[idx_idx2]
                s = (dx * dx).sum(list(range(1, len(x0.shape))), keepdim=True)
                L[idx_idx2] = torch.clip(
                    s
                    / (dx * (x[idx_idx2] - x_bar_old[idx_idx2]))
                    .sum(list(range(1, len(x0.shape))), keepdim=True)
                    .clip(min=0, max=None),
                    min=1.0,
                    max=None,
                )
            L_old.copy_(L)

            # Line search on v
            for ii in range(150):
                v = prox_g(
                    x[idx_idx2] - grad_fx / L[idx_idx2],
                    y[idx_idx2],
                    weighting / L[idx_idx2],
                )
                dx = v - x[idx_idx2]
                bound = c[idx_idx2, None, None, None] - delta * (dx * dx).sum(
                    list(range(1, len(x0.shape))), keepdim=True
                )
                energy_new2 = f(v, y[idx_idx2]) + weighting * g(v, y[idx_idx2])
                if torch.all(energy_new2 <= bound.view(-1) * (1 + 1e-4)):
                    break
                L[idx_idx2] = torch.where(
                    energy_new2[:, None, None, None] <= bound,
                    L[idx_idx2],
                    L[idx_idx2] / rho,
                )
            x[idx] = z[idx]
            idx3 = (energy_new2 <= energy_new[idx2]).nonzero().view(-1)
            tmp = idx_idx2[idx3]
            x[tmp] = v[idx3]
        else:
            x[idx] = z[idx]

        if i > 0:
            res[idx] = torch.norm(
                x[idx] - x_old[idx], p=2, dim=list(range(1, len(x0.shape)))
            ) / torch.norm(x[idx], p=2, dim=list(range(1, len(x0.shape))))
        assert not torch.any(
            torch.isnan(res)
        ), "Numerical errors! Some values became NaN!"
        condition = res >= tol
        idx = condition.nonzero().view(-1)  # Update which data to still iterate on

        if torch.max(res) < tol:
            if verbose:
                print(f"Converged in iter {i}, tol {torch.max(res).item():.6f}")
            break
        t_old = t
        t = (np.sqrt(4.0 * t_old ** 2 + 1.0) + 1.0) / 2.0  # Eq 159
        q_old = q
        q = eta * q + 1.0  # Eq 160
        c[idx] = (
            eta * q_old * c[idx] + f(x[idx], y[idx]) + weighting * g(x[idx], y[idx])
        ) / q  # Eq 161
        x_bar_old.copy_(x_bar)
        grad_f_old.copy_(grad_f)
    if verbose and (torch.max(res) >= tol):
        warnings.warn(f"max iter reached, tol {torch.max(res).item():.6f}")
    converged = res < tol
    return x, L, i, converged
