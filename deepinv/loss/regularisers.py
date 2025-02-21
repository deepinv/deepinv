import torch
from deepinv.loss.loss import Loss


class JacobianSpectralNorm(Loss):
    r"""
    Computes the spectral norm of the Jacobian.

    Given a function :math:`f:\mathbb{R}^n\to\mathbb{R}^n`, this module computes the spectral
    norm of the Jacobian of :math:`f` in :math:`x`, i.e.

    .. math::
        \|\frac{df}{du}(x)\|_2.

    This spectral norm is computed with a power method leveraging jacobian vector products, as proposed in `<https://arxiv.org/abs/2012.13247v2>`_.

    :param int max_iter: maximum numer of iteration of the power method.
    :param float tol: tolerance for the convergence of the power method.
    :param bool eval_mode: set to `False` if one does not want to backpropagate through the spectral norm (default), set to `True` otherwise.
    :param bool verbose: whether to print computation details or not.

    |sep|

    :Examples:

    .. doctest::

        >>> import torch
        >>> from deepinv.loss.regularisers import JacobianSpectralNorm
        >>> _ = torch.manual_seed(0)
        >>> _ = torch.cuda.manual_seed(0)
        >>>
        >>> reg_l2 = JacobianSpectralNorm(max_iter=10, tol=1e-3, eval_mode=False, verbose=True)
        >>> A = torch.diag(torch.Tensor(range(1, 51)))  # creates a diagonal matrix with largest eigenvalue = 50
        >>> x = torch.randn_like(A).requires_grad_()
        >>> out = A @ x
        >>> regval = reg_l2(out, x)
        >>> print(regval) # returns approx 50
        tensor([49.0202])
    """

    def __init__(self, max_iter=10, tol=1e-3, eval_mode=False, verbose=False):
        super(JacobianSpectralNorm, self).__init__()
        self.name = "jsn"
        self.max_iter = max_iter
        self.tol = tol
        self.eval = eval_mode
        self.verbose = verbose

    def forward(self, y, x, **kwargs):
        """
        Computes the spectral norm of the Jacobian of :math:`f` in :math:`x`.

        .. warning::
            The input :math:`x` must have requires_grad=True before evaluating :math:`f`.

        :param torch.Tensor y: output of the function :math:`f` at :math:`x`.
        :param torch.Tensor x: input of the function :math:`f`.
        """
        u = torch.randn_like(x)
        u = u / torch.norm(u.flatten(), p=2)

        zold = torch.zeros_like(u)

        for it in range(self.max_iter):
            # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            w = torch.ones_like(y, requires_grad=True)
            v = torch.autograd.grad(
                torch.autograd.grad(y, x, w, create_graph=True),
                w,
                u,
                create_graph=not self.eval,
            )[
                0
            ]  # v = A(u)

            (v,) = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)

            z = torch.dot(u.flatten(), v.flatten()) / torch.norm(u, p=2) ** 2

            if it > 0:
                rel_var = torch.norm(z - zold)
                if rel_var < self.tol and self.verbose:
                    print(
                        "Power iteration converged at iteration: ",
                        it,
                        ", val: ",
                        z.sqrt().item(),
                        ", relvar :",
                        rel_var.item(),
                    )
                    break
            zold = z.detach().clone()

            u = v / torch.norm(v.flatten(), p=2)

            if self.eval:
                w.detach_()
                v.detach_()
                u.detach_()

        return z.view(-1).sqrt()


class FNEJacobianSpectralNorm(Loss):
    r"""
    Computes the Firm-Nonexpansiveness Jacobian spectral norm.

    Given a function :math:`f:\mathbb{R}^n\to\mathbb{R}^n`, this module computes the spectral
    norm of the Jacobian of :math:`2f-\operatorname{Id}` (where :math:`\operatorname{Id}` denotes the
    identity) in :math:`x`, i.e.

    .. math::
        \|\frac{d(2f-\operatorname{Id})}{du}(x)\|_2,

    as proposed in `<https://arxiv.org/abs/2012.13247v2>`_.
    This spectral norm is computed with the :class:`deepinv.loss.JacobianSpectralNorm` class.

    :param int max_iter: maximum numer of iteration of the power method.
    :param float tol: tolerance for the convergence of the power method.
    :param bool eval_mode: set to `False` if one does not want to backpropagate through the spectral norm (default), set to `True` otherwise.
    :param bool verbose: whether to print computation details or not.

    """

    def __init__(self, max_iter=10, tol=1e-3, verbose=False, eval_mode=False):
        super(FNEJacobianSpectralNorm, self).__init__()
        self.spectral_norm_module = JacobianSpectralNorm(
            max_iter=max_iter, tol=tol, verbose=verbose, eval_mode=eval_mode
        )

    def forward(
        self, y_in, x_in, model, *args_model, interpolation=False, **kwargs_model
    ):
        r"""
        Computes the Firm-Nonexpansiveness (FNE) Jacobian spectral norm of a model.

        :param torch.Tensor y_in: input of the model (by default).
        :param torch.Tensor x_in: an additional point of the model (by default).
        :param torch.nn.Module model: neural network, or function, of which we want to compute the FNE Jacobian spectral norm.
        :param `*args_model`: additional arguments of the model.
        :param bool interpolation: whether to input to model an interpolation between y_in and x_in instead of y_in (default is `False`).
        :param `**kargs_model`: additional keyword arguments of the model.
        """

        if interpolation:
            eta = torch.rand(y_in.size(0), 1, 1, 1, requires_grad=True).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_in.detach()
        else:
            x = y_in

        x.requires_grad_()
        x_out = model(x, *args_model, **kwargs_model)

        y = 2.0 * x_out - x

        return self.spectral_norm_module(y, x)
