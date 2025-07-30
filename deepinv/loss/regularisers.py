import torch
from deepinv.loss.loss import Loss


class JacobianSpectralNorm(Loss):
    r"""
    Computes the spectral norm of the Jacobian.

    Given a function :math:`f:\mathbb{R}^n\to\mathbb{R}^n`, this module computes the spectral
    norm of the Jacobian of :math:`f` in :math:`x`, i.e.

    .. math::

        \|\frac{df}{du}(x)\|_2.

    This spectral norm is computed with a power method leveraging jacobian vector products, as proposed by :footcite:t:`pesquet2021learning`.

    .. note::

        This implementation assumes that the input :math:`x` is batched with shape `(B, ...)`, where B is the batch size.

    :param int max_iter: maximum numer of iteration of the power method.
    :param float tol: tolerance for the convergence of the power method.
    :param bool eval_mode: set to ``False`` if one does not want to backpropagate through the spectral norm (default), set to ``True`` otherwise.
    :param bool verbose: whether to print computation details or not.
    :param str reduction: reduction in batch dimension. One of ["mean", "sum", "max"], operation to be performed after all spectral norms have been computed. If ``None``, a vector of length ``batch_size`` will be returned. Defaults to "max".
    :param int reduced_batchsize: if not `None`, the batch size will be reduced to this value for the computation of the spectral norm. Can be useful to reduce memory usage and computation time when the batch size is large.

    |sep|

    :Examples:

    .. doctest::

        >>> import torch
        >>> from deepinv.loss.regularisers import JacobianSpectralNorm
        >>> _ = torch.manual_seed(0)
        >>>
        >>> reg_l2 = JacobianSpectralNorm(max_iter=100, tol=1e-5, eval_mode=False, verbose=True)
        >>> A = torch.diag(torch.Tensor(range(1, 51))).unsqueeze(0)  # creates a diagonal matrix with largest eigenvalue = 50
        >>> x = torch.randn((1, A.shape[1])).unsqueeze(0).requires_grad_()
        >>> out = x @ A
        >>> regval = reg_l2(out, x)
        >>> print(regval) # returns approx 50
        tensor(49.9999)

    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-3,
        eval_mode: bool = False,
        verbose: bool = False,
        reduction: str = "max",
        reduced_batchsize: int = None,
    ):
        super(JacobianSpectralNorm, self).__init__()
        self.name = "jsn"
        self.max_iter = max_iter
        self.tol = tol
        self.eval = eval_mode
        self.verbose = verbose
        self.reduced_batchsize = reduced_batchsize

        self.reduction = lambda x: x
        if reduction is not None:
            if not isinstance(reduction, str):
                raise ValueError("Reduction should be a string or None.")
            elif reduction.lower() == "mean":
                self.reduction = lambda x: torch.mean(x)
            elif reduction.lower() == "sum":
                self.reduction = lambda x: torch.sum(x)
            elif reduction.lower() == "max":
                self.reduction = lambda x: torch.max(x)
            elif reduction.lower() == "none":
                pass
            else:
                raise ValueError(
                    'Reduction should be "mean", "sum", "max", "none" or None.'
                )

    @staticmethod
    def _batched_dot(x, y):
        """
        Computes the dot product between corresponding batch elements.

        :param torch.Tensor x: tensor of shape (B, N)
        :param torch.Tensor y: tensor of shape alike to x

        Returns 1D tensor wth
        """

        return torch.einsum("bn,bn->b", x, y)

    def _reduce_batch(self, x, y):
        """
        Reduces the batch dimension of the input tensors x and y.
        """
        if self.reduced_batchsize is not None:
            x = x[: self.reduced_batchsize]
            y = y[: self.reduced_batchsize]
        return x, y

    def forward(self, y, x, **kwargs):
        """
        Computes the spectral norm of the Jacobian of :math:`f` in :math:`x`.

        .. warning::

            The input :math:`x` must have requires_grad=True before evaluating :math:`f`.

        :param torch.Tensor y: output of the function :math:`f` at :math:`x`, of dimension `(B, ...)`
        :param torch.Tensor x: input of the function :math:`f`, of dimension `(B, ...)`

        If x has multiple dimensions, it's assumed the first one corresponds to the batch dimension.
        """

        x, y = self._reduce_batch(x, y)

        assert x.shape[0] == y.shape[0], ValueError(
            f"x and y should have the same number of instances. Got {x.shape[0]} vs. {y.shape[0]}"
        )

        n_dims = x.dim()

        u = torch.randn_like(x)
        # Normalize each batch element
        u = u / torch.norm(u.flatten(start_dim=1, end_dim=-1), p=2, dim=-1).view(
            -1, *[1] * (n_dims - 1)
        )

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

            # multiply corresponding batch elements
            z = (
                self._batched_dot(
                    u.flatten(start_dim=1, end_dim=-1),
                    v.flatten(start_dim=1, end_dim=-1),
                )
                / torch.norm(u.flatten(start_dim=1, end_dim=-1), p=2, dim=-1) ** 2
            )

            if it > 0:
                rel_var = torch.norm(z - zold)
                if rel_var < self.tol and self.verbose:
                    print(
                        "Power iteration converged at iteration: ",
                        it,
                        ", val: ",
                        z.sqrt().tolist(),
                        ", relvar :",
                        rel_var.item(),
                    )
                    break
            zold = z.detach().clone()

            u = v / torch.norm(v.flatten(start_dim=1, end_dim=-1), p=2, dim=-1).view(
                -1, *[1] * (n_dims - 1)
            )

            if self.eval:
                w.detach_()
                v.detach_()
                u.detach_()

        return self.reduction(z.view(-1).sqrt())


class FNEJacobianSpectralNorm(Loss):
    r"""
    Computes the Firm-Nonexpansiveness Jacobian spectral norm.

    Given a function :math:`f:\mathbb{R}^n\to\mathbb{R}^n`, this module computes the spectral
    norm of the Jacobian of :math:`2f-\operatorname{Id}` (where :math:`\operatorname{Id}` denotes the
    identity) in :math:`x`, i.e.

    .. math::

        \|\frac{d(2f-\operatorname{Id})}{du}(x)\|_2,

    as proposed by :footcite:t:`pesquet2021learning`.
    This spectral norm is computed with the :class:`deepinv.loss.JacobianSpectralNorm` class.

    .. note::

        This implementation assumes that the input :math:`x` is batched with shape `(B, ...)`, where B is the batch size.

    :param int max_iter: maximum numer of iteration of the power method.
    :param float tol: tolerance for the convergence of the power method.
    :param bool eval_mode: set to ``False`` if one does not want to backpropagate through the spectral norm (default), set to ``True`` otherwise.
    :param bool verbose: whether to print computation details or not.
    :param str reduction: reduction in batch dimension. One of ["mean", "sum", "max"], operation to be performed after all spectral norms have been computed. If ``None``, a vector of length ``batch_size`` will be returned. Defaults to "max".
    :param int reduced_batchsize: if not `None`, the batch size will be reduced to this value for the computation of the spectral norm. Can be useful to reduce memory usage and computation time when the batch size is large.

    |sep|

    :Examples:

    .. doctest::

        >>> import torch
        >>> from deepinv.loss.regularisers import FNEJacobianSpectralNorm
        >>> _ = torch.manual_seed(0)
        >>>
        >>> reg_fne = FNEJacobianSpectralNorm(max_iter=100, tol=1e-5, eval_mode=False, verbose=True)
        >>> A = torch.diag(torch.Tensor(range(1, 51))).unsqueeze(0)  # creates a diagonal matrix with largest eigenvalue = 50
        >>>
        >>> def model_base(x):
        ...     return x @ A
        >>>
        >>> def FNE_model(x):
        ...     A_bis = torch.linalg.inv((A + torch.eye(A.shape[1])))  # Creates the resolvent of A, which is firmly nonexpansive
        ...     return x @ A_bis
        >>>
        >>> x = torch.randn((1, A.shape[1])).unsqueeze(0)
        >>>
        >>> out = model_base(x)
        >>> regval = reg_fne(out, x, model_base)
        >>> print(regval) # returns approx 99 (model is expansive, with Lipschitz constant 50)
        tensor(98.9999)
        >>> out = FNE_model(x)
        >>> regval = reg_fne(out, x, FNE_model)
        >>> print(regval) # returns a value smaller than 1 (model is firmly nonexpansive)
        tensor(0.9595)
    """

    def __init__(
        self,
        max_iter: int = 10,
        tol: float = 1e-3,
        eval_mode: bool = False,
        verbose: bool = False,
        reduction: str = "max",
        reduced_batchsize: int = None,
    ):
        super(FNEJacobianSpectralNorm, self).__init__()

        self.reduced_batchsize = reduced_batchsize

        self.spectral_norm_module = JacobianSpectralNorm(
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            eval_mode=eval_mode,
            reduction=reduction,
            reduced_batchsize=reduced_batchsize,
        )

    def _reduce_batch(self, x, y):
        """
        Reduces the batch dimension of the input tensors x and y.
        """
        if self.reduced_batchsize is not None:
            x = x[: self.reduced_batchsize]
            y = y[: self.reduced_batchsize]
        return x, y

    def forward(
        self, y_in, x_in, model, *args_model, interpolation=False, **kwargs_model
    ):
        r"""
        Computes the Firm-Nonexpansiveness (FNE) Jacobian spectral norm of a model.

        :param torch.Tensor y_in: input of the model (by default), of dimension `(B, ...)`.
        :param torch.Tensor x_in: an additional point of the model (by default), of dimension `(B, ...)`.
        :param torch.nn.Module model: neural network, or function, of which we want to compute the FNE Jacobian spectral norm.
        :param `*args_model`: additional arguments of the model.
        :param bool interpolation: whether to input to model an interpolation between y_in and x_in instead of y_in (default is `False`).
        :param `**kargs_model`: additional keyword arguments of the model.
        """

        y_in, x_in = self._reduce_batch(y_in, x_in)

        if interpolation:
            eta = torch.rand(
                (y_in.size(0),) + (1,) * (y_in.dim() - 1), requires_grad=True
            ).to(y_in.device)
            x = eta * y_in.detach() + (1 - eta) * x_in.detach()
        else:
            x = y_in

        x.requires_grad_()
        x_out = model(x, *args_model, **kwargs_model)

        y = 2.0 * x_out - x

        return self.spectral_norm_module(y, x)
