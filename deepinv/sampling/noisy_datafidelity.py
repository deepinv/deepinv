from __future__ import annotations
import torch
from deepinv.optim import DataFidelity, Distance
from deepinv.optim.linear import conjugate_gradient
import deepinv as dinv
from deepinv.physics import Physics
from deepinv.models import Denoiser


class NoisyDataFidelity(DataFidelity):
    r"""
    Preconditioned data fidelity term for noisy data :math:`- \log p(y|x + \sigma(t) \omega)`
    with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This is a base class for the conditional classes for approximating :math:`\log p_t(y|x_t)` used in diffusion
    algorithms for inverse problems, in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method computing the score :math:`\nabla_{x_t} \log p_t(y|x_t)`.

    By default we have

    .. math::

         \nabla_{x_t} \log p(y|x + \sigma(t) \omega) = P(\forw{x_t'}-y),


    where :math:`P` is a preconditioner and :math:`x_t'` is an estimation of the image :math:`x`.
    By default, :math:`P` is defined as :math:`A^\top`, :math:`x_t' = x_t` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.

    :param deepinv.optim.Distance d: Distance metric to use for the data fidelity term. Default to :class:`deepinv.optim.L2Distance`.
    :param float weight: Weighting factor for the data fidelity term. Default to 1.
    """

    def __init__(self, d: Distance = None, weight=1.0, *args, **kwargs):
        super().__init__()
        if d is not None:
            self.d = Distance(d)
        else:
            self.d = dinv.optim.L2Distance()
        self.weight = weight

    def precond(
        self, u: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        The preconditioner :math:`P` for the data fidelity term. Default to :math:`A^{\top}`.

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.

        :return: (torch.Tensor) preconditionned tensor :math:`P(u)`.
        """
        return (
            physics.A_adjoint(u)
            if isinstance(physics, dinv.physics.LinearPhysics)
            else physics.A_dagger(u)
        )

    def diff(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the difference :math:`A(x) - y` between the forward operator applied to the current iterate and the input data.


        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics), physics=physics)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y) * self.weight


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    Diffusion posterior sampling data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in :cite:`chung2022diffusion`.

    .. math::
            \nabla_x \log p_t(y|x) = \nabla_x \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|

    where :math:`\sigma = \sigma(t)` is the noise level, :math:`m` is the number of measurements (size of :math:`y`),
    and :math:`\lambda` controls the strength of the approximation.

    .. seealso::
        This class can be used for building custom DPS-based diffusion models.
        A self-contained implementation of the original DPS algorithm can be find in :class:`deepinv.sampling.DPS`.

    :param deepinv.models.Denoiser denoiser: Denoiser network
    :param float weight: Weighting factor for the data fidelity term. Default to 100.
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight=1.0,
        clip: tuple = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        if clip is not None:
            assert len(clip) == 2
            clip = sorted(clip)
        self.clip = clip
        self.weight = weight

    def precond(
        self, x: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :param float sigma: Standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the score. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) score term (and denoised output if `get_model_outputs` is `True`).
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            out = self.forward(
                x,
                y,
                physics,
                sigma,
                *args,
                get_model_outputs=get_model_outputs,
                **kwargs,
            )
            # In case we also want the denoised output
            if get_model_outputs:
                l2_loss = out[0]
            else:
                l2_loss = out

            grad_outputs = torch.ones_like(l2_loss)
        norm_grad = torch.autograd.grad(
            outputs=l2_loss, inputs=x, grad_outputs=grad_outputs
        )[0]
        if get_model_outputs:
            return norm_grad, out[1]
        else:
            return norm_grad

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma,
        *args,
        get_model_outputs=False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Returns the loss term :math:`\frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :param bool get_model_outputs: If `True`, also return the denoised output along with the loss. Default to `False`.

        :return: (:class:`torch.Tensor` or tuple of :class:`torch.Tensor`) loss term (and denoised output if `get_model_outputs` is `True`).
        """

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)

        x0_t = self.denoiser(x.to(torch.float32), sigma, *args, **kwargs)

        if self.clip is not None:
            x0_t = torch.clip(x0_t, self.clip[0], self.clip[1])  # optional

        out = (self.d(physics.A(x0_t), y) * y.numel() / y.size(0)).sqrt() * self.weight

        if get_model_outputs:
            return out, x0_t
        else:
            return out


def _reshape_batch_parameter(
    parameter: torch.Tensor | float, reference: torch.Tensor
) -> torch.Tensor:
    """Reshape a scalar or batch-wise parameter for tensor broadcasting."""
    dtype = reference.real.dtype if reference.is_complex() else reference.dtype
    parameter = torch.as_tensor(parameter, device=reference.device, dtype=dtype)
    if parameter.numel() == 1:
        return parameter.squeeze()
    if (
        parameter.shape[0] == reference.shape[0]
        and all(size == 1 for size in parameter.shape[1:])
        and parameter.ndim < reference.ndim
    ):
        return parameter.view(parameter.shape[0], *([1] * (reference.ndim - 1)))
    return parameter


class PiGDMDataFidelity(NoisyDataFidelity):
    r"""
    Pseudoinverse-guided diffusion model (PiGDM) data-fidelity term.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in :cite:`song2023pseudoinverse`.

    .. math::

       \nabla_{x_t} \log p_t(y|x_t) = \lambda J_D(x_t, \sigma_t)^\top A^\top
        \left(r_t^2 A A^\top + \mathrm{Id}\right)^{-1}
        \left(A D(x_t, \sigma_t) - y\right),

    where :math:`D` is a denoiser, :math:`J_D` is its Jacobian, and
    :math:`r_t^2 = \sigma_t^2 / (1 + \sigma_t^2)`. The inverse is evaluated
    exactly for :class:`deepinv.physics.DecomposablePhysics` operators and
    approximated with conjugate gradient for other linear operators.

    :param deepinv.models.Denoiser denoiser: Denoiser network. It may be left as
        ``None`` when the data fidelity is passed to
        :class:`deepinv.sampling.PosteriorDiffusion`, which supplies its denoiser.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    :param int cg_max_iter: Maximum number of conjugate-gradient iterations.
        Default: ``3``.
    :param float cg_tol: Relative conjugate-gradient tolerance. Default: ``1e-4``.
    :param bool verbose: If ``True``, print conjugate-gradient convergence
        information. Default: ``False``.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight: float = 1.0,
        cg_max_iter: int = 3,
        cg_tol: float = 1e-4,
        verbose: bool = False,
    ):
        super().__init__(weight=weight)
        self.denoiser = denoiser
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.verbose = verbose

    def solve_inverse(
        self,
        physics: Physics,
        u: torch.Tensor,
        r_t2: torch.Tensor | float,
    ) -> torch.Tensor:
        r"""
        Apply :math:`(r_t^2 A A^\top + \mathrm{Id})^{-1}` to ``u``.

        :param deepinv.physics.Physics physics: Linear physics operator.
        :param torch.Tensor u: Tensor in the measurement space.
        :param torch.Tensor, float r_t2: PiGDM covariance parameter
            :math:`r_t^2`.
        :return: Solution in the measurement space.
        """
        if isinstance(physics, dinv.physics.DecomposablePhysics):
            transformed_u = physics.U_adjoint(u)
            r_t2 = _reshape_batch_parameter(r_t2, transformed_u)
            singular_values = physics.mask.to(
                device=transformed_u.device, dtype=transformed_u.dtype
            )
            denominator = 1.0 + r_t2 * singular_values.conj() * singular_values
            return physics.U(transformed_u / denominator)

        r_t2 = _reshape_batch_parameter(r_t2, u)

        def operator(v):
            return v + r_t2 * physics.A_A_adjoint(v)

        return conjugate_gradient(
            operator,
            u,
            max_iter=self.cg_max_iter,
            tol=self.cg_tol,
            verbose=self.verbose,
        )

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the PiGDM data-fidelity gradient.

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Linear physics operator.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.
        :return: PiGDM gradient, with the same shape and dtype as ``x``.
        """
        if not isinstance(physics, dinv.physics.LinearPhysics):
            raise ValueError("PiGDMDataFidelity only supports linear physics.")
        if self.denoiser is None:
            raise ValueError("PiGDMDataFidelity requires a denoiser.")

        input_dtype = x.dtype
        with torch.enable_grad():
            x_denoiser = x.detach().to(torch.float32)
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.to(device=x.device, dtype=torch.float32)
            denoised, denoiser_vjp = torch.func.vjp(
                lambda z: self.denoiser(z, sigma, *args, **kwargs),
                x_denoiser,
            )
            measurement = physics.A(denoised)
            difference = measurement - y.to(
                device=measurement.device, dtype=measurement.dtype
            )
            sigma_t = torch.as_tensor(
                sigma, device=difference.device, dtype=difference.dtype
            )
            r_t2 = sigma_t.square() / (1.0 + sigma_t.square())
            inverse_difference = self.solve_inverse(physics, difference, r_t2)
            adjoint = physics.A_adjoint(inverse_difference).to(denoised.dtype)
            gradient = denoiser_vjp(adjoint)[0]

        return (self.weight * gradient).to(input_dtype)


class MomentMatchingDataFidelity(NoisyDataFidelity):
    r"""
    Moment-matching data-fidelity term for diffusion posterior sampling.

    This corresponds to the :math:`p(y|x_t)` approximation proposed in :cite:`rozet2024learning`.


    .. math::

        \nabla_{x_t} \log p_t(y|x_t) = \lambda J_D(x_t, \sigma_t)^\top A^\top
        \left(A J_D(x_t, \sigma_t)^\top A^\top + \mathrm{Id}\right)^{-1}
        \left(A D(x_t, \sigma_t) - y\right).

    The Jacobian products are evaluated with vector-Jacobian products, without
    materializing the denoiser Jacobian, and the measurement-space system is
    approximated with conjugate gradient.

    .. note::

        Conjugate gradient assumes that the effective moment-matching operator
        is symmetric positive definite, as is expected for an exact MMSE
        denoiser covariance.

    :param deepinv.models.Denoiser denoiser: Denoiser network. It may be left as
        ``None`` when the data fidelity is passed to
        :class:`deepinv.sampling.PosteriorDiffusion`, which supplies its denoiser.
    :param float weight: Weighting factor :math:`\lambda`. Default: ``1.0``.
    :param int cg_max_iter: Maximum number of conjugate-gradient iterations.
        Default: ``3``.
    :param float cg_tol: Relative conjugate-gradient tolerance. Default: ``1e-4``.
    :param bool verbose: If ``True``, print conjugate-gradient convergence
        information. Default: ``False``.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight: float = 1.0,
        cg_max_iter: int = 3,
        cg_tol: float = 1e-4,
        verbose: bool = False,
    ):
        super().__init__(weight=weight)
        self.denoiser = denoiser
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.verbose = verbose

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        sigma: torch.Tensor | float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the moment-matching data-fidelity gradient.

        :param torch.Tensor x: Current noisy iterate.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Linear physics operator.
        :param torch.Tensor, float sigma: Diffusion noise standard deviation.
        :return: Moment-matching gradient, with the same shape and dtype as
            ``x``.
        """
        if not isinstance(physics, dinv.physics.LinearPhysics):
            raise ValueError("MomentMatchingDataFidelity only supports linear physics.")
        if self.denoiser is None:
            raise ValueError("MomentMatchingDataFidelity requires a denoiser.")

        input_dtype = x.dtype
        with torch.enable_grad():
            x_denoiser = x.detach().to(torch.float32)
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.to(device=x.device, dtype=torch.float32)
            denoised, denoiser_vjp = torch.func.vjp(
                lambda z: self.denoiser(z, sigma, *args, **kwargs),
                x_denoiser,
            )
            measurement = physics.A(denoised)
            difference = measurement - y.to(
                device=measurement.device, dtype=measurement.dtype
            )

            def operator(v):
                adjoint = physics.A_adjoint(v).to(denoised.dtype)
                covariance_product = denoiser_vjp(adjoint)[0]
                return v + physics.A(covariance_product)

            inverse_difference = conjugate_gradient(
                operator,
                difference,
                max_iter=self.cg_max_iter,
                tol=self.cg_tol,
                verbose=self.verbose,
            )
            adjoint = physics.A_adjoint(inverse_difference).to(denoised.dtype)
            gradient = denoiser_vjp(adjoint)[0]

        return (self.weight * gradient).to(input_dtype)
