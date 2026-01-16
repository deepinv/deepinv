import deepinv as dinv
import scipy.special as sp
from utils import compute_spectral_norm
import torch
from warnings import warn
from deepinv.optim.utils import least_squares

class Scattering(dinv.physics.Physics):
    def __init__(self,
                 img_width,
                 receivers,
                 transmitters,
                 solver,
                 wavenumbers=10,
                 box_length=1.,
                 wave_type='circular_wave',
                 device='cpu',
                 dtype=torch.complex128,
                 verbose=False,
                 normalize=True,
                 save_total_field=True,
                 no_grad_solver=False,
                 tol=0.001,
                 ):
        r"""
        Inverse scattering physics model in 2 dimensions.

        For each of the :math:`i=1,\dots,N_t` transmitters, the forward model is given by

        .. math::

            \begin{align*}
                u_i &= v_i + g * (x \circ v_i)
                y_i &= G_s (x \circ u_i)
            \end{align*}

        where :math:`u` is the scattered field, :math:`v` is the incident field,
        :math:`g(r) = H(\|r\|^2)` is Green's function in 2D,
        :math:`y \in \mathbb{C}^{N_r}` are the measurements at the receivers for the :math:`i`th transmitter,
        and :math:`x` is the unknown scattering potential to be recovered.

        All spatial quantities are discretized in a square grid of size :math:`N \times N` img_width, which
        is assumed to cover a square domain of size :math:`L \times L` in space, where :math:`L` is the box length.

        .. note::

            Letting B=batch, T=transmitters, R=receivers, H, W = img_width
            The sizes of tensors are u = (B, T, H, W), g = (R, H, W),
            y = (B, T, R), x = (B, 1, H, W), v = (1, T, H, W)

        :param int img_width: Number of img_width per image side (`H=W`).
        :param torch.Tensor receivers: Tensor of shape `(2, R)` with receiver x/y positions.
        :param torch.Tensor transmitters: Tensor of shape `(2, T)` with transmitter x/y positions.
        :param callable solver: Helmholtz equation solver callable with signature solver(k2, s, **kwargs) -> total_field (:class:`torch.Tensor`).
        :param Union[int, torch.Tensor] wavenumbers: Scalar or 1D tensor of wavenumbers (real or complex).
        :param float box_length: Physical side length of the square domain.
        :param str wave_type: 'circular' or 'plane'.
        :param str device: Torch device string, e.g. 'cpu' or 'cuda'.
        :param torch.dtype dtype: Torch `dtype` for tensors (e.g. `torch.complex128`).
        :param bool verbose: Enable verbose/debug printing.
        :param bool normalize: If `True`, normalize the linear operator on init.
        :param bool save_total_field: If `True`, store computed total field in the object.
        :param bool no_grad_solver: If `True`, run the solver inside torch.no_grad().
        :param float tol: Tolerance used internally by the physics/solver.
        """

        super(Scattering, self).__init__()
        assert wave_type in ['circular_wave',
                             'plane_wave'], 'Wave type not recognized, options are "circular_wave" or "plane_wave"'

        # store a single scalar wavenumber (no wavenumber dimension)
        if not isinstance(wavenumbers, torch.Tensor):
            self.wavenumber = torch.tensor(wavenumbers, device=device, dtype=dtype)
        else:
            self.wavenumber = wavenumbers.to(device, dtype=dtype).reshape(())

        assert (2 * box_length * self.wavenumber.real / (2 * torch.pi)) < img_width, \
            "The number of img_width is not enough to sample the largest wavenumber. Increase the number of img_width or decrease the wavenumber."


        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.tol = tol

        self.GS_norm = 1.
        self.solver = solver
        self.solver.verbose = verbose

        self.no_grad_solver = no_grad_solver
        self.receivers = receivers.to(device).to(dtype)
        self.transmitters = transmitters.to(device).to(dtype)
        self.box_length = box_length
        self.pixel_area = (box_length / img_width) ** 2
        self.wave_type = wave_type
        image_domain = torch.linspace(-box_length / 2, box_length / 2, img_width, device=device, dtype=dtype)
        self.x_domain, self.y_domain = torch.meshgrid(image_domain, -image_domain)
        self.img_width = img_width
        self.save_total_field = save_total_field

        # incident field
        self.total_field = None
        self.img_width = img_width
        self.update_parameters(receivers=self.receivers, transmitters=self.transmitters)

        if normalize:
            x = torch.ones((1, 1, self.img_width, self.img_width), device=self.device, dtype=self.dtype)
            self.linear_op.update_total_field(self.incident_field)  # compute the total field for the linear operator
            norm = self.linear_op.compute_norm(x)  # compute the spectral norm of the linear operator
            self.GS_norm = 1 / norm.abs().sqrt()
            self.linear_op.GS *= self.GS_norm  # normalize the linear operator

    def compute_norm(self, x):
        r"""
        Computes the spectral norm of the overall forward operator.

        :param torch.Tensor x: Dummy input tensor of shape (B, 1, H, W).
        :param dtype dtype: torch.dtype used for internal computations
        """
        return compute_spectral_norm(self, x, verbose=self.verbose)

    def normalize(self, x):
        """
        Normalize the incident field and noise model by the operator norm.

        :param dtype dtype: torch.dtype used for internal computations
        """
        norm = self.compute_norm(x).sqrt()
        self.incident_field /= norm
        if hasattr(self.noise_model, "sigma"):
            self.noise_model.sigma /= norm

    def update_parameters(self, receivers=None, transmitters=None, **kwargs):
        """
        Update transmitter and receiver parameters and recompute dependent fields/operators.

        :param dtype dtype: torch.dtype used for internal tensors
        """
        if transmitters is not None:
            self.transmitters = transmitters.to(self.device).to(self.dtype)
            # recompute the incident field
            self.generate_incident_field()

        if receivers is not None:
            self.receivers = receivers.to(self.device).to(self.dtype)
            # recompute the output linear operator
            Gs = self.compute_Gs(normalize=False).to(self.device).to(self.dtype)
            Gs *= self.GS_norm  # normalize the Green's function operator
            self.linear_op = BornOperator(Gs, self.incident_field, verbose=self.verbose)

    def set_solver(self, solver):
        """
        Set the solver used to compute the total field.

        :param dtype dtype: torch.dtype used for solver inputs/outputs
        """
        self.solver = solver

    def generate_incident_field(self):
        """
        Generate incident fields on the image grid and at receiver positions.

        Produces:
            - self.incident_field of shape (1, T, H, W)
            - self.incident_field_receivers of shape (1, T, R)

        :param dtype dtype: torch.dtype used for the generated fields
        """
        x_domain = self.x_domain.flatten()
        y_domain = self.y_domain.flatten()
        x_transmitters, y_transmitters = self.transmitters[0, :], self.transmitters[1, :]

        x_receivers, y_receivers = self.receivers[0, :], self.receivers[1, :]

        if self.wave_type == "plane_wave":  # u = exp(i*k*x) = exp(i*2*pi*(cos(theta)*x + sin(theta)*y))
            # to angles
            transmitter_angles = torch.atan2(y_transmitters.real, x_transmitters.real)
            k = self.wavenumber
            wave_number_x = (k * torch.cos(transmitter_angles))  # (T,)
            wave_number_y = (k * torch.sin(transmitter_angles))  # (T,)
            aux = wave_number_x.unsqueeze(1) * x_domain.unsqueeze(0) + wave_number_y.unsqueeze(1) * y_domain.unsqueeze(0)  # (T, H*W)
            incident_field = torch.exp(1j * aux)
            incident_field = incident_field.reshape(1, incident_field.size(0), self.img_width, self.img_width)  # (1, T, H, W)

            # compute incident field at receivers (T, R)
            incident_field_receivers = torch.exp(1j * (wave_number_x.unsqueeze(1) * x_receivers.unsqueeze(0) +
                                                       wave_number_y.unsqueeze(1) * y_receivers.unsqueeze(0)))
            self.incident_field_receivers = incident_field_receivers.reshape(1, incident_field_receivers.size(0),
                                                                             incident_field_receivers.size(1))  # (1, T, R)

        else:  # circular_wave
            transmitter_x, circle_x = torch.meshgrid(x_transmitters, x_domain)
            transmitter_y, circle_y = torch.meshgrid(y_transmitters, y_domain)
            dist_transmitter_circles = torch.sqrt((circle_x - transmitter_x) ** 2 + (circle_y - transmitter_y) ** 2)

            # multiply distances by scalar wavenumber
            dist_transmitter_circles = dist_transmitter_circles * self.wavenumber  # (T, H*W)
            incident_field = green_function(dist_transmitter_circles)
            incident_field = incident_field.reshape(1, incident_field.size(0), self.img_width, self.img_width)  # (1, T, H, W)

            # compute incident field at receivers: make (T, R)
            tx_x, rx_x = torch.meshgrid(x_transmitters, x_receivers)
            tx_y, rx_y = torch.meshgrid(y_transmitters, y_receivers)
            dist_tr = torch.sqrt((tx_x - rx_x) ** 2 + (tx_y - rx_y) ** 2)  # (T, R)
            dist_tr = dist_tr * self.wavenumber
            incident_field_receivers = green_function(dist_tr)  # (T, R)
            self.incident_field_receivers = incident_field_receivers.reshape(1, incident_field_receivers.size(0),
                                                                             incident_field_receivers.size(1))  # (1, T, R)

        self.incident_field = incident_field.to(self.device).to(self.dtype)
        norm = self.incident_field.abs().max()
        self.incident_field /= norm

        self.incident_field_receivers = self.incident_field_receivers.to(self.device).to(self.dtype)
        self.incident_field_receivers /= norm

        print('incident_field_receivers', self.incident_field_receivers.shape)

    def compute_Gs(self, normalize=False):
        """
        Compute Green's function sampled at receiver positions and grid points.

        Returns a tensor of shape (R, H, W).

        :param bool normalize: If True apply the normalization scaling for operator Gs.
        :param dtype dtype: torch.dtype used for the returned Gs
        """
        x_domain = self.x_domain.flatten()
        y_domain = self.y_domain.flatten()

        x_circles, x_receivers = torch.meshgrid(x_domain, self.receivers[0, :])
        y_circles, y_receivers = torch.meshgrid(y_domain, self.receivers[1, :])
        dist_receivers_circles = torch.sqrt((x_circles - x_receivers) ** 2 + (y_circles - y_receivers) ** 2)
        dist_receivers_circles = dist_receivers_circles.T  # (R, H*W)
        # scalar wavenumber
        k = self.wavenumber
        Gs = green_function(dist_receivers_circles * k)  # (R, H*W)
        Gs = Gs.reshape(self.receivers.size(1), self.img_width, self.img_width)  # (R, H, W)

        if normalize:
            Gs *= k
        else:
            Gs *= k ** 2

        Gs *= self.pixel_area
        return Gs

    def compute_total_field(self, x, **kwargs):
        """
        Solve for the total field given scattering potential x using the provided solver.

        :param torch.Tensor x: Scattering potential tensor of shape (B,1,H,W).
        :param dtype dtype: torch.dtype used for internal and output fields
        """
        # This solves the following linear problem: (I - G_d*diag(x))*Et = Ei

        if x.abs().sum() <= 1e-5:
            warn('The input x is zero')

        # scalar wavenumber -> broadcast with x and incident_field
        k02 = (self.wavenumber ** 2)
        # x.unsqueeze(1) -> add transmitter dimension (B,1,H,W)
        k2 = k02 * (x.unsqueeze(1) + 1)
        # s has shape (B, T, H, W) via broadcasting with incident_field (1, T, H, W)
        s = (k2 - k02) * self.incident_field
        return self.solver(k2, s) + self.incident_field

    def set_verbose(self, verbose):
        """
        Toggle verbosity.

        :param dtype dtype: torch.dtype used in logs/operations (informational)
        """
        self.verbose = verbose
        self.linear_op.verbose = verbose
        self.solver.set_verbose(verbose)

    def compute_field_out(self, x, total_field):
        """
        Compute sensor outputs y = G_s * diag(x) * Et.

        :param torch.Tensor x: Scattering potential (B,1,H,W).
        :param torch.Tensor total_field: Total field Et (B,T,H,W) or (1,T,H,W).
        :param dtype dtype: torch.dtype used in computations
        """
        # This computes y = G_s*diag(x)*Et
        self.linear_op.update_total_field(total_field)
        return self.linear_op.A(x)

    def A(self, x, receivers=None, transmitters=None, **kwargs):
        """
        Forward operator wrapper: updates parameters, solves for total field and returns measurements.

        :param torch.Tensor x: Scattering potential (B,1,H,W).
        :param dtype dtype: torch.dtype used for inputs/outputs
        """
        self.update_parameters(receivers, transmitters, **kwargs)

        if self.no_grad_solver:
            with torch.no_grad():
                total_field = self.compute_total_field(x)  # ,**dict)
                total_field = total_field.detach()
        else:
            total_field = self.compute_total_field(x)

        out = self.compute_field_out(x, total_field)
        return out

    def A_dagger(self, y, linear=False, x_init=None, max_iter=2,
                  use_init=True, rel_tol=1e-3, **kwargs):
        """
        Pseudo-inverse / adjoint-based reconstruction via alternating optimization.

        :param torch.Tensor y: Measurements tensor (B,T,R).
        :param dtype dtype: torch.dtype used for internal computations and returned x
        """
        if linear:
            max_iter = 1

        if x_init is not None:
            x = x_init
        else:
            x = torch.ones((y.shape[0], 1, self.img_width, self.img_width), dtype=y.dtype, device=y.device) * .05

        flag = True
        if use_init:
            total_field = self.incident_field
        else:
            total_field = None

        for i in range(max_iter):
            prev_x = x.clone()

            if linear:
                total_field = self.incident_field
            else:
                total_field = self.compute_total_field(x, init=total_field if use_init else None)

            self.linear_op.update_total_field(total_field)
            x = self.linear_op.A_dagger(y, init=x if use_init else None)
            rel_err = (x - prev_x).abs().pow(2).mean() / prev_x.abs().pow(2).mean()
            if rel_err < rel_tol:
                flag = False
                if self.verbose:
                    print('Alternated optimization pseudo inverse converged at iteration', i)
                break

        if flag and not linear and max_iter != 2:
            if self.verbose:
                print('Alternated optimization pseudo inverse did not converge')

        return x

    def subsample_transmitters(self, transmitters, y):
        """
        Subsample transmitters and corresponding measurements.

        :param torch.Tensor transmitters: Indices of transmitters to keep.
        :param torch.Tensor y: Measurement tensor (B,T,R,...).
        :param dtype dtype: torch.dtype used in returned physics and data
        """
        physics1 = self.clone()
        physics1.update(transmitters=self.transmitters[:, transmitters])
        y1 = y[:, transmitters, ...]
        return y1, physics1


class BornOperator(dinv.physics.LinearPhysics):
    def __init__(self, GS, total_field, verbose=False):
        super(BornOperator, self).__init__()
        self.total_field = total_field
        self.verbose = verbose
        self.GS = GS
        # self.total_field = torch.ones_like(total_field, dtype=total_field.dtype, device=total_field.device)

    def update_total_field(self, total_field):
        """
        Update the stored total field used by the linear operator.

        :param torch.Tensor total_field: New total field tensor (1,T,H,W) or (B,T,H,W).
        :param dtype dtype: torch.dtype used for total_field
        """
        self.total_field = total_field
        # self.total_field = torch.ones_like(total_field, dtype=total_field.dtype, device=total_field.device)

    def A(self, x):  # Gs has size (R, H, W)
        """
        Linear forward operation: y = Gs * (x * total_field).

        :param torch.Tensor x: Scattering potential (B,1,H,W).
        :param dtype dtype: torch.dtype used for returned measurements
        """
        # x: (B,1,H,W), total_field: (1,T,H,W) -> aux (B,T,H,W)
        aux = x * self.total_field
        # GS: (R, H, W) -> result y: (B, T, R)
        y = torch.einsum('bthw, rhw->btr', aux, self.GS)
        return y

    def A_adjoint(self, y):
        """
        Adjoint operation mapping measurements back to image domain.

        :param torch.Tensor y: Measurements (B,T,R).
        :param dtype dtype: torch.dtype used for returned image estimate
        """
        # y: (B, T, R), GS: (R, H, W) -> aux: (B, T, H, W)
        aux = torch.einsum('btr, rhw->bthw', y, self.GS.conj())
        x = self.total_field.conj() * aux  # (B, T, H, W)
        x = x.sum(1, keepdim=True)  # (B,1,H,W)
        return x

    def A_dagger(self, y, init=None, solver='lsqr', gamma=1e3):
        """
        Solve least-squares for x given y using the operator A and its adjoint.

        :param torch.Tensor y: Measurements (B,T,R).
        :param dtype dtype: torch.dtype used for solver internals and returned x
        """
        # dim = [i for i in range(y.dim()) if i > 0]
        x = least_squares(A=self.A, AT=self.A_adjoint, gamma=gamma, y=y, solver=solver, dim=[0],
                          init=init, max_iter=100, tol=5e-3, verbose=self.verbose)
        return x



class HelmholtzSolver(torch.nn.Module):
    def __init__(self, device='cpu', dtype=torch.complex128):
        r"""
        Base class for solving Helmholtz equations.

        The Helmholtz equation is of the form:

        .. math::

            \nabla^2 u(r) + k^2(r) u(r) = -s(r)

        where :math:`r` is the spatial coordinate, :math:`k` is the (space-varying) wavenumber,
        :math:`u` is the field to be solved, and :math:`s` is the source term.

        :param device:
        :param dtype:
        """
        super(HelmholtzSolver, self).__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, k2, source, init=None, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")


class LinearSolver(HelmholtzSolver):
    """
    Base class for linear Helmholtz solvers.
    This class is intended to be subclassed for specific linear solver implementations.
    """

    def __init__(self, scattered_field=None, device='cpu', dtype=torch.complex128):
        super(LinearSolver, self).__init__(device=device, dtype=dtype)
        self.scattered_field = scattered_field

    def set_verbose(self, verbose):
        """
        Set the verbosity of the solver.
        This method should be implemented in subclasses if needed.
        """
        pass

    def forward(self, k2, source, init=None, **kwargs):
        """
        Forward method to solve the Helmholtz equation using a linear solver.
        This method should be implemented in subclasses.
        """
        return torch.zeros_like(source) if self.scattered_field is None else self.scattered_field


class LippmanSchwingerFunction(torch.autograd.Function):
    """
    Implements the Lippmann-Schwinger forward solver and its analytical
    adjoint-state method for the backward pass.
    """

    @staticmethod
    def forward(ctx, k2, source, init, g_fourier, k02, solver_params):
        """
        Solves for the scattered field using an iterative solver.
        """
        m = k2 - k02  # Scattering potential

        # Define the forward operator A(u) = u - G * (m * u)
        A = lambda u: u - apply_filter(m * u, g_fourier)

        # Define its adjoint AT(u) = u - m.conj() * G.conj()(u)
        AT = lambda u: u - m.conj() * apply_filter(u, g_fourier.conj())

        # The right-hand side is the incident field u_i = G * source
        b_incident_field = apply_filter(source, g_fourier)

        # Solve A(u_s) = b for the scattered field u_s
        scattered_field = least_squares(A=A, AT=AT, y=b_incident_field, init=init, **solver_params)

        # Save necessary tensors for the backward pass
        ctx.solver_params = solver_params
        ctx.save_for_backward(m, scattered_field, b_incident_field, g_fourier)

        return scattered_field


    @staticmethod
    def backward(ctx, grad_scattered_field):
        """
        Computes the gradient of the loss with respect to k2 and source
        using the pre-derived closed-form Frechet derivative.
        """
        # Unpack saved tensors and parameters
        solver_params = ctx.solver_params
        m, scattered_field, b_incident_field, g_fourier = ctx.saved_tensors

        # Initialize gradients to None
        grad_k2 = grad_source = grad_init = None

        # Only proceed if gradients for k2 or source are required
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None, None, None

        ### Adjoint-State Calculation

        # Define the forward operator A(u) = u - G * (m * u)
        A = lambda u: u - apply_filter(m * u, g_fourier)

        # Define its adjoint AT(u) = u - m.conj() * G.conj()(u)
        AT = lambda u: u - m.conj() * apply_filter(u, g_fourier.conj())

        # 1. Adjoint Solve: Solve AT(v) = grad_output for the adjoint field 'v'
        init_adjoint = scattered_field.conj().clone()

        v = least_squares(A=AT, AT=A, y=grad_scattered_field, init=init_adjoint, **solver_params)

        g_adj_v = apply_filter(v, g_fourier.conj())
        # 2. Gradient Calculation for k2
        if ctx.needs_input_grad[0]:
            grad_k2 = scattered_field.conj() * g_adj_v

        # 3. Gradient Calculation for source
        if ctx.needs_input_grad[1]:
            # grad_source = G_adj(v)
            grad_source = g_adj_v

        # Return gradients in the same order as inputs to forward()
        return grad_k2, grad_source, grad_init, None, None, None

class LippmanSchwingerSolver(HelmholtzSolver):
    def __init__(self, pixels, box_length, wavenumbers, device='cpu', dtype=torch.complex128, min_iter=1,
                 max_iter=500, verbose=False, ls_solver='lsqr', stop_crit=1e-5, adjoint_state=True):
        super(LippmanSchwingerSolver, self).__init__(device=device, dtype=dtype)

        # Pre-compute Green's function in Fourier space and move to device
        self.e = 0.01
        _, self.g_fourier = green_fourier(pixels, box_length, (wavenumbers.pow(2)+1j*self.e).sqrt(), vico=True)
        self.k02 = wavenumbers.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).pow(2).to(device)
        self.g_fourier = self.g_fourier.to(device).to(dtype)
        self.adjoint_state = adjoint_state

        # Store solver parameters in a dictionary for easy passing
        self.solver_params = {
            'min_iter': min_iter,
            'max_iter': max_iter,
            'verbose': verbose,
            'solver': ls_solver,
            'tol': stop_crit
        }

    def set_verbose(self, verbose):
        """
        Set the verbosity of the solver.
        """
        self.solver_params['verbose'] = verbose

    def forward(self, k2, source, init=None, **kwargs):
        """
        Calls the custom autograd function to solve the equation.
        PyTorch will automatically use the defined .backward() method during backpropagation.
        """
        # sizes total_field = (B, T, F, H, W), green_function = (1, F, H, W), y = (B, R, F), x = (B, 1, H, W), incident_field = (T, R, H, W)
        # Pack non-tensor parameters and call the custom function
        if self.adjoint_state:
            # Use the Lippmann-Schwinger function with adjoint state
            return LippmanSchwingerFunction.apply(k2, source, init, self.g_fourier,
                                                  self.k02, self.solver_params)
        else:
            m = k2 - self.k02  # Scattering potential

            # Define the forward operator A(u) = u - G * (m * u)
            A = lambda u: u - apply_filter(m * u, self.g_fourier)

            # Define its adjoint AT(u) = u - m.conj() * G.conj()(u)
            AT = lambda u: u - m.conj() * apply_filter(u, self.g_fourier.conj())

            # The right-hand side is the incident field u_i = G * source
            b_incident_field = apply_filter(source, self.g_fourier)

            scattered_field = least_squares(A=A, AT=AT, y=b_incident_field, init=init, **self.solver_params)

            return scattered_field

def random_sensors(number, radius, max_angle=360, offset_angle=0, device='cpu'):
    r"""
    Generates random sensors in a circular pattern.

    :param number: Number of sensors
    :param radius: Radius of the circle
    :param max_angle: Maximum angle for the sensors
    :param offset_angle: Offset angle for the sensors
    :param device: Device to place the tensors on
    :param dtype dtype: torch.dtype used for output tensor values (inferred from device)
    :return: Tensor of shape (2, number) with x and y positions of the sensors
    """
    receiver_angles = torch.rand(number, device=device) * (max_angle / 360.) * 2 * torch.pi + offset_angle
    receiver_radii = radius * torch.ones_like(receiver_angles, device=device)
    x_pos = receiver_radii * torch.cos(receiver_angles)
    y_pos = receiver_radii * torch.sin(receiver_angles)
    return torch.stack([x_pos, y_pos])  # 2, number


def circular_sensors(number, radius, max_angle=360, offset_angle=0, device='cpu'):
    """
    Generate equispaced sensors on a circle.

    :param int number: Number of sensors.
    :param float radius: Radius of the circle.
    :param float max_angle: Maximum angle in degrees covered by sensors.
    :param float offset_angle: Offset angle (radians).
    :param str device: Torch device for tensors.
    :param dtype dtype: torch.dtype used for output tensors (inferred from device)
    :return: Tensor of shape (2, number) with x and y positions.
    """
    receiver_angles = torch.linspace(0, max_angle / 360 * 2 * torch.pi, number + 1, device=device)[:-1] + offset_angle
    receiver_radii = radius * torch.ones_like(receiver_angles)
    x_pos = receiver_radii * torch.cos(receiver_angles)
    y_pos = receiver_radii * torch.sin(receiver_angles)
    return torch.stack([x_pos, y_pos])  # 2, number



def hankel1(n, x):
    """
    Wrapper for scipy.special.hankel1 that preserves device and dtype.

    :param int n: Order of the Hankel function.
    :param torch.Tensor x: Input tensor.
    :param dtype dtype: torch.dtype used for the returned tensor (matches x.dtype)
    :return: Tensor with hankel1 applied elementwise, on same device/dtype as x.
    """
    device = x.device
    dtype = x.dtype
    out = sp.hankel1(n, x.to("cpu")).to(device=device, dtype=dtype)
    return out


def jv(n, x):
    """
    Wrapper for scipy.special.jv that preserves device and dtype.

    :param int n: Order of the Bessel function.
    :param torch.Tensor x: Input tensor.
    :param dtype dtype: torch.dtype used for the returned tensor (matches x.dtype)
    :return: Tensor with jv applied elementwise, on same device/dtype as x.
    """
    device = x.device
    dtype = x.dtype
    out = sp.jv(n, x.to("cpu")).to(device=device, dtype=dtype)
    return out


def green_function(r, remove_nans=False):
    """
    Green's function in 2D based on Hankel function H_0^{(1)}.

    :param torch.Tensor r: Radial argument(s) (can be tensor).
    :param bool remove_nans: If True replace NaNs (singularity) with max abs value.
    :param dtype dtype: torch.dtype used for the returned tensor (matches r.dtype)
    :return: Complex tensor with Green's function values.
    """
    out = 1j / 4 * hankel1(0, r)
    if remove_nans:
        out[torch.isnan(out)] = out.abs().max()  # singularity at 0
    return out


def green_fourier(img_width, box_length, wavenumbers, vico=True):
    r"""

     proper green function discretisaton from "Fast convolution with free-space Green’s functions"

    :param int img_width: image width H=W
    :param float box_length: physical box length
    :param torch.Tensor or scalar wavenumbers: scalar wavenumber expected
    :param bool vico: whether to use Vico's correction
    :param dtype dtype: torch.dtype used for intermediate and returned tensors
    """
    n = 4 * img_width
    aux = torch.fft.fftfreq(n, d=4 * box_length / n).to(wavenumbers.device)

    fx_domain = aux.unsqueeze(1)
    fy_domain = aux.unsqueeze(0)
    s = torch.sqrt(fx_domain ** 2 + fy_domain ** 2) * 2 * torch.pi  # (n, n)
    k = wavenumbers  # scalar

    filterf = 1.0 + 0j
    if vico:
        L = 1.5 * box_length  # for d=2
        constant = 1j * torch.pi * L / 2
        filterf = filterf + constant * s * jv(1, L * s) * hankel1(0, L * k)
        filterf = filterf - constant * k * jv(0, L * s) * hankel1(1, L * k)

    filterf = filterf / (s ** 2 - k ** 2)
    filterf = filterf / 2

    if torch.isnan(filterf).any():
        print('nan')

    # to (1, H, W)
    filter = torch.fft.ifft2(filterf, norm='ortho')
    filter = torch.fft.fftshift(filter)[..., img_width:3 * img_width, img_width:3 * img_width]
    filter = torch.fft.ifftshift(filter)
    filter = filter.unsqueeze(0)

    filterf = torch.fft.fft2(filter, norm='ortho')

    return filter, filterf



def apply_filter(field, filterf, factor=2, padding_mode='constant'):
    """
    Apply a Fourier-domain filter to a field with optional factor padding.

    :param torch.Tensor field: Input field with shape (..., H, W).
    :param torch.Tensor filterf: Fourier filter of shape (1, H*factor, W*factor) or broadcastable.
    :param int factor: Oversampling / padding factor.
    :param str padding_mode: Padding mode used before FFT.
    :param dtype dtype: torch.dtype used for returned field (matches input)
    :return: Filtered field cropped to original size.
    """
    H, W = field.shape[-2], field.shape[-1]
    H2 = H * (factor - 1) // 2
    W2 = W * (factor - 1) // 2
    # pad input
    field = torch.nn.functional.pad(field, (W2, W2, H2, H2), mode=padding_mode, value=0)

    # filter in fourier space
    xf = torch.fft.fft2(field, norm='ortho')
    yf = xf * filterf.unsqueeze(0)  # .conj()
    y = torch.fft.ifft2(yf, norm='ortho')
    # crop output
    y = y[..., H2:H2 + H, W2:W2 + W]
    return y
