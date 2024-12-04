from functools import partial
from deepinv.physics.forward import Physics, LinearPhysics
from deepinv.physics.compressed_sensing import CompressedSensing
from deepinv.optim.phase_retrieval import spectral_methods
import torch
import numpy as np


class PhaseRetrieval(Physics):
    r"""
    Phase Retrieval base class corresponding to the operator

    .. math::

        \forw{x} = |Bx|^2.

    The linear operator :math:`B` is defined by a :class:`deepinv.physics.LinearPhysics` object.

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param deepinv.physics.forward.LinearPhysics B: the linear forward operator.
    """

    def __init__(
        self,
        B: LinearPhysics,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "Phase Retrieval"

        self.B = B

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Applies the forward operator to the input x.

        Note here the operation includes the modulus operation.

        :param torch.Tensor x: signal/image.
        """
        return self.B(x, **kwargs).abs().square()

    def A_dagger(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Computes an initial reconstruction for the image :math:`x` from the measurements :math:`y`.

        We use the spectral methods defined in :class:`deepinv.optim.phase_retrieval.spectral_methods` to obtain an initial inverse.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) an initial reconstruction for image :math:`x`.
        """
        return spectral_methods(y, self, **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.A_dagger(y, **kwargs)

    def B_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.B.A_adjoint(y, **kwargs)

    def B_dagger(self, y):
        r"""
        Computes the linear pseudo-inverse of :math:`B`.

        :param torch.Tensor y: measurements.
        :return: (torch.Tensor) the reconstruction image :math:`x`.
        """
        return self.B.A_dagger(y)

    def forward(self, x, **kwargs):
        r"""
        Applies the phase retrieval measurement operator, i.e. :math:`y = \noise{|Bx|^2}` (with noise :math:`N` and/or sensor non-linearities).

        :param torch.Tensor,list[torch.Tensor] x: signal/image
        :return: (torch.Tensor) noisy measurements
        """
        return self.sensor(self.noise(self.A(x, **kwargs)))

    def A_vjp(self, x, v):
        r"""
        Computes the product between a vector :math:`v` and the Jacobian of the forward operator :math:`A` at the input x, defined as:

        .. math::

            A_{vjp}(x, v) = 2 \overline{B}^{\top} \text{diag}(Bx) v.

        :param torch.Tensor x: signal/image.
        :param torch.Tensor v: vector.
        :return: (torch.Tensor) the VJP product between :math:`v` and the Jacobian.
        """
        return 2 * self.B_adjoint(self.B(x) * v)


class RandomPhaseRetrieval(PhaseRetrieval):
    r"""
    Random Phase Retrieval forward operator. Creates a random :math:`m \times n` sampling matrix :math:`B` where :math:`n` is the number of elements of the signal and :math:`m` is the number of measurements.

    This class generates a random i.i.d. Gaussian matrix

    .. math::

        B_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).

    An existing operator can be loaded from a saved .pth file via ``self.load_state_dict(save_path)``, in a similar fashion to :class:`torch.nn.Module`.

    :param int m: number of measurements.
    :param tuple img_shape: shape (C, H, W) of inputs.
    :param bool channelwise: Channels are processed independently using the same random forward operator.
    :param torch.type dtype: Forward matrix is stored as a dtype. Default is torch.cfloat.
    :param str device: Device to store the forward matrix.
    :param torch.Generator (Optional) rng: a pseudorandom random number generator for the parameter generation.
        If ``None``, the default Generator of PyTorch will be used.

    |sep|

    :Examples:

        Random phase retrieval operator with 10 measurements for a 3x3 image:

        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
        >>> physics = RandomPhaseRetrieval(m=10,img_shape=(1, 3, 3), rng=torch.Generator('cpu'))
        >>> physics(x)
        tensor([[2.3043, 1.3553, 0.0087, 1.8518, 1.0845, 1.1561, 0.8668, 2.2031, 0.4542,
                 0.0225]])
    """

    def __init__(
        self,
        m,
        img_shape,
        channelwise=False,
        dtype=torch.cfloat,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        self.m = m
        self.img_shape = img_shape
        self.channelwise = channelwise
        self.dtype = dtype
        self.device = device
        if rng is None:
            self.rng = torch.Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physic generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator on {self.device}."
            self.rng = rng
        self.initial_random_state = self.rng.get_state()

        B = CompressedSensing(
            m=m,
            img_shape=img_shape,
            fast=False,
            channelwise=channelwise,
            dtype=dtype,
            device=device,
            rng=self.rng,
        )
        super().__init__(B)
        self.name = f"RPR_m{self.m}"


class PtychographyLinearOperator(LinearPhysics):
    r"""
    Forward linear operator for phase retrieval in ptychography. Modelling multiple applications of the shifted probe and Fourier transform on an input image.

    This operator performs multiple 2D Fourier transforms on the probe function applied to the shifted input image according to specific offsets, and concatenates them.
    The probe function is applied element by element to the input image.

    .. math::

        B = \left[ \begin{array}{c} B_1 \\ B_2 \\ \vdots \\ B_{n_{\text{img}}} \end{array} \right],
        B_l = F P T_l, \quad l = 1, \dots, n_{\text{img}},

    where :math:`F` is the 2D Fourier transform, :math:`P` is the probe function and :math:`T_l` is a shift.

    :param tuple img_size: Shape of the input image (height, width).
    :param probe: A 2D tensor representing the probe function.
    :param str probe_type: Type of probe (e.g., "disk"), used if `probe` is not provided.
    :param int probe_radius: Radius of the probe, used if `probe` is not provided.
    :param array_like shifts: shifts of the probe.
    :param int fov: Field of view used for calculating shifts if `shifts` is not provided.
    :param int n_img: Number of shifted probe positions (should be a perfect square).
    :param torch.device, str device: Device "cpu" or "gpu".
    """

    def __init__(
        self,
        img_size=None,
        probe=None,
        shifts=None,
        probe_type=None,
        probe_radius=None,  # probe parameters
        fov=None,
        n_img: int = 25,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device

        if probe is not None:
            self.probe = probe
            self.img_size = img_size if img_size is not None else probe.shape
        else:
            self.img_size = img_size
            self.probe_type = probe_type
            self.probe_radius = probe_radius
            self.probe = self.construct_probe(
                type=probe_type, probe_radius=probe_radius
            )

        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            self.fov = fov
            self.shifts = self.generate_shifts(n_img=n_img, fov=fov)

        self.probe = (
            self.probe / self.get_overlap_img(self.probe, self.shifts).mean().sqrt()
        )

    def A(self, x, **kwargs):
        """
        Applies the forward operator to the input image `x` by shifting the probe,
        multiplying element-wise, and performing a 2D Fourier transform.

        :param x: Input image tensor.
        :return: Concatenated Fourier transformed tensors after applying shifted probes.
        """
        op_fft2 = partial(torch.fft.fft2, norm="ortho")
        f = lambda x, x_shift, y_shift: op_fft2(
            self.probe * self.shift(x, x_shift, y_shift)
        )
        return torch.cat(
            [f(x, x_shift, y_shift) for (x_shift, y_shift) in self.shifts], dim=1
        )

    def A_adjoint(self, y, **kwargs):
        """
        Applies the adjoint operator to `y`.

        :param y: Transformed image data tensor of size (batch_size, n_img, height, width).
        :return: Reconstructed image tensor.
        """
        op_ifft2 = partial(torch.fft.ifft2, norm="ortho")
        g = lambda s, x_shift, y_shift: self.shift(
            self.probe * op_ifft2(s), -x_shift, -y_shift
        )
        for i in range(len(self.shifts)):
            if i == 0:
                x = g(y[:, i, :, :].unsqueeze(1), self.shifts[i, 0], self.shifts[i, 1])
            else:
                x += g(y[:, i, :, :].unsqueeze(1), self.shifts[i, 0], self.shifts[i, 1])
        return x

    def construct_probe(self, type="disk", probe_radius=10):
        """
        Constructs the probe based on the specified type and radius.

        :param str type: Type of probe shape, e.g., "disk".
        :param int probe_radius: Radius of the probe shape.
        :return: Tensor representing the constructed probe.
        """
        if type == "disk" or type is None:
            x = torch.arange(self.img_size[0], dtype=torch.float64)
            y = torch.arange(self.img_size[1], dtype=torch.float64)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            probe = torch.zeros(self.img_size, device=self.device)
            probe[
                torch.sqrt(
                    (X - self.img_size[0] // 2) ** 2 + (Y - self.img_size[1] // 2) ** 2
                )
                < probe_radius
            ] = 1
        else:
            raise NotImplementedError(f"Probe type {type} not implemented")
        return probe

    def generate_shifts(self, n_img, fov=None):
        """
        Generates the array of probe shifts across the image, based on probe radius and field of view.

        :param size: Size of the image.
        :param int n_img: Number of shifts (must be a perfect square).
        :param probe_radius: Radius of the probe.
        :param fov: Field of view for shift computation.
        :return np.ndarray: Array of (x, y) shifts.
        """
        if fov is None:
            fov = self.img_size[-1]
        start_shift = -fov // 2
        end_shift = fov // 2

        assert int(np.sqrt(n_img)) ** 2 == n_img, "n_img needs to be a perfect square"
        side_n_img = int(np.sqrt(n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        y_shifts, x_shifts = np.meshgrid(shifts, shifts, indexing="ij")
        return np.concatenate(
            [x_shifts.reshape(n_img, 1), y_shifts.reshape(n_img, 1)], axis=1
        )

    def shift(self, x, x_shift, y_shift, pad_zeros=True):
        """
        Applies a shift to the tensor `x` by `x_shift` and `y_shift`.

        :param x: Input tensor.
        :param x_shift: Shift in x-direction.
        :param y_shift: Shift in y-direction.
        :param pad_zeros: If True, pads shifted regions with zeros.
        :return: Shifted tensor.
        """
        x = torch.roll(x, (x_shift, y_shift), dims=(-2, -1))

        if pad_zeros:
            if x_shift < 0:
                x[..., x_shift:, :] = 0
            elif x_shift > 0:
                x[..., 0:x_shift, :] = 0
            if y_shift < 0:
                x[..., :, y_shift:] = 0
            elif y_shift > 0:
                x[..., :, 0:y_shift] = 0
        return x

    def get_overlap_img(self, probe, shifts):
        """
        Computes the overlapping image intensities from probe shifts, used for normalization.

        :param probe: Probe tensor.
        :param shifts: Array of probe shifts.
        :return: Tensor representing the overlap image.
        """
        overlap_img = torch.zeros_like(probe, dtype=torch.float32)
        for x_shift, y_shift in shifts:
            overlap_img += torch.abs(self.shift(probe, x_shift, y_shift)) ** 2
        return overlap_img


class Ptychography(PhaseRetrieval):
    r"""
    Ptychography forward operator. Corresponding to the operator

    .. math::

         \forw{x} = \left| Bx \right|^2

    where :math:`B` is the linear forward operator defined by a :class:`deepinv.physics.PtychographyLinearOperator` object.

    :param tuple in_shape: Shape of the input image (height, width).
    :param probe: A 2D tensor representing the probe function.
    :param str probe_type: Type of probe (e.g., "disk"), used if `probe` is not provided.
    :param int probe_radius: Radius of the probe, used if `probe` is not provided.
    :param array_like shifts: shifts of the probe.
    :param int fov: Field of view used for calculating shifts if `shifts` is not provided.
    :param int n_img: Number of shifted probe positions (should be a perfect square).
    :param device: Device "cpu" or "gpu".
    """

    def __init__(
        self,
        in_shape=None,
        probe=None,
        shifts=None,
        probe_type=None,
        probe_radius=None,  # probe parameters
        fov=None,
        n_img: int = 25,
        device="cpu",
        **kwargs,
    ):
        B = PtychographyLinearOperator(
            img_size=in_shape,
            probe=probe,
            shifts=shifts,
            probe_type=probe_type,
            probe_radius=probe_radius,
            fov=fov,
            n_img=n_img,
            device=device,
        )
        self.probe = B.probe
        self.shifts = B.shifts
        self.device = device

        super().__init__(B, **kwargs)
        self.name = f"Ptychography_PR"
