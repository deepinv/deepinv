from __future__ import annotations
import torch
from torch import Tensor
from deepinv.physics.forward import LinearPhysics
import deepinv.physics.functional as dF


class Reverberation(LinearPhysics):
    r"""
    Reverberation operator for single-channel dereverberation problems.

    This forward operator convolves a dry (anechoic) audio signal :math:`x` with a
    room impulse response (RIR) :math:`h`, in order to simulate the reverberation
    incurred by recording the signal in a room:

    .. math::

        y = h * x

    where :math:`*` denotes a *causal* 1D convolution, i.e.

    .. math::

        y[t] = \sum_{k=0}^{K-1} h[k]\, x[t-k], \qquad x[t] = 0 \text{ for } t < 0,

    so that the output has the same length (number of samples) as the input; any
    part of the reverberation tail extending past the recording length is discarded,
    following common practice in the dereverberation literature, see e.g.
    :footcite:t:`lemercier2023diffusion`.

    RIRs :math:`h` can be simulated using the room-acoustics simulator
    `pyroomacoustics <https://github.com/LCAV/pyroomacoustics>`_ via
    :class:`deepinv.physics.generator.RIRGenerator`, or loaded from a dataset of
    measured RIRs.

    :param torch.Tensor filter: RIR :math:`h`. Tensor of size ``(b, c, K)`` where
        ``b`` can be either ``1`` or the batch size, and ``c`` can be either ``1`` or
        the number of channels of the signal, e.g. generated with
        :class:`deepinv.physics.generator.RIRGenerator`.
    :param torch.device, str device: Device on which the physics' buffers will be
        created. If a buffer is updated via ``physics.update_parameters()``, if not
        ``None``, it will be automatically casted to the device of the replaced
        buffer, else, use the device of the provided value. To change the device of
        all buffers, please use ``physics.to(device)``.

    .. note::

        This class makes it possible to change the RIR at runtime by passing a new
        filter to the forward method, e.g. ``y = physics(x, filter=h)``. The new
        filter :math:`h` is then stored as the current filter.

    |sep|

    :Examples:

        Reverberation operator with a toy 2-tap RIR (a direct path followed by a
        single reflection) applied to a short signal with a single unit impulse:

        >>> import torch
        >>> from deepinv.physics import Reverberation
        >>> x = torch.zeros((1, 1, 8))  # single-channel signal with 8 samples
        >>> x[:, :, 0] = 1
        >>> h = torch.tensor([[[1.0, 0.5]]])  # direct path + one reflection at 0.5 amplitude
        >>> physics = Reverberation(filter=h)
        >>> y = physics(x)
        >>> y
        tensor([[[1.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])

    """

    def __init__(
        self,
        filter: Tensor = None,
        device: torch.device | str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            isinstance(filter, Tensor) or filter is None
        ), f"The filter must be a torch.Tensor or None, got filter of type {type(filter)}."

        self.register_buffer("filter", filter)
        self.to(device)

    def A(self, x: Tensor, filter: Tensor = None, **kwargs) -> Tensor:
        r"""
        Applies the reverberation operator to the input signal.

        :param torch.Tensor x: dry input signal of size ``(B, C, T)``.
        :param torch.Tensor filter: RIR :math:`h` to be applied to the input signal.
            If not ``None``, it uses this filter instead of the one defined in the
            class, and the provided filter is stored as the current filter.
        :raises ValueError: if the input tensor does not have 3 dimensions.
        """
        self.update_parameters(filter=filter, **kwargs)

        if x.dim() != 3:
            raise ValueError(
                f"Expected Tensor dimension to be 3, i.e. (B, C, T), got {x.dim()}"
            )
        return dF.causal_conv1d(x, self.filter)

    def A_adjoint(self, y: Tensor, filter: Tensor = None, **kwargs) -> Tensor:
        r"""
        Adjoint operator of the reverberation operator.

        :param torch.Tensor y: reverberant signal of size ``(B, C, T)``.
        :param torch.Tensor filter: RIR :math:`h` to be applied to the input signal.
            If not ``None``, it uses this filter instead of the one defined in the
            class, and the provided filter is stored as the current filter.
        :raises ValueError: if the input tensor does not have 3 dimensions.
        """
        self.update_parameters(filter=filter, **kwargs)

        if y.dim() != 3:
            raise ValueError(
                f"Expected Tensor dimension to be 3, i.e. (B, C, T), got {y.dim()}"
            )
        return dF.causal_conv1d_adjoint(y, self.filter)
