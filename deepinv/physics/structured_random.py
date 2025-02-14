import math
from deepinv.physics.functional import dst1
import numpy as np
import torch

from deepinv.physics.forward import LinearPhysics


def compare(input_shape: tuple, output_shape: tuple) -> str:
    r"""
    Compare input and output shape to determine the sampling mode.

    :param tuple input_shape: Input shape (C, H, W).
    :param tuple output_shape: Output shape (C, H, W).

    :return: The sampling mode in ["oversampling","undersampling","equisampling].
    """
    if input_shape[1] == output_shape[1] and input_shape[2] == output_shape[2]:
        return "equisampling"
    elif input_shape[1] <= output_shape[1] and input_shape[2] <= output_shape[2]:
        return "oversampling"
    elif input_shape[1] >= output_shape[1] and input_shape[2] >= output_shape[2]:
        return "undersampling"
    else:
        raise ValueError(
            "Does not support different sampling schemes on height and width."
        )


def padding(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Zero padding function for oversampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (:class:`torch.Tensor`) the zero-padded tensor.
    """
    change_top = math.ceil(abs(input_shape[1] - output_shape[1]) / 2)
    change_bottom = math.floor(abs(input_shape[1] - output_shape[1]) / 2)
    change_left = math.ceil(abs(input_shape[2] - output_shape[2]) / 2)
    change_right = math.floor(abs(input_shape[2] - output_shape[2]) / 2)
    assert change_top + change_bottom == abs(input_shape[1] - output_shape[1])
    assert change_left + change_right == abs(input_shape[2] - output_shape[2])
    return torch.nn.ZeroPad2d((change_left, change_right, change_top, change_bottom))(
        tensor
    )


def trimming(tensor: torch.Tensor, input_shape: tuple, output_shape: tuple):
    r"""
    Trimming function for undersampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple input_shape: shape of the input tensor.
    :param tuple output_shape: shape of the output tensor.

    :return: (:class:`torch.Tensor`) the trimmed tensor.
    """
    change_top = math.ceil(abs(input_shape[1] - output_shape[1]) / 2)
    change_bottom = math.floor(abs(input_shape[1] - output_shape[1]) / 2)
    change_left = math.ceil(abs(input_shape[2] - output_shape[2]) / 2)
    change_right = math.floor(abs(input_shape[2] - output_shape[2]) / 2)
    assert change_top + change_bottom == abs(input_shape[1] - output_shape[1])
    assert change_left + change_right == abs(input_shape[2] - output_shape[2])
    if change_bottom == 0:
        tensor = tensor[..., change_top:, :]
    else:
        tensor = tensor[..., change_top:-change_bottom, :]
    if change_right == 0:
        tensor = tensor[..., change_left:]
    else:
        tensor = tensor[..., change_left:-change_right]
    return tensor


def generate_diagonal(
    shape: tuple,
    mode: str,
    dtype=torch.cfloat,
    device="cpu",
    generator=torch.Generator("cpu"),
):
    r"""
    Generate a random tensor as the diagonal matrix.
    """

    if mode == "uniform_phase":
        diag = torch.rand(shape)
        diag = 2 * np.pi * diag
        diag = torch.exp(1j * diag)
    elif mode == "rademacher":
        diag = torch.where(
            torch.rand(shape, device=device, generator=generator) > 0.5, -1.0, 1.0
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return diag.to(device)


class StructuredRandom(LinearPhysics):
    r"""
    Structured random linear operator model corresponding to the operator

    .. math::

        A(x) = \prod_{i=1}^N (F D_i) x,

    where :math:`F` is a matrix representing a structured transform, :math:`D_i` are diagonal matrices, and :math:`N` refers to the number of layers. It is also possible to replace :math:`x` with :math:`Fx` as an additional 0.5 layer.

    :param tuple input_shape: input shape. If (C, H, W), i.e., the input is a 2D signal with C channels, then zero-padding will be used for oversampling and cropping will be used for undersampling.
    :param tuple output_shape: shape of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math`F` transform is included, ie :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`. Default is 1.
    :param Callable transform_func: structured transform function. Default is :func:`deepinv.physics.functional.dst1`.
    :param Callable transform_func_inv: structured inverse transform function. Default is :func:`deepinv.physics.functional.dst1`.
    :param list diagonals: list of diagonal matrices. If None, a random :math:`{-1,+1}` mask matrix will be used. Default is None.
    :param str device: device of the physics. Default is 'cpu'.
    :param torch.Generator rng: Random number generator. Default is None.
    """

    def __init__(
        self,
        input_shape,
        output_shape,
        n_layers=1,
        transform_func=dst1,
        transform_func_inv=dst1,
        diagonals=None,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        if len(input_shape) == 3:
            mode = compare(input_shape, output_shape)
        else:
            mode = None

        if diagonals is None:
            diagonals = [
                generate_diagonal(
                    shape=input_shape,
                    mode="rademacher",
                    dtype=torch.float,
                    generator=rng,
                    device=device,
                )
            ]

        def A(x):
            if mode == "oversampling":
                x = padding(x, input_shape, output_shape)

            if n_layers - math.floor(n_layers) == 0.5:
                x = transform_func(x)
            for i in range(math.floor(n_layers)):
                diagonal = diagonals[i]
                x = diagonal * x
                x = transform_func(x)

            if mode == "undersampling":
                x = trimming(x, input_shape, output_shape)

            return x

        def A_adjoint(y):
            if mode == "undersampling":
                y = padding(y, input_shape, output_shape)

            for i in range(math.floor(n_layers)):
                diagonal = diagonals[-i - 1]
                y = transform_func_inv(y)
                y = torch.conj(diagonal) * y
            if n_layers - math.floor(n_layers) == 0.5:
                y = transform_func_inv(y)

            if mode == "oversampling":
                y = trimming(y, input_shape, output_shape)

            return y

        super().__init__(A=A, A_adjoint=A_adjoint, **kwargs)
