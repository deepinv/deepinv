import math
from deepinv.physics.functional import dst1
import numpy as np
import torch

from deepinv.physics.forward import LinearPhysics
from deepinv.utils.decorators import _deprecated_alias


@_deprecated_alias(input_shape="img_size", output_shape="output_size")
def compare(img_size: tuple, output_size: tuple) -> str:
    r"""
    Compare input and output shape to determine the sampling mode.

    :param tuple img_size: Input shape (C, H, W).
    :param tuple output_size: Output shape (C, H, W).

    :return: The sampling mode in ["oversampling","undersampling","equisampling].
    """
    if img_size[1] == output_size[1] and img_size[2] == output_size[2]:
        return "equisampling"
    elif img_size[1] <= output_size[1] and img_size[2] <= output_size[2]:
        return "oversampling"
    elif img_size[1] >= output_size[1] and img_size[2] >= output_size[2]:
        return "undersampling"
    else:
        raise ValueError(
            "Does not support different sampling schemes on height and width."
        )


@_deprecated_alias(input_shape="img_size", output_shape="output_size")
def padding(tensor: torch.Tensor, img_size: tuple, output_size: tuple):
    r"""
    Zero padding function for oversampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple img_size: shape of the input tensor.
    :param tuple output_size: shape of the output tensor.

    :return: (:class:`torch.Tensor`) the zero-padded tensor.
    """
    change_top = math.ceil(abs(img_size[1] - output_size[1]) / 2)
    change_bottom = math.floor(abs(img_size[1] - output_size[1]) / 2)
    change_left = math.ceil(abs(img_size[2] - output_size[2]) / 2)
    change_right = math.floor(abs(img_size[2] - output_size[2]) / 2)
    assert change_top + change_bottom == abs(img_size[1] - output_size[1])
    assert change_left + change_right == abs(img_size[2] - output_size[2])
    return torch.nn.functional.pad(
        tensor,
        (change_left, change_right, change_top, change_bottom),
        mode="constant",
        value=0,
    )


@_deprecated_alias(input_shape="img_size", output_shape="output_size")
def trimming(tensor: torch.Tensor, img_size: tuple, output_size: tuple):
    r"""
    Trimming function for undersampling in structured random phase retrieval.

    :param torch.Tensor tensor: input tensor.
    :param tuple img_size: shape of the input tensor.
    :param tuple output_size: shape of the output tensor.

    :return: (:class:`torch.Tensor`) the trimmed tensor.
    """
    change_top = math.ceil(abs(img_size[1] - output_size[1]) / 2)
    change_bottom = math.floor(abs(img_size[1] - output_size[1]) / 2)
    change_left = math.ceil(abs(img_size[2] - output_size[2]) / 2)
    change_right = math.floor(abs(img_size[2] - output_size[2]) / 2)
    assert change_top + change_bottom == abs(img_size[1] - output_size[1])
    assert change_left + change_right == abs(img_size[2] - output_size[2])
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

    :param tuple img_size: input shape. If (C, H, W), i.e., the input is a 2D signal with C channels, then zero-padding will be used for oversampling and cropping will be used for undersampling.
    :param tuple output_size: shape of outputs.
    :param float n_layers: number of layers :math:`N`. If ``layers=N + 0.5``, a first :math`F` transform is included, ie :math:`A(x)=|\prod_{i=1}^N (F D_i) F x|^2`. Default is 1.
    :param Callable transform_func: structured transform function. Default is :func:`deepinv.physics.functional.dst1`.
    :param Callable transform_func_inv: structured inverse transform function. Default is :func:`deepinv.physics.functional.dst1`.
    :param list diagonals: list of diagonal matrices. If None, a random :math:`{-1,+1}` mask matrix will be used. Default is None.
    :param str device: device of the physics. Default is 'cpu'.
    :param torch.Generator rng: Random number generator. Default is None.
    """

    @_deprecated_alias(input_shape="img_size", output_shape="output_size")
    def __init__(
        self,
        img_size,
        output_size,
        n_layers=1,
        transform_func=dst1,
        transform_func_inv=dst1,
        diagonals=None,
        device="cpu",
        rng: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if len(img_size) == 3:
            self.mode = compare(img_size, output_size)
        else:
            self.mode = None

        self.img_size = img_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.transform_func = transform_func
        self.transform_func_inv = transform_func_inv

        if diagonals is None:
            diagonals = torch.stack(
                [
                    generate_diagonal(
                        shape=img_size,
                        mode="rademacher",
                        dtype=torch.float,
                        generator=rng,
                        device=device,
                    )
                    for _ in range(math.floor(self.n_layers))
                ],
                dim=0,
            )
        elif isinstance(diagonals, list):
            diagonals = torch.stack(diagonals, dim=0)
        self.register_buffer("diagonals", diagonals)
        self.to(device)

    def A(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.mode == "oversampling":
            x = padding(x, self.img_size, self.output_size)

        if self.n_layers - math.floor(self.n_layers) == 0.5:
            x = self.transform_func(x)
        for i in range(math.floor(self.n_layers)):
            diagonal = self.diagonals[i]
            x = diagonal * x
            x = self.transform_func(x)

        if self.mode == "undersampling":
            x = trimming(x, self.img_size, self.output_size)

        return x

    def A_adjoint(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.mode == "undersampling":
            y = padding(y, self.img_size, self.output_size)

        for i in range(math.floor(self.n_layers)):
            diagonal = self.diagonals[-i - 1]
            y = self.transform_func_inv(y)
            y = torch.conj(diagonal) * y
        if self.n_layers - math.floor(self.n_layers) == 0.5:
            y = self.transform_func_inv(y)

        if self.mode == "oversampling":
            y = trimming(y, self.img_size, self.output_size)

        return y
