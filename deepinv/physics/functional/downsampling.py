from torch import Tensor


def downsample(x: Tensor, factor: int):
    return x[:, :, ::factor, ::factor]
