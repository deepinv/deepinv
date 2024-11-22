import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from .base import Denoiser

# code adapted from https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598


class MedianFilter(Denoiser):
    r"""
    Median filter.

    It computes the median value of a sliding window over the input tensor. The window is defined by the kernel size.

    :param int kernel_size: size of pooling kernel, int or 2-tuple
    :param padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
    :param same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=9, padding=0, same=True):
        super(MedianFilter, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(1)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x, sigma=None, **kwargs):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
