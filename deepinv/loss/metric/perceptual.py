from deepinv.loss.metric.metric import import_pyiqa, Metric
import torch
import torch.nn.functional as F


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Calculates the LPIPS :math:`\text{LPIPS}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> from deepinv.utils import load_example
    >>> from deepinv.loss.metric import LPIPS
    >>> m = LPIPS() # doctest: +IGNORE_RESULT
    >>> x = load_example("celeba_example.jpg", img_size=128)
    >>> x_net = x - 0.01
    >>> m(x_net, x) # doctest: +ELLIPSIS
    tensor([...])

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param bool as_loss: if True, returns LPIPS as a loss. Default: False.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
    """

    def __init__(self, device="cpu", check_input_range=False, as_loss=False, **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.lpips = pyiqa.create_metric(
            "lpips", check_input_range=check_input_range, device=device, as_loss=as_loss
        ).to(device)
        self.lower_better = self.lpips.lower_better

    def metric(self, x_net, x, *args, **kwargs):
        return self.lpips(x_net, x).squeeze(-1)


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    Calculates the NIQE :math:`\text{NIQE}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.
    It is a no-reference image quality metric that estimates the quality of images.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> from deepinv.utils import load_example
    >>> from deepinv.loss.metric import NIQE
    >>> m = NIQE() # doctest: +IGNORE_RESULT
    (...)
    >>> x_net = load_example("celeba_example.jpg", img_size=128)
    >>> m(x_net) # doctest: +ELLIPSIS
    tensor([...])

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).
    """

    def __init__(self, device="cpu", check_input_range=False, **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.niqe = pyiqa.create_metric(
            "niqe", check_input_range=check_input_range, device=device
        ).to(device)
        self.lower_better = self.niqe.lower_better

    def metric(self, x_net, *args, **kwargs):
        n = self.niqe(x_net).float()
        return n.unsqueeze(0) if n.dim() == 0 else n


class BlurStrength(Metric):
    """
    No-reference blur strength metric for batched images.

    Returns a value in (0, 1) for each image in the batch, where 0 indicates a very sharp image and 1 indicates a very blurry image.

    The metric has been introduced in :cite:t:`crete2007blur`.

    :param int h_size: size of the uniform blur filter. Default: 11.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

    |sep|

    :Example:

    >>> from deepinv.loss.metric import BlurStrength
    >>> m = BlurStrength()
    >>> x_net = torch.randn(2, 3, 16, 16)  # batch of 2 RGB images
    >>> m(x_net).shape
    torch.Size([2])

    """

    def __init__(self, h_size=11, **kwargs):
        super().__init__(**kwargs)
        self.h_size = h_size
        self.lower_better = False

    def metric(self, x_net, *args, **kwargs):
        """
        Compute blur strength metric for a batch of images.

        :param x_net: (B, C, ...) input tensors with C=1 or 3 channels. The spatial dimensions can be 1D, 2D, or higher.
        :return: (B,) tensor of blur strength values in (0,1) for each image in the batch.
        """
        # convert to grayscale: (B, C, ...) → (B, 1, ...)
        assert x_net.shape[1] in [1, 3], "Input must have 1 or 3 channels."

        x = x_net

        if x.shape[1] == 3:  # RGB to grayscale
            r = x[:, 0:1]
            g = x[:, 1:2]
            b = x[:, 2:3]
            x = 0.2989 * r + 0.5870 * g + 0.1140 * b

        B, C, *spatial = x.shape
        n_spatial = len(spatial)

        # slices = slice(2, s-1) per dimension
        slices = (slice(None), slice(None)) + tuple(slice(2, s - 1) for s in spatial)

        # Compute metric for each spatial axis
        results = []

        # spatial axes start at dim=2
        for ax in range(2, 2 + n_spatial):
            # 1D uniform blur
            filt = uniform_filter1d(x, self.h_size, axis=ax)

            # Sobel derivatives
            sharp = torch.abs(sobel_1d(x, axis=ax))
            blur = torch.abs(sobel_1d(filt, axis=ax))

            # clamp/sharpness difference
            t = torch.clamp(sharp - blur, min=0)

            # sums over all except batch dimension
            m1 = sharp[slices].sum(dim=list(range(1, sharp.ndim)))
            m2 = t[slices].sum(dim=list(range(1, t.ndim)))

            # per-image blur per-axis
            axis_blur = torch.abs(m1 - m2) / (m1 + 1e-12)
            results.append(axis_blur)

        results = torch.stack(results, dim=1)  # (B, n_spatial)
        return results.max(dim=1).values  # (B,)


def uniform_filter1d(x, size, axis):
    """
    Batched 1D uniform filter along an arbitrary axis.
    x: (B, C, ...)
    """
    pad = size // 2
    kernel = torch.ones(1, 1, size, device=x.device, dtype=x.dtype) / size

    # move axis to last dim
    x_perm = x.transpose(axis, -1)
    orig_shape = x_perm.shape

    # flatten spatial dims except last
    x_flat = x_perm.reshape(-1, 1, orig_shape[-1])

    x_flat = F.pad(x_flat, (pad, pad), mode="reflect")
    out = F.conv1d(x_flat, kernel)

    out = out.reshape(orig_shape)
    out = out.transpose(axis, -1)
    return out


def sobel_1d(x, axis):
    """
    Batched 1D Sobel derivative along an arbitrary axis.
    x: (B, C, ...)
    """
    kernel = torch.tensor([[-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    pad = 1

    # move target axis to last dim
    x_perm = x.transpose(axis, -1)
    orig_shape = x_perm.shape

    # flatten all leading dims
    x_flat = x_perm.reshape(-1, 1, orig_shape[-1])

    x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
    out = F.conv1d(x_pad, kernel.unsqueeze(0))

    out = out.reshape(orig_shape)
    out = out.transpose(axis, -1)
    return out
