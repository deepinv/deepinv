from deepinv.loss.metric.metric import import_pyiqa, Metric


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.

    See docs for ``forward()`` below for more details.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import LPIPS
    >>> m = LPIPS()
    >>> x_net = x = torch.ones(1, 2, 8, 8) # B,C,H,W
    >>> m(x_net, x)
    tensor(0.)

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.lpips = pyiqa.create_metric("lpips").to(device)
        self.lower_better = self.lpips.lower_better

    def metric(self, x_net, x, *args, **kwargs):
        return self.lpips(x_net, x)


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    It is a no-reference image quality metric that estimates the quality of images.

    See docs for ``forward()`` below for more details.

    :Example:

    >>> import torch
    >>> from deepinv.loss.metric import NIQE
    >>> m = NIQE()
    >>> x_net = x = torch.ones(1, 2, 8, 8) # B,C,H,W
    >>> m(x_net)
    tensor(0.)

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.niqe = pyiqa.create_metric("niqe").to(device)
        self.lower_better = self.niqe.lower_better

    def metric(self, x_net, *args, **kwargs):
        return self.niqe(x_net)
