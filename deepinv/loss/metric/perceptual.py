from deepinv.loss.metric.metric import import_pyiqa, Metric


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    See docs for ``forward()`` below for more details.

    :Example:

    >>> from deepinv.utils.demo import get_image_url, load_url_image
    >>> from deepinv.loss.metric import LPIPS
    >>> ();m = LPIPS();() # doctest: +ELLIPSIS
    (...)
    >>> x = load_url_image(get_image_url("celeba_example.jpg"), img_size=128)
    >>> x_net = x + 0.01
    >>> m(x_net, x)
    tensor([0.0005])

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.lpips = pyiqa.create_metric("lpips").to(device)
        self.lower_better = self.lpips.lower_better

    def metric(self, x_net, x, *args, **kwargs):
        return self.lpips(x_net, x).squeeze(-1)


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    It is a no-reference image quality metric that estimates the quality of images.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    See docs for ``forward()`` below for more details.

    :Example:

    >>> from deepinv.utils.demo import get_image_url, load_url_image
    >>> from deepinv.loss.metric import NIQE
    >>> ();m = NIQE();() # doctest: +ELLIPSIS
    (...)
    >>> x_net = load_url_image(get_image_url("celeba_example.jpg"), img_size=128)
    >>> m(x_net)
    tensor([8.1880])

    :param str device: device to use for the metric computation. Default: 'cpu'.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.niqe = pyiqa.create_metric("niqe").to(device)
        self.lower_better = self.niqe.lower_better

    def metric(self, x_net, *args, **kwargs):
        n = self.niqe(x_net).float()
        return n.unsqueeze(0) if n.dim() == 0 else n
