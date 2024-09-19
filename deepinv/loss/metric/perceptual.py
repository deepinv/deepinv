from deepinv.loss.metric.metric import import_pyiqa, Metric


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.

    :param bool train: if ``True``, the metric is used for training. Default: ``False``.
    :param str device: device to use for the metric computation. Default: 'cpu'.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.lpips = pyiqa.create_metric("lpips").to(device)

    def metric(self, x_net, x, *args, **kwargs):
        r"""
        Computes the LPIPS metric.

        :param torch.Tensor x_net: reconstructed image.
        :param torch.Tensor x: reference image.
        :return: torch.Tensor size (batch_size,).
        """
        return self.lpips(x_net, x)


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    It is a no-reference image quality metric that estimates the quality of images.

    :param str device: device to use for the metric computation. Default: 'cpu'.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.niqe = pyiqa.create_metric("niqe").to(device)

    def metric(self, x_net, *args, **kwargs):
        r"""
        Computes the NIQE metric (no reference).

        :param torch.Tensor x_net: input tensor.
        :return: torch.Tensor size (batch_size,).
        """
        return self.niqe(x_net)
