from deepinv.loss.metric.metric import import_pyiqa, import_cleanfid, Metric


class LPIPS(Metric):
    r"""
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Calculates the LPIPS :math:`\text{LPIPS}(\hat{x},x)` where :math:`\hat{x}=\inverse{y}`.

    Computes the perceptual similarity between two images, based on a pre-trained deep neural network.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> from deepinv.utils.demo import load_example
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
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    """

    def __init__(self, device="cpu", check_input_range=False, **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.lpips = pyiqa.create_metric(
            "lpips", check_input_range=check_input_range
        ).to(device)
        self.lower_better = self.lpips.lower_better

    def metric(self, x_net, x, *args, **kwargs):
        return self.lpips(x_net, x).squeeze(-1)
    

class FID(Metric):
    r"""
    Frechet Inception Distance (FID) metric.

    Computes the Frechet Inception Distance between two sets of images, which corresponds to the Wasserstein distance between empirical Gaussians in the feature space of a pre-trained Inception network.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`, based on the implementation from `clean-fid <https://github.com/GaParmar/clean-fid?tab=readme-ov-file#computing-fid>`_.
    """
    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.device = device
        self.fid_metric = pyiqa.create_metric("fid").to(device)
        
    def metric(self, out_dir, ref_dir = None, dataset_name = None, dataset_res=1024, dataset_split='train', num_workers = 4, batch_size = 8, **kwargs):
        return self.fid_metric(fdir1 = out_dir, fdir2 = ref_dir, dataset_name=dataset_name, dataset_res=dataset_res, dataset_split=dataset_split, num_workers = num_workers, batch_size = batch_size, device=self.device, **kwargs)


class NIQE(Metric):
    r"""
    Natural Image Quality Evaluator (NIQE) metric.

    Calculates the NIQE :math:`\text{NIQE}(\hat{x})` where :math:`\hat{x}=\inverse{y}`.
    It is a no-reference image quality metric that estimates the quality of images.
    Uses implementation from `pyiqa <https://pypi.org/project/pyiqa/>`_.

    .. note::

        By default, no reduction is performed in the batch dimension.

    :Example:

    >>> from deepinv.utils.demo import load_example
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
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    :param bool check_input_range: if True, ``pyiqa`` will raise error if inputs aren't in the appropriate range ``[0, 1]``.
    """

    def __init__(self, device="cpu", check_input_range=False, **kwargs):
        super().__init__(**kwargs)
        pyiqa = import_pyiqa()
        self.niqe = pyiqa.create_metric("niqe", check_input_range=check_input_range).to(
            device
        )
        self.lower_better = self.niqe.lower_better

    def metric(self, x_net, *args, **kwargs):
        n = self.niqe(x_net).float()
        return n.unsqueeze(0) if n.dim() == 0 else n
