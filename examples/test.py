from deepinv.loss.metric.perceptual import NIQE
import torch

import requests
import numpy as np
from PIL import Image
from io import BytesIO
from deepinv.loss.metric.metric import Metric, import_pyiqa


class OldNIQE(Metric):
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
        self.niqe = pyiqa.create_metric(
            "niqe", check_input_range=check_input_range, device=device
        ).to(device)
        self.lower_better = self.niqe.lower_better

    def metric(self, x_net, *args, **kwargs):
        n = self.niqe(x_net).float()
        return n.unsqueeze(0) if n.dim() == 0 else n


if __name__ == "__main__":
    # https://quality.nfdi4ing.de/en/latest/image_quality/NIQE.html

    n = NIQE()
    old_n = OldNIQE()
    url = "https://quality.nfdi4ing.de/en/latest/_images/Reference_Image.png"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    arr = np.asarray(img).transpose(2, 1, 0)
    t = torch.tensor(arr).unsqueeze(0).float()
    print(n.metric(t))
    t = torch.tensor(arr).unsqueeze(0).float()
    print(old_n.metric(t))
    url = "https://quality.nfdi4ing.de/en/latest/_images/Image_Dark.png"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    arr = np.asarray(img).transpose(2, 1, 0)
    t = torch.tensor(arr).unsqueeze(0).float()
    print(n.metric(t))
    t = torch.tensor(arr).unsqueeze(0).float()
    print(old_n.metric(t))

    url = "https://quality.nfdi4ing.de/en/latest/_images/Image_Sunshine.png"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    arr = np.asarray(img).transpose(2, 1, 0)
    t = torch.tensor(arr).unsqueeze(0).float()
    print(n.metric(t))
    t = torch.tensor(arr).unsqueeze(0).float()
    print(old_n.metric(t))
