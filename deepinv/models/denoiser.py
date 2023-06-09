import copy
import torch
import torch.nn as nn


def online_weights_path():
    return "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files="


models = {}


# TAKEN FROM https://github.com/jaewon-lee-b/lte/blob/main/models/models.py
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(model_spec, args=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    model = models[model_spec["name"]](**model_args)
    return model


class Denoiser(nn.Module):
    r"""
    Builds a (possibly pretrained) denoiser.

    The input should be a dictionary containing the inputs to a denoiser in :ref:`denoiser-docs`.

    For example:

    ::

        # Load the DnCNN denoiser with weights trained using the Lipschitz constraints.
        model_spec = {
        "name": "dncnn",
        "args": {
            "device": dinv.device,
            "in_channels": 3,
            "out_channels": 3,
            "pretrained": "download_lipschitz",
            },
        }

        model = Denoiser(model_spec)

    :param dict model_spec: a dictionary containing the necessary information for generating the model.
    """

    def __init__(self, model_spec=None, denoiser=None):
        super(Denoiser, self).__init__()
        if denoiser is not None:
            self.denoiser = denoiser
        elif model_spec is not None:
            self.denoiser = make(model_spec)
        else:
            raise ValueError("Either denoiser or model_spec must be provided.")

    def forward(self, x, sigma):
        r""" """
        return self.denoiser(x, sigma)


class ScoreDenoiser(Denoiser):
    r"""
    Approximates the score of a distribution using a denoiser.

    This approximates the score of a distribution using Tweedie's formula, i.e.,

    .. math::

        - \nabla \log p_{\sigma}(x) \propto \left(x-D(x,\sigma)\right)/\sigma^2

    where :math:`p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)` is the prior convolved with a Gaussian kernel,
    :math:`D(\cdot,\sigma)` is a (trained or model-based) denoiser with noise level :math:`\sigma`,
    which is typically set to a low value.

    If ``sigma_normalize=False``, the score is computed without normalization, i.e.,
    :math:`x-D(x,\sigma)`. This can be useful when using this class in the context of
    `Regularization by Denoising (RED) <https://arxiv.org/abs/1611.02862>` which doesn't require the normalization.


    .. note::

        This class can also be used with maximum-a-posteriori (MAP) denoisers,
        but :math:`p_{\sigma}(x)` is not given by the convolution with a Gaussian kernel, but rather
        given by the Moreau-Yosida envelope of :math:`p(x)`, i.e.,

        .. math::

            p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.


    """

    def __init__(self, *args, sigma_normalize=True, **kwargs):
        super(ScoreDenoiser, self).__init__(*args, **kwargs)
        self.sigma_normalize = sigma_normalize

    def forward(self, x, sigma):
        r"""
        Applies the denoiser to the input signal.

        :param torch.Tensor x: the input tensor.
        :param float sigma: the noise level.
        """
        if self.sigma_normalize:
            return (1 / sigma ** 2) * (x - self.denoiser(x, sigma))
        else:
            return x - self.denoiser(x, sigma)
