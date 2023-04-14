import copy
import torch
import torch.nn as nn


def online_weights_path():
    return 'https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files='

models = {}

# TAKEN FROM https://github.com/jaewon-lee-b/lte/blob/main/models/models.py
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    return model


class Denoiser(nn.Module):
    r'''
    Base denoiser class.

    Plug-and-Play (PnP) / Regularization bu Denoising (RED) algorithms for Image Restoration.

    Consists in replacing prox_g or grad_g with a denoiser.

    TODO

    :param model_spec: a dictionary must contain the necessary information for generating the model.
    '''
    def __init__(self, model_spec=None):
        super(Denoiser, self).__init__()
        self.denoiser = make(model_spec)

    def forward(self, x, sigma):
        r'''
        '''
        return self.denoiser(x, sigma)


class ScoreDenoiser(Denoiser):
    r'''
    Approximates the score of a distribution using an MMSE denoiser.

    TODO : talk about sigma_normalize paramter with RED 

    This approximates the score of a distribution using Tweedie's formula, i.e.,

    .. math::

        - \nabla \log p_{\sigma}(x) \propto \left(x-D(x,\sigma)\right)/\sigma^2

    where :math:`p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)` is the prior convolved with a Gaussian kernel,
    :math:`D(\cdot,\sigma)` is a (trained or model-based) denoiser with noise level :math:`\sigma`,
    which is typically set to a low value.

    .. note::

        This class can also be used with maximum-a-posteriori (MAP) denoisers,
        but :math:`p_{\sigma}(x)` is not given by the convolution with a Gaussian kernel, but rather
        given by the Moreau-Yosida envelope of :math:`p(x)`, i.e.,

        .. math::

            p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.


    '''
    def __init__(self, *args, sigma_normalize=True, **kwargs):
        super(ScoreDenoiser, self).__init__(*args, **kwargs)
        self.sigma_normalize = sigma_normalize

    def forward(self, x, sigma):
        if self.sigma_normalize :
            return (x - self.denoiser(x, sigma)) / sigma**2
        else :
            return (x - self.denoiser(x, sigma))
