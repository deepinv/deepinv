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

    TODO

    :param model_spec:
    '''
    def __init__(self, model_spec=None):
        super(Denoiser, self).__init__()
        self.denoiser = make(model_spec)

    def forward(self, x, sigma):
        r'''
        '''
        return self.denoiser(x, sigma)


class ProxDenoiser(Denoiser):
    def __init__(self, *args, **kwargs):
        super(ProxDenoiser, self).__init__(*args, **kwargs)

    def forward(self, x, sigma):
        return self.denoiser(x, sigma)


class ScoreDenoiser(Denoiser):
    r'''
    Approximates the score of a distribution using an MMSE denoiser.

    This approximates the score of a distribution using Tweedie's formula, i.e.,

    .. math::

        - \nabla \log p_{\sigma}(x) \propto \left(x-D(x,\sigma)\right)/\sigma^2

    where :math:`p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)` is the prior convolved with a Gaussian kernel,
    :math:`D(\cdot,\sigma)` is a (trained or model-based) denoiser with noise level :math:`\sigma`,
    which is typically set to a low value.
    '''
    def __init__(self, *args, **kwargs):
        super(ScoreDenoiser, self).__init__(*args, **kwargs)

    def forward(self, x, sigma):
        return (x - self.denoiser(x, sigma)) / sigma**2

class REDDenoiser(Denoiser):
    r'''
    TODO
    '''
    def __init__(self, *args, **kwargs):
        super(REDDenoiser, self).__init__(*args, **kwargs)

    def forward(self, x, sigma):
        return x - self.denoiser(x, sigma)