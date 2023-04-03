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

    :param model_spec:
    :param init:
    :param float stepsize:
    :param float sigma_denoiser:
    :param int max_iter:
    :param str,torch.device device: cpu or gpu
    '''
    def __init__(self, model_spec=None, init=None, stepsize=1., sigma_denoiser=0.05, max_iter=None,
                 device=torch.device('cpu')):
        super(Denoiser, self).__init__()

        self.denoiser = make(model_spec)
        self.max_iter = max_iter
        self.device = device  # Is this really needed?
        self.init = init

        if isinstance(sigma_denoiser, float):
            if self.max_iter is not None:
                self.sigma_denoiser = [sigma_denoiser] * self.max_iter
            else:
                self.sigma_denoiser = sigma_denoiser
        elif isinstance(sigma_denoiser, list):
            print(len(sigma_denoiser))
            print('max ister ', self.max_iter)
            assert len(sigma_denoiser) == self.max_iter
            self.sigma_denoiser = sigma_denoiser
        else:
            print(sigma_denoiser)
            raise ValueError('sigma_denoiser must be either float or a list of length max_iter')

        if isinstance(stepsize, float):
            if self.max_iter is not None:
                self.stepsize = [stepsize] * max_iter  # Should be a list
            else:
                self.stepsize = stepsize
        elif isinstance(stepsize, list):
            assert len(stepsize) == self.max_iter
            self.stepsize = stepsize
        else:
            raise ValueError('stepsize must be either float or a list of length max_iter')

    def forward(self, x, sigma):
        r'''
        '''
        return self.denoiser(x, sigma)


class ProxDenoiser(Denoiser):
    def __init__(self, *args, **kwargs):
        super(ProxDenoiser, self).__init__(*args, **kwargs)

    def forward(self, x, sigma, it=None):
        if isinstance(self.denoiser, list) or isinstance(self.denoiser, nn.ModuleList):
            out = self.denoiser[it](x, sigma)
        else:
            out = self.denoiser(x, sigma)
        return out


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

    def forward(self, x, sigma=None, it=None):
        if sigma is not None:
            if isinstance(self.denoiser, list) or isinstance(self.denoiser, nn.ModuleList):
                out = x - self.denoiser[it](x, sigma)
            else:
                out = x - self.denoiser(x, sigma)
        else:
            out = (x - self.denoiser(x, self.sigma_denoiser))/self.sigma_denoiser**2
        return out
