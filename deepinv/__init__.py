from .__about__ import *
import torch

# import utils

from utils.nn import adjust_learning_rate, save_model
from utils.logger import AverageMeter, ProgressMeter, get_timestamp
from utils.metric import cal_psnr
from utils.plotting import plot_debug
from deepinv.diffops.models import ar

# from deepinv.diffops.models.iterative import denoising

import numpy as np

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


try:
    from .diffops import models, physics, transform

    __all__ += ['models']
except ImportError:
    pass

try:
    from deepinv import loss
    __all__ += ['loss']
except ImportError:
    print('Warning: couldnt import loss subpackage')
    pass


try:
    from deepinv.diffops import models
    __all__ += ['iterative']
except ImportError:
    print('Warning: couldnt import models subpackage')
    pass

try:
    from deepinv.diffops.models import iterative
    __all__ += ['iterative']
except ImportError:
    print('Warning: couldnt import iterative subpackage')
    pass

try:
    from deepinv.diffops import physics
    __all__ += ['physics']
except ImportError:
    print('Warning: couldnt import physics subpackage')
    pass

try:
    from deepinv import datasets
    __all__ += ['datasets']
except ImportError:
    print('Warning: couldnt import datasets subpackage')
    pass

try:
    __all__ += ['transform']
except ImportError:
    print('Warning: couldnt import transform subpackage')
    pass


try:
    from deepinv.diffops import noise
    __all__ += ['noise']
except ImportError:
    pass

try:
    from torch import optim
    __all__ += ['optim']
except ImportError:
    pass

try:
    from deepinv.loss import loss as metric
    __all__ += ['metric']
except ImportError:
    pass

# GLOBAL PROPERTY
dtype = torch.float
device = torch.device(f'cuda:0')


def load_checkpoint(model, path_checkpoint, device):
    checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def data_parallel(model, ngpu=1):
    if ngpu>1:
        model = torch.nn.DataParallel(model, list(range(ngpu)))
    return model

def train(model,
          train_dataloader,
          learning_rate,
          epochs,
          schedule,
          loss_closure=None,  # list
          loss_weight=None,
          optimizer=None,
          physics=None,
          dtype=torch.float,
          device=torch.device(f"cuda:0"),
          ckp_interval=100,
          save_path=None,
          verbose=False,
          save_dir = '.'):

    losses = AverageMeter('loss', ':.3e')
    meters = [losses]
    losses_verbose = []
    if verbose:
        losses_verbose = [AverageMeter('loss_' + l.name, ':.3e') for l in loss_closure]
        psnr_net = AverageMeter('psnr_net', ':.2f')
        psnr_fbp = AverageMeter('psnr_fbp', ':.2f')

        for loss in losses_verbose:
            meters.append(loss)
        meters.append(psnr_fbp)
        meters.append(psnr_net)


    progress = ProgressMeter(epochs, meters, surfix=f"[{save_path}]")

    save_path = save_dir +  '/ckp/{}'.format('_'.join([get_timestamp(), save_path]))

    f = model
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate, cos=False, epochs=epochs, schedule=schedule)

        for i, x in enumerate(train_dataloader):
            x = x[0] if isinstance(x, list) else x
            x = x.type(dtype).to(device)  # todo: dataloader is only for y

            y0 = physics(x)  # generate noisy measurement input y

            x1 = f(y0, physics)

            y1 = physics.A(x1)

            loss_total = 0
            for l, w in zip(loss_closure, loss_weight):
                loss = 0
                if l.name in ['mc']:
                    loss = w * l(y0, x1, physics)
                if l.name in ['ms']:
                    loss = w * l(y0, physics, f)
                if l.name in ['sup']:
                    loss = w * l(x1, x)
                if l.name.startswith('suremc'):
                    loss = w * l(y0, y1, physics, f)
                if l.name in ['ei', 'rei']:
                    loss = w * l(x1, physics, f)
                loss_total += loss

                if verbose:
                    for loss_verbose in losses_verbose:
                        loss_verbose.update(loss.item())


            losses.update(loss_total.item())

            if verbose:
                psnr_fbp.update(cal_psnr(physics.A_dagger(y0), x))
                psnr_net.update(cal_psnr(x1, x))

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        progress.display(epoch + 1)
        save_model(epoch, model, optimizer, ckp_interval, epochs, save_path)

    return model



def test(model,
          test_dataloader,
          physics=None,
          dtype=torch.float,
          device=torch.device(f"cuda:0"),
          plot=True):

    # f = denoising(model)

    f = ar(model)

    psnr_fbp = []
    psnr_net = []

    for i, x in enumerate(test_dataloader):
        x = x[0] if isinstance(x, list) else x
        x = x.type(dtype).to(device)

        y0 = physics(x)  # generate measurement input y

        x1 = f(y0, physics)

        if i==0 and plot:
            plot_debug([physics.A_dagger(y0), x1, x], ['Linear Inv.', 'Estimated', 'Ground Truth'])

        psnr_fbp.append(cal_psnr(physics.A_dagger(y0), x))
        psnr_net.append(cal_psnr(x1, x))

    test_psnr = np.mean(psnr_net)
    print('Test PSNR {}'.format(test_psnr))

    return test_psnr