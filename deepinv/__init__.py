from .__about__ import *
import torch


from utils.nn import adjust_learning_rate, save_model
from utils.logger import AverageMeter, ProgressMeter

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
    from deepinv import models
    __all__ += ['models']
except ImportError:
    pass

try:
    from deepinv import loss
    __all__ += ['loss']
except ImportError:
    pass


try:
    from deepinv.models import iterative
    __all__ += ['iterative']
except ImportError:
    pass


try:
    from deepinv import datasets
    __all__ += ['datasets']
except ImportError:
    pass

try:
    from deepinv import nn
    __all__ += ['nn']
except ImportError:
    pass


try:
    from deepinv.diffops import physics
    __all__ += ['physics']
except ImportError:
    pass


try:
    from deepinv.diffops import transform
    __all__ += ['transform']
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


# def iterative(mode, backbone_net, weight_tied, step_size, iterations, device, physics):
#     return models.iterative.unroll(mode, backbone_net, weight_tied, step_size, iterations, device, physics)


def train(model, train_dataloader, learning_rate, epochs, schedule,
          loss_closure=None,
          optimizer=None,
          physics=None,
          dtype=torch.float,
          device=torch.device(f"cuda:0")):

    losses = AverageMeter('loss', ':.2e')
    meters = [losses]
    progress = ProgressMeter(epochs, meters, prefix=f"[deepinv]")

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate, cos=False, epochs=epochs, schedule=schedule)

        for i, x in enumerate(train_dataloader):
            x = x[0] if isinstance(x, list) else x
            x = x.type(dtype).to(device) # todo: dataloader is only for y

            y0 = physics.A(x)  # generate measurement input y

            loss = loss_closure(y0, model)

            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress.display(epoch + 1)
        save_model(epoch, model, optimizer)
    return model
