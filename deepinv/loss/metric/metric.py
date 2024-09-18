from types import ModuleType

import torch

from deepinv.loss.loss import Loss
from deepinv.loss.metric.functional import complex_abs


def import_pyiqa() -> ModuleType:
    try:
        import pyiqa

        return pyiqa
    except ImportError:
        raise ImportError(
            "Metric not available. Please install the pyiqa package with `pip install pyiqa`."
        )


class Metric(Loss):
    def __init__(self, complex_abs=False, train_loss=False):
        super().__init__()
        self.train_loss = train_loss
        self.complex_abs = complex_abs  # NOTE assumes C in dim=1

    def metric(
        self, x_net: torch.Tensor, x: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x_net, x, y, physics, model, **kwargs):
        if self.complex_abs:
            x_net, x = complex_abs(x_net), complex_abs(x)
            # TODO if x complex dtype tensors then do abs on torch.abs

        m = self.metric(x_net=x_net, x=x, y=y, physics=physics, model=model, **kwargs)
        m = 1.0 - m if self.train_loss else m
        return m
