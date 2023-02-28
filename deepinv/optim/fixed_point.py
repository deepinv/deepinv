import torch
import torch.nn as nn
from deepinv.optim.utils import check_conv

class FixedPoint(nn.Module):
    '''
    '''
    def __init__(self, iterator, max_iter=50, early_stop=True, crit_conv=None, verbose=False) :
        super().__init__()
        self.iterator = iterator 
        self.max_iter = max_iter
        self.crit_conv = crit_conv
        self.verbose = verbose
        self.early_stop = early_stop

    def forward(self, init, *args):
        x = init
        # print('And inside fixed point, init is ', init.shape)
        for it in range(self.max_iter):
            x_prev = x if type(x) is not tuple else x[0]
            x = self.iterator(x, it, *args)
            x_out = x if type(x) is not tuple else x[0]
            if self.early_stop and check_conv(x_prev, x_out, it, self.crit_conv, self.verbose):
                break
        return x_out


