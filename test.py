import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.optim import DataFidelity
from deepinv.optim.data_fidelity import L2, IndicatorL2, L1, AmplitudeLoss
from deepinv.optim.prior import Prior, PnP, RED
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.optim_iterators import GDIteration


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.Tensor([[[1], [4], [-0.5]]]).to(device)
y = torch.Tensor([[[1], [1], [1]]]).to(device)

data_fidelity = L1()
# Check prox
threshold = 0.5
prox_manual = torch.Tensor([[[1.0], [3.5], [0.0]]]).to(device)
prox = data_fidelity.d.prox(x, y, gamma=threshold)
print(prox)
assert torch.allclose(prox, prox_manual)
