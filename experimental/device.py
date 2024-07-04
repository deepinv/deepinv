import sys
sys.path.append("/home/zhhu/workspaces/deepinv/")

import deepinv as dinv
import torch

print(torch.cuda.is_available())
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
print(device)