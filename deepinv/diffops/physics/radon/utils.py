import torch
import torch.nn.functional as F

if torch.__version__>'1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

# constants
PI = 4*torch.ones(1).atan()
SQRT2 = (2*torch.ones(1)).sqrt()

def fftfreq(n):
    val = 1.0/n
    results = torch.zeros(n)
    N = (n-1)//2 + 1
    p1 = torch.arange(0, N)
    results[:N] = p1
    p2 = torch.arange(-(n//2), 0)
    results[N:] = p2
    return results*val

def deg2rad(x):
    return x*PI/180