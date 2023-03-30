import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from deepinv.physics.forward import Physics

if torch.__version__>'1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(input, grid, align_corners=True, mode=mode)
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample

# constants
PI = 4*torch.ones(1).atan()
SQRT2 = (2*torch.ones(1)).sqrt()


class CT(Physics):
    r'''
    Radon and Inverse Radon (IRadon) operators for computed tomography (CT) image reconstruction problems.
    '''
    def __init__(self, img_width, radon_view, uniform=True, device='cuda:0', I0=1e5, **kwargs):
        '''
        :param img_width: (int), width of CT image (squared).
        :param radon_view: (int), number of views (angles)
        :param uniform: (bool), 'True': uniformly sampled, 'False': limited angles
        :param device:  (str) options = 'cpu', 'cuda=0'
        :param I0: X-ray source density
        '''
        super(CT, self).__init__(**kwargs)
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta).to(device)
        self.iradon = IRadon(img_width, theta).to(device)

        self.name='ct'
        self.I0 = I0

        # used for normalzation input
        self.MAX = 0.032 / 5
        self.MIN = 0

    def forward(self, x):
        m = self.I0 * torch.exp(-self.radon(x))  # clean GT measurement
        m = self.noise(m) # Mixed-Poisson-Gaussian Noise
        return m

    def A(self, x):
        m = self.I0 * torch.exp(-self.radon(x)) # clean GT measurement
        return m

    def A_dagger(self, y):
        # approximated pseudo-inverse: Filtered Back Projection (FBP)
        return self.iradon(y, filtering=True)

    def A_adjoint(self, y):
        # true adjoint: Back Projection without using Ramper filtering
        return self.iradon(y, filtering=False)

class Radon(nn.Module):
    def __init__(self, in_size=None, theta=None, circle=False, dtype=torch.float):
        super(Radon, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle)

    def forward(self, x):
        N, C, W, H = x.shape
        assert (W == H)

        if self.all_grids is None:
            self.all_grids = self._create_grids(self.theta, W, self.circle)

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape
        out = torch.zeros(N, C, W, len(self.theta), device=x.device, dtype=self.dtype)

        for i in range(len(self.theta)):
            rotated = grid_sample(x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device))
            out[..., i] = rotated.sum(2)

        return out

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)
            all_grids.append(affine_grid(R, torch.Size([1, 1, grid_size, grid_size])))
        return all_grids

class IRadon(nn.Module):
    def __init__(self, use_filter, in_size=None, theta=None, circle=False,
                 out_size=None, dtype=torch.float):
        super(IRadon, self).__init__()
        self.circle = circle
        self.theta = theta if theta is not None else torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle)
        self.filter = use_filter if use_filter is not None else lambda x: x

    def forward(self, x, filtering=True):
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = int((it_size / SQRT2).floor()) if not self.circle else it_size
        # if None in [self.ygrid, self.xgrid, self.all_grids]:
        if self.ygrid is None or self.xgrid is None or self.all_grids is None :
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)

        x = self.filter(x) if filtering else x

        reco = torch.zeros(x.shape[0], ch_size, it_size, it_size, device=x.device, dtype=self.dtype)
        for i_theta in range(len(self.theta)):
            reco += grid_sample(x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1).to(x.device))

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1]))

        if self.circle:
            reconstruction_circle = (self.xgrid ** 2 + self.ygrid ** 2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(x.shape[0], ch_size, 1, 1)
            reco[~reconstruction_circle] = 0.

        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype)
        return torch.meshgrid(unitrange, unitrange)

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for i_theta in range(len(angles)):
            X = torch.ones(grid_size, dtype=self.dtype).view(-1, 1).repeat(1, grid_size) * i_theta * 2. / (
                        len(angles) - 1) - 1.
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0))
        return all_grids

class RampFilter(nn.Module):
    def __init__(self):
        super(RampFilter, self).__init__()

    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = \
            max(64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil()))
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0,0,0,pad_width))
        f = self._get_fourier_filter(padded_tensor.shape[2]).to(x.device)
        fourier_filter = self.create_filter(f)
        fourier_filter = fourier_filter.unsqueeze(-2)
        projection = torch.rfft(padded_tensor.transpose(2,3), 1, onesided=False).transpose(2,3) * fourier_filter
        return torch.irfft(projection.transpose(2,3), 1, onesided=False).transpose(2,3)[:,:,:input_size,:]

    def _get_fourier_filter(self, size):
        n = torch.cat([
            torch.arange(1, size / 2 + 1, 2),
            torch.arange(size / 2 - 1, 0, -2)
        ])

        f = torch.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (PI * n) ** 2

        fourier_filter = torch.rfft(f, 1, onesided=False)
        fourier_filter[:,1] = fourier_filter[:,0]

        return 2*fourier_filter

    def create_filter(self, f):
        return f

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
