import torch
from torch import nn
import torch.nn.functional as F

from .utils import SQRT2, deg2rad, affine_grid, grid_sample

class Stackgram(nn.Module):
    def __init__(self, out_size, theta=None, circle=True, mode='nearest', dtype=torch.float):
        super(Stackgram, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size = out_size if circle else int((SQRT2*out_size).ceil())
        self.dtype = dtype
        self.all_grids = self._create_grids(self.theta, in_size)
        self.mode = mode

    def forward(self, x):
        stackgram = torch.zeros(x.shape[0], len(self.theta), self.in_size, self.in_size, device=x.device, dtype=self.dtype)

        for i_theta in range(len(self.theta)):
            repline = x[...,i_theta]
            repline = repline.unsqueeze(-1).repeat(1,1,1,repline.shape[2])
            linogram = grid_sample(repline, self.all_grids[i_theta].repeat(x.shape[0],1,1,1).to(x.device), mode=self.mode)
            stackgram[:,i_theta] = linogram

        return stackgram

    def _create_grids(self, angles, grid_size):
        all_grids = []
        for i_theta in range(len(angles)):
            t = deg2rad(angles[i_theta])
            R = torch.tensor([[t.sin(), t.cos(), 0.],[t.cos(), -t.sin(), 0.]], dtype=self.dtype).unsqueeze(0)
            all_grids.append(affine_grid(R, torch.Size([1,1,grid_size,grid_size])))
        return all_grids

class IStackgram(nn.Module):
    def __init__(self, out_size, theta=None, circle=True, mode='bilinear', dtype=torch.float):
        super(IStackgram, self).__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.out_size = out_size
        self.in_size = in_size = out_size if circle else int((SQRT2*out_size).ceil())
        self.dtype = dtype
        self.all_grids = self._create_grids(self.theta, in_size)
        self.mode = mode

    def forward(self, x):
        sinogram = torch.zeros(x.shape[0], 1, self.in_size, len(self.theta), device=x.device, dtype=self.dtype)

        for i_theta in range(len(self.theta)):
            linogram = x[:,i_theta].unsqueeze(1)
            repline = grid_sample(linogram, self.all_grids[i_theta].repeat(x.shape[0],1,1,1).to(x.device), mode=self.mode)
            repline = repline[...,repline.shape[-1]//2]
            sinogram[...,i_theta] = repline

        return sinogram

    def _create_grids(self, angles, grid_size):
        all_grids = []
        for i_theta in range(len(angles)):
            t = deg2rad(angles[i_theta])
            R = torch.tensor([[t.sin(), t.cos(), 0.],[t.cos(), -t.sin(), 0.]], dtype=self.dtype).unsqueeze(0)
            all_grids.append(affine_grid(R, torch.Size([1,1,grid_size,grid_size])))
        return all_grids