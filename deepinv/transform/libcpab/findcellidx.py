# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:31:34 2018

@author: nsde
"""

#%%
import torch

#%%
def mymin(a, b):
    return torch.where(a < b, a, torch.round(b))

#%%
def findcellidx(ndim, p, nc):
    if ndim==1:   return findcellidx1D(p, *nc)
    elif ndim==2: return findcellidx2D(p, *nc)
    elif ndim==3: return findcellidx3D(p, *nc)
    
#%%
def findcellidx1D(p, nx):
    n = p.shape[1]
    idx = torch.floor(p[0] * nx)
    idx = torch.max(torch.zeros(n, device=p.device), 
                    torch.min(idx, (nx-1)*torch.ones(n, device=p.device)))
    return idx.flatten().to(torch.int64)

#%%
def findcellidx2D(p, nx, ny):
    n = p.shape[1]
    inc_x = 1.0 / nx
    inc_y = 1.0 / ny
    
    #p0 = torch.min(nx * inc_x - 1e-8, torch.max(0.0, p[0]))
    #p1 = torch.min(ny * inc_y - 1e-8, torch.max(0.0, p[1]))
    p0 = torch.clamp(p[0], 0.0, nx * inc_x - 1e-8)
    p1 = torch.clamp(p[1], 0.0, ny * inc_y - 1e-8)
    
    xmod = torch.fmod(p0, inc_x)
    ymod = torch.fmod(p1, inc_y)
    
    x = xmod / inc_x
    y = ymod / inc_y
    
    idx = mymin((nx-1)*torch.ones(n, device=p.device), (p0-xmod) / inc_x) + \
          mymin((ny-1)*torch.ones(n, device=p.device), (p1-ymod) / inc_y) * nx
    idx *= 4
    
    # Out of bound left
    cond1 = (p[0]<=0) & ((p[1]<=0) & (p[1]/inc_y<p[0]/inc_x))
    cond2 = (~ cond1) & (p[0]<=0) & ((p[1] >= ny * inc_y) & (p[1]/inc_y - ny > -p[0]/inc_x))
    cond3 = (~ cond1) & (~ cond2) & (p[0]<=0)
    idx[cond2] += 2
    idx[cond3] += 3

    # Out of bound right
    out = cond1 | cond2 | cond3
    cond4 = (~ out) & (p[0] >= nx*inc_x) & ((p[1]<=0) & (-p[1]/inc_y > p[0]/inc_x - nx))
    cond5 = (~ out) & (~ cond4) & (p[0] >= nx*inc_x) & ((p[1] >= ny*inc_y) & (p[1]/inc_y - ny > p[0]/inc_x-nx))
    cond6 = (~ out) & (~ cond4) & (~ cond5) & (p[0] >= nx*inc_x)
    idx[cond5] += 2
    idx[cond6] += 1
    
    # Out of bound up, nothing to do
    
    # Out of bound down
    out = out | cond4 | cond5 | cond6
    cond7 = (~ out) & (p[1] >= ny*inc_y)
    idx[cond7] += 2

    # Ok, we are inbound
    out = out | cond7
    cond8 = (~ out) & (x<y) & (1-x<y)
    cond9 = (~ out) & (~ cond8) & (x<y)
    cond10 = (~ out) & (~ cond8) & (~ cond9) & (x>=y) & (1-x<y)
    idx[cond8] += 2
    idx[cond9] += 3
    idx[cond10] += 1
    return idx.flatten().to(torch.int64)
    
#%%
def findcellidx3D(p, nx, ny, nz):
    # Conditions for points outside
    cond =  (p[0,:] < 0.0) | (p[0,:] > 1.0) | \
            (p[1,:] < 0.0) | (p[1,:] > 1.0) | \
            (p[2,:] < 0.0) | (p[2,:] > 1.0) 
        
    # Push the points inside boundary
    inc_x, inc_y, inc_z = 1.0 / nx, 1.0 / ny, 1.0 / nz
    half = 0.5
    points_outside = p[:, cond]
    points_outside -= half
    abs_x = torch.abs(points_outside[0])
    abs_y = torch.abs(points_outside[1])
    abs_z = torch.abs(points_outside[2])
    push_x = ((half * inc_x)*((abs_x < abs_y) & (abs_x < abs_z))).to(torch.float32)
    push_y = ((half * inc_y)*((abs_y < abs_x) & (abs_x < abs_z))).to(torch.float32)
    push_z = ((half * inc_z)*((abs_z < abs_x) & (abs_x < abs_y))).to(torch.float32)
    cond_x = abs_x > half
    cond_y = abs_y > half
    cond_z = abs_z > half
    points_outside[0, cond_x] = points_outside[0, cond_x].sign() * (half - push_x[cond_x])
    points_outside[1, cond_y] = points_outside[1, cond_y].sign() * (half - push_y[cond_y])
    points_outside[2, cond_z] = points_outside[2, cond_z].sign() * (half - push_z[cond_z])
    points_outside += half
    p[:, cond] = points_outside

    # Find row, col, depth placement and cell placement
    inc_x, inc_y, inc_z = 1.0/nx, 1.0/ny, 1.0/nz
    zero = torch.tensor(0.0).to(p.device)
    p0 = torch.min(torch.tensor(nx * inc_x - 1e-4).to(p.device), torch.max(zero, p[0]))
    p1 = torch.min(torch.tensor(ny * inc_y - 1e-4).to(p.device), torch.max(zero, p[1]))
    p2 = torch.min(torch.tensor(nz * inc_z - 1e-4).to(p.device), torch.max(zero, p[2]))

    xmod = torch.fmod(p0, inc_x)
    ymod = torch.fmod(p1, inc_y)
    zmod = torch.fmod(p2, inc_z)
    
    i = mymin(torch.tensor(nx - 1).to(torch.float32).to(p.device), ((p0 - xmod) / inc_x))
    j = mymin(torch.tensor(ny - 1).to(torch.float32).to(p.device), ((p1 - ymod) / inc_y))
    k = mymin(torch.tensor(nz - 1).to(torch.float32).to(p.device), ((p2 - zmod) / inc_z))
    idx = 5 * (i + j * nx + k * nx * ny)

    x = xmod / inc_x
    y = ymod / inc_y
    z = zmod / inc_z
    
    # Find subcell location
    cond =  ((k%2==0) & (i%2==0) & (j%2==1)) | \
            ((k%2==0) & (i%2==1) & (j%2==0)) | \
            ((k%2==1) & (i%2==0) & (j%2==0)) | \
            ((k%2==1) & (i%2==1) & (j%2==1))

    tmp = x.clone()
    x[cond] = y[cond]
    y[cond] = 1-tmp[cond]
    
    cond1 = (-x-y+z) >= 0
    cond2 = (x+y+z-2) >= 0
    cond3 = (-x+y-z) >= 0
    cond4 = (x-y-z) >= 0
    idx[cond1] += 1
    idx[cond2 & (~cond1)] += 2
    idx[cond3 & (~cond1) & (~cond2)] += 3
    idx[cond4 & (~cond1) & (~cond2) & (~cond3)] += 4
    idx = idx.flatten().to(torch.int64)
    return idx
