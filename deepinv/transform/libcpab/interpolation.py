# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:26:30 2018

@author: nsde
"""

#%%
import torch

#%%
def interpolate(ndim, data, grid, outsize):
    if ndim==1: return interpolate1D(data, grid, outsize)
    elif ndim==2: return interpolate2D(data, grid, outsize)
    elif ndim==3: return interpolate3D(data, grid, outsize)

#%%    
def interpolate1D(data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    n_channels = data.shape[1]
    width = data.shape[2]
    out_width = outsize[0]
    
    # Extract points
    x = grid[:,0].flatten()

    # Scale to domain
    x = x * (width-1)
    
    # Do sampling
    x0 = torch.floor(x).to(torch.int64); x1 = x0+1
    
    # Clip values
    x0 = torch.clamp(x0, 0, width-1)
    x1 = torch.clamp(x1, 0, width-1)
    
    # Batch effect
    batch_size = out_width
    batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t().flatten()
    
    # Index
    c0 = data[batch_idx, :, x0]
    c1 = data[batch_idx, :, x1]
    
    # Interpolation weights
    xd = (x-x0.to(torch.float32)).reshape((-1,1))
    
    # Do interpolation
    c = c0*(1-xd) + c1*xd
    
    # Reshape
    new_data = torch.reshape(c, (n_batch, out_width, n_channels))
    new_data = new_data.permute(0, 2, 1)
    return new_data.contiguous()
    
#%%    
def interpolate2D(data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    n_channels = data.shape[1]
    width = data.shape[2]
    height = data.shape[3]
    out_width, out_height = outsize
    
    # Extract points
    x = grid[:,0].flatten()
    y = grid[:,1].flatten()
    
    # Scale to domain
    x = x * (width-1)
    y = y * (height-1)
    
    # Do sampling
    x0 = torch.floor(x).to(torch.int64); x1 = x0+1
    y0 = torch.floor(y).to(torch.int64); y1 = y0+1
    
    # Clip values
    x0 = torch.clamp(x0, 0, width-1)
    x1 = torch.clamp(x1, 0, width-1)
    y0 = torch.clamp(y0, 0, height-1)
    y1 = torch.clamp(y1, 0, height-1)
    
    # Batch effect
    batch_size = out_width*out_height
    batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t().flatten()
    
    # Index
    c00 = data[batch_idx, :, x0, y0]
    c01 = data[batch_idx, :, x0, y1]
    c10 = data[batch_idx, :, x1, y0]
    c11 = data[batch_idx, :, x1, y1]
    
    # Interpolation weights
    xd = (x-x0.to(torch.float32)).reshape((-1,1))
    yd = (y-y0.to(torch.float32)).reshape((-1,1))
    
    # Do interpolation
    c0 = c00*(1-xd) + c10*xd
    c1 = c01*(1-xd) + c11*xd
    c = c0*(1-yd) + c1*yd
    
    # Reshape
    new_data = torch.reshape(c, (n_batch, out_height, out_width, n_channels))
    new_data = new_data.permute(0, 3, 2, 1)
    return new_data.contiguous()
    
#%%    
def interpolate3D(data, grid, outsize):
    # Problem size
    n_batch = data.shape[0]
    n_channels = data.shape[1]
    width = data.shape[2]
    height = data.shape[3]
    depth = data.shape[4]
    out_width, out_height, out_depth = outsize
    
    # Extract points
    x = grid[:,0].flatten()
    y = grid[:,1].flatten()
    z = grid[:,2].flatten()
    
    # Scale to domain
    x = x * (width-1)
    y = y * (height-1)
    z = z * (depth-1)
    
    # Do sampling
    x0 = torch.floor(x).to(torch.int64); x1 = x0+1
    y0 = torch.floor(y).to(torch.int64); y1 = y0+1
    z0 = torch.floor(z).to(torch.int64); z1 = z0+1
    
    # Clip values
    x0 = torch.clamp(x0, 0, width-1)
    x1 = torch.clamp(x1, 0, width-1)
    y0 = torch.clamp(y0, 0, height-1)
    y1 = torch.clamp(y1, 0, height-1)
    z0 = torch.clamp(z0, 0, depth-1)
    z1 = torch.clamp(z1, 0, depth-1)
    
    # Batch effect
    batch_size = out_width*out_height*out_depth
    #batch_idx = (torch.arange(n_batch)*batch_size).repeat(batch_size)
    batch_idx = torch.arange(n_batch).repeat(batch_size, 1).t().flatten()
    # Index
    c000 = data[batch_idx, :, x0, y0, z0]
    c001 = data[batch_idx, :, x0, y0, z1]
    c010 = data[batch_idx, :, x0, y1, z0]
    c011 = data[batch_idx, :, x0, y1, z1]
    c100 = data[batch_idx, :, x1, y0, z0]
    c101 = data[batch_idx, :, x1, y0, z1]
    c110 = data[batch_idx, :, x1, y1, z0]
    c111 = data[batch_idx, :, x1, y1, z1]
    
    # Interpolation weights
    xd = (x-x0.to(torch.float32)).reshape((-1,1))
    yd = (y-y0.to(torch.float32)).reshape((-1,1))
    zd = (z-z0.to(torch.float32)).reshape((-1,1))
   
    # Do interpolation
    c00 = c000*(1-xd) + c100*xd
    c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd
    c11 = c011*(1-xd) + c111*xd
    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd
    c = c0*(1-zd) + c1*zd
    
    # Reshape
    new_data = torch.reshape(c, (n_batch, out_depth, out_height, out_width, n_channels))
    new_data = new_data.permute(0, 4, 3, 2, 1)
    return new_data.contiguous()