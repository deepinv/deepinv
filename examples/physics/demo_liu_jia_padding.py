import deepinv as dinv

import numpy as np
import scipy
from scipy import fftpack
import torch

from math import cos, sin
from numpy import zeros, ones, prod, array, pi, log, min, mod, arange, sum, mgrid, exp, pad, round
from numpy.random import randn, rand
from scipy.signal import convolve2d
import cv2
import random

def liu_jia_pad(img, img_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):

    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha*2+H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w)/(H_w-1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1, 0] + a*r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1, -1] + a*r_A[-alpha, -1]

    r_B = np.zeros((H, alpha*2+W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w)/(W_w-1)
    r_B[0, alpha:-alpha] = (1-a)*r_B[0, alpha-1] + a*r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1, alpha-1] + a*r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha-1:])
        r_A[alpha-1:, :] = A2
        r_B[:, alpha-1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1, :])
        r_A[alpha-1:-alpha+1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha-1:-alpha+1])
        r_B[:, alpha-1:-alpha+1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:])
        r_C[alpha-1:, alpha-1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1])
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2
    C = r_C
    # return C
    A = A[alpha-1:-alpha-1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H)-1
    k = np.arange(2, W)-1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4*boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k+1)] + boundary_image[np.ix_(j, k-1)] + boundary_image[np.ix_(j-1, k)] + boundary_image[np.ix_(j+1, k)]

    del(j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del(f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1]
    del(f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0)/2
    else:
        tt = fftpack.dst(f2, type=1)/2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0)/2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1)/2)
    del(f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W-1), np.arange(1, H-1))
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2)

    # divide
    f3 = f2sin/denom
    del(f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3*2, type=1, axis=1)/(2*(f3.shape[1]+1))
    else:
        tt = fftpack.idst(f3*2, type=1, axis=0)/(2*(f3.shape[0]+1))
    del(f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1)/(2*(tt.shape[0]+1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2, type=1, axis=0)/(2*(tt.shape[1]+1)))
    del(tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct


device = "cpu"
x = dinv.utils.load_example("butterfly.png", img_size=64).to(device)

# Define blur kernel and physics
kernel = torch.tensor([[1/16, 2/16, 1/16],
                           [2/16, 4/16, 2/16],
                           [1/16, 2/16, 1/16]], device=device)
kernel /= kernel.sum()
kernel = kernel.unsqueeze(0).unsqueeze(0)
physics = dinv.physics.Blur(filter=kernel, padding="valid")
y = physics(x)

# Crop for comparison
if kernel.shape[-2] % 2 != 1 or kernel.shape[-1] % 2 != 1:
    raise ValueError("Kernel size is expected to be odd")

margin = (
    (kernel.shape[-2] - 1) // 2,
    (kernel.shape[-1] - 1) // 2,
)
x = x[..., margin[0]: -margin[0], margin[1]: -margin[1]]

# Liu-Jia Padding
H, W = y.shape[-2:]
padding = (H // 4, W // 4)
y = liu_jia_pad(y.squeeze(0).permute(1, 2, 0).cpu().numpy(), (H + 2 * padding[0], W + 2 * padding[1]))
y = torch.from_numpy(y).permute(2, 0, 1).unsqueeze(0)
y = y.roll(shifts=padding, dims=(2, 3))

# Deconvolution
# 1. Pad k to make it the size of y with the central tap at (0,0)
k = torch.nn.functional.pad(
    kernel,
    (
        0,
        y.shape[-1] - kernel.shape[-1],
        0,
        y.shape[-2] - kernel.shape[-2],
    ),
)
k = k.roll(shifts=(-(kernel.shape[-2] // 2), -(kernel.shape[-1] // 2)), dims=(2, 3))
# 2. Compute the OTF
otf = torch.fft.fft2(k)
# 3. Compute the DFT of y
x_hat = torch.fft.fft2(y)
# 4. Apply the inverse filter formula
x_hat = x_hat / (otf + 1e-3)
# 5. Compute the inverse DFT
x_hat = torch.fft.ifft2(x_hat).real
# 6. Clip and quantize
x_hat = torch.clamp(x_hat, 0, 1)
x_hat = torch.round(x_hat * 255) / 255

# Cropping
margin = (
    (y.shape[-2] - H) // 2,
    (y.shape[-1] - W) // 2,
)
y = y[..., margin[0]: -margin[0], margin[1]: -margin[1]]
x_hat = x_hat[..., margin[0]: -margin[0], margin[1]: -margin[1]]

if x.shape != y.shape:
    raise ValueError("Shapes do not match after cropping")

psnr_fn = dinv.metric.PSNR()
psnr = psnr_fn(y, x).item()
psnr_x_hat = psnr_fn(x_hat ,x).item()

dinv.utils.plot([x, y, x_hat], ["x", f"y ({psnr:.1f} dB)", f"x_hat ({psnr_x_hat:.1f} dB)"])
