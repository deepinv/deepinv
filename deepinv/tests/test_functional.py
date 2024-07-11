#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:48:05 2024

@author: fsarron
"""

import deepinv as dinv
import torch
import pytest

dtype = torch.float64
device = "cpu"


def test_conv2d_adjointness():
    torch.manual_seed(0)
    size_im = ([5, 5], [6, 6], [5, 6], [6, 5])
    size_filt = ([3, 3], [4, 4], [3, 4], [4, 3])

    paddings = ("valid", "constant", "circular", "reflect", "replicate")

    for pad in paddings:
        for sim in size_im:
            for sfil in size_filt:
                x = torch.rand(sim)[None, None].to(dtype)
                h = torch.rand(sfil)[None, None].to(dtype)
                Ax = dinv.physics.functional.conv2d(x, h, padding=pad)
                y = torch.rand_like(Ax)
                Aty = dinv.physics.functional.conv_transpose2d(y, h, padding=pad)

                Axy = torch.sum(Ax * y)
                Atyx = torch.sum(Aty * x)

                print(torch.abs(Axy - Atyx))

                assert torch.abs(Axy - Atyx) < 1e-5


def test_conv3d_norm():
    torch.manual_seed(0)
    max_iter = 1000
    tol = 1e-6

    size_im = ([3, 5, 5, 5], [3, 6, 6, 6], [3, 5, 5, 6], [3, 5, 6, 5])
    size_filt = ([3, 3, 3], [4, 4, 4], [4, 3, 4], [3, 4, 3])

    paddings = ("circular",)

    for pad in paddings:
        print(pad)
        for sim in size_im:
            for sfil in size_filt:
                x = torch.randn(sim)[None].to(dtype).to(device)
                x /= torch.norm(x)
                h = torch.rand(sfil)[None, None].to(dtype).to(device)
                h /= h.sum()
                h = torch.zeros(1, 1, 5, 5, 5)
                h[:, :, 1:4, 1:4, 1:4] = 1
                h /= h.sum()

                zold = torch.zeros_like(x)
                for it in range(max_iter):
                    y = dinv.physics.functional.conv3d_fft(x, h, padding=pad)
                    y = dinv.physics.functional.conv_transpose3d_fft(y, h, padding=pad)
                    z = (
                        torch.matmul(x.conj().reshape(-1), y.reshape(-1))
                        / torch.norm(x) ** 2
                    )

                    rel_var = torch.norm(z - zold)
                    if rel_var < tol:
                        print(
                            f"Power iteration converged at iteration {it}, value={z.item():.2f}"
                        )
                        break
                    zold = z
                    x = y / torch.norm(y)


def test_conv3d_adjointness():
    torch.manual_seed(0)

    size_im = ([5, 5, 5], [6, 6, 6], [5, 5, 6], [5, 6, 5])
    size_filt = ([3, 3, 3], [4, 4, 4], [4, 3, 4], [3, 4, 3])

    paddings = ("valid", "circular")

    for pad in paddings:
        print(pad)
        for sim in size_im:
            for sfil in size_filt:
                # print(sim, sfil)
                x = torch.rand(sim)[None, None].to(dtype)
                h = torch.rand(sfil)[None, None].to(dtype)
                Ax = dinv.physics.functional.conv3d_fft(x, h, padding=pad)
                y = torch.rand_like(Ax)
                Aty = dinv.physics.functional.conv_transpose3d_fft(y, h, padding=pad)

                Axy = torch.sum(Ax * y)
                Atyx = torch.sum(Aty * x)

                assert torch.abs(Axy - Atyx) < 1e-3


# test_conv3d_norm()
# test_conv2d_adjointness()
# print('')
# print('')
# print('')
# print('')
# test_conv3d_adjointness()
