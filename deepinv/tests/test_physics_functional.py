#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:48:05 2024

@author: fsarron
"""
import pytest
import torch
import deepinv as dinv


def test_conv2d_adjointness(device):
    torch.manual_seed(0)

    nchannels = ((1, 1), (3, 1), (3, 3))

    for nchan_im, nchan_filt in nchannels:
        size_im = (
            [nchan_im, 5, 5],
            [nchan_im, 6, 6],
            [nchan_im, 5, 6],
            [nchan_im, 6, 5],
        )
        size_filt = (
            [nchan_filt, 3, 3],
            [nchan_filt, 4, 4],
            [nchan_filt, 3, 4],
            [nchan_filt, 4, 3],
        )

        paddings = ("valid", "constant", "circular", "reflect", "replicate")

        for pad in paddings:
            for sim in size_im:
                for sfil in size_filt:
                    x = torch.rand(sim)[None].to(device)
                    h = torch.rand(sfil)[None].to(device)
                    Ax = dinv.physics.functional.conv2d(x, h, padding=pad)
                    y = torch.rand_like(Ax)
                    Aty = dinv.physics.functional.conv_transpose2d(y, h, padding=pad)

                    Axy = torch.sum(Ax * y)
                    Atyx = torch.sum(Aty * x)

                    assert torch.abs(Axy - Atyx) < 1e-3


def test_conv3d_norm(device):
    torch.manual_seed(0)
    max_iter = 1000
    tol = 1e-6

    # Note : does not work for nchan_im, nchan_filt = (3, 3)
    nchannels = ((1, 1), (3, 1))
    paddings = ("circular",)

    for nchan_im, nchan_filt in nchannels:
        size_im = (
            [nchan_im, 5, 5, 5],
            [nchan_im, 6, 6, 6],
            [nchan_im, 5, 5, 6],
            [nchan_im, 5, 6, 5],
        )
        size_filt = (
            [nchan_filt, 3, 3, 3],
            [nchan_filt, 4, 4, 4],
            [nchan_filt, 4, 3, 4],
            [nchan_filt, 3, 4, 3],
        )

        for pad in paddings:
            for sim in size_im:
                for sfil in size_filt:
                    x = torch.randn(sim)[None].to(device)
                    x /= torch.norm(x)
                    h = torch.rand(sfil)[None].to(device)
                    h /= h.sum()

                    zold = torch.zeros_like(x)
                    for it in range(max_iter):
                        y = dinv.physics.functional.conv3d_fft(x, h, padding=pad)
                        y = dinv.physics.functional.conv_transpose3d_fft(
                            y, h, padding=pad
                        )
                        z = (
                            torch.matmul(x.conj().reshape(-1), y.reshape(-1))
                            / torch.norm(x) ** 2
                        )

                        rel_var = torch.norm(z - zold)
                        if rel_var < tol:
                            break
                        zold = z
                        x = y / torch.norm(y)

                    assert torch.abs(zold.item() - torch.ones(1)) < 1e-2


def test_conv3d_adjointness(device):
    torch.manual_seed(0)

    nchannels = ((1, 1), (3, 1), (3, 3))

    for nchan_im, nchan_filt in nchannels:
        size_im = (
            [nchan_im, 5, 5, 5],
            [nchan_im, 6, 6, 6],
            [nchan_im, 5, 5, 6],
            [nchan_im, 5, 6, 5],
        )
        size_filt = (
            [nchan_filt, 3, 3, 3],
            [nchan_filt, 4, 4, 4],
            [nchan_filt, 4, 3, 4],
            [nchan_filt, 3, 4, 3],
        )

        paddings = ("valid", "circular")

        for pad in paddings:
            for sim in size_im:
                for sfil in size_filt:
                    # print(sim, sfil)
                    x = torch.rand(sim)[None].to(device)
                    h = torch.rand(sfil)[None].to(device)
                    Ax = dinv.physics.functional.conv3d_fft(x, h, padding=pad)
                    y = torch.rand_like(Ax)
                    Aty = dinv.physics.functional.conv_transpose3d_fft(
                        y, h, padding=pad
                    )

                    Axy = torch.sum(Ax * y)
                    Atyx = torch.sum(Aty * x)

                    assert torch.abs(Axy - Atyx) < 1e-3


@pytest.mark.parametrize("kernel", ["cubic", "gaussian"])
@pytest.mark.parametrize("scale", [2, 0.5])
@pytest.mark.parametrize("antialiasing", [True, False])
def test_imresize(kernel, scale, antialiasing):
    sigma = 2
    img_size = (1, 64, 64)
    x = torch.randn(1, *img_size)
    y = dinv.physics.functional.imresize_matlab(
        x,
        scale=scale,
        kernel=kernel,
        sigma=sigma,
        padding_type="reflect",
        antialiasing=antialiasing,
    )
    assert y.shape == (
        1,
        img_size[0],
        int(img_size[1] * scale),
        int(img_size[2] * scale),
    )


def test_imresize_div2k():
    x = dinv.utils.load_example("div2k_valid_hr_0877.png") * 255.0
    y = dinv.utils.load_example("div2k_valid_lr_bicubic_0877x4.png") * 255.0
    y2 = dinv.physics.functional.imresize_matlab(x, scale=1 / 4).round()
    assert dinv.metric.PSNR()(y2 / 255.0, y / 255.0) > 59
