# -*- coding: utf-8 -*-
import numpy as np
import torch
import deepinv.diffunet.utils.utils_image as util
from functools import partial

from deepinv.diffunet.guided_diffusion.script_util import add_dict_to_argparser
import argparse

"""
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
"""


def test_mode(
    model_fn,
    model_diffusion,
    L,
    mode=0,
    refield=32,
    min_size=256,
    sf=1,
    modulo=1,
    noise_level=0,
    vec_t=None,
    model_out_type="pred_xstart",
    diffusion=None,
    ddim_sample=False,
    alphas_cumprod=None,
):
    """
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some testing modes
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (0) normal: test(model, L)
    # (1) pad: test_pad(model, L, modulo=16)
    # (2) split: test_split(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (3) x8: test_x8(model, L, modulo=1)
    # (4) split and x8: test_split_x8(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (4) split only once: test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # ---------------------------------------
    """

    model = partial(
        model_fn,
        model_diffusion=model_diffusion,
        diffusion=diffusion,
        ddim_sample=False,
        alphas_cumprod=alphas_cumprod,
    )

    if mode == 0:
        E = test(model, L, noise_level, vec_t, model_out_type)
    elif mode == 1:
        E = test_pad(model, L, modulo, noise_level, vec_t, model_out_type)
    elif mode == 2:
        E = test_split(
            model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type
        )
    elif mode == 3:
        E = test_x8(model, L, modulo, noise_level, vec_t, model_out_type)
    elif mode == 4:
        E = test_split_x8(
            model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type
        )
    elif mode == 5:
        E = test_onesplit(
            model, L, refield, min_size, sf, modulo, noise_level, vec_t, model_out_type
        )
    return E


"""
# ---------------------------------------
# normal (0)
# ---------------------------------------
"""


def test(model, L, noise_level=15, vec_t=None, model_out_type="pred_xstart"):
    E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
    return E


"""
# ---------------------------------------
# pad (1)
# ---------------------------------------
"""


def test_pad(
    model, L, modulo=16, noise_level=15, vec_t=None, model_out_type="pred_xstart"
):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h / modulo) * modulo - h)
    paddingRight = int(np.ceil(w / modulo) * modulo - w)
    L = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(L)
    E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
    E = E[..., :h, :w]
    return E


"""
# ---------------------------------------
# split (function)
# ---------------------------------------
"""


def test_split_fn(
    model,
    L,
    refield=32,
    min_size=256,
    sf=1,
    modulo=1,
    noise_level=15,
    vec_t=None,
    model_out_type="pred_xstart",
):
    """
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    """
    h, w = L.size()[-2:]
    if h * w <= min_size**2:
        L = torch.nn.ReplicationPad2d(
            (
                0,
                int(np.ceil(w / modulo) * modulo - w),
                0,
                int(np.ceil(h / modulo) * modulo - h),
            )
        )(L)
        E = model(L, noise_level, vec_t=vec_t, model_out_type=model_out_type)
        E = E[..., : h * sf, : w * sf]
    else:
        top = slice(0, (h // 2 // refield + 1) * refield)
        bottom = slice(h - (h // 2 // refield + 1) * refield, h)
        left = slice(0, (w // 2 // refield + 1) * refield)
        right = slice(w - (w // 2 // refield + 1) * refield, w)
        Ls = [
            L[..., top, left],
            L[..., top, right],
            L[..., bottom, left],
            L[..., bottom, right],
        ]

        if h * w <= 4 * (min_size**2):
            Es = [
                model(Ls[i], noise_level, vec_t=vec_t, model_out_type=model_out_type)
                for i in range(4)
            ]
        else:
            Es = [
                test_split_fn(
                    model,
                    Ls[i],
                    refield=refield,
                    min_size=min_size,
                    sf=sf,
                    modulo=modulo,
                    noise_level=noise_level,
                    vec_t=vec_t,
                    model_out_type=model_out_type,
                )
                for i in range(4)
            ]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., : h // 2 * sf, : w // 2 * sf] = Es[0][..., : h // 2 * sf, : w // 2 * sf]
        E[..., : h // 2 * sf, w // 2 * sf : w * sf] = Es[1][
            ..., : h // 2 * sf, (-w + w // 2) * sf :
        ]
        E[..., h // 2 * sf : h * sf, : w // 2 * sf] = Es[2][
            ..., (-h + h // 2) * sf :, : w // 2 * sf
        ]
        E[..., h // 2 * sf : h * sf, w // 2 * sf : w * sf] = Es[3][
            ..., (-h + h // 2) * sf :, (-w + w // 2) * sf :
        ]
    return E


def test_onesplit(
    model,
    L,
    refield=32,
    min_size=256,
    sf=1,
    modulo=1,
    noise_level=15,
    vec_t=None,
    model_out_type="pred_xstart",
):
    """
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256X256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    """
    h, w = L.size()[-2:]

    top = slice(0, (h // 2 // refield + 1) * refield)
    bottom = slice(h - (h // 2 // refield + 1) * refield, h)
    left = slice(0, (w // 2 // refield + 1) * refield)
    right = slice(w - (w // 2 // refield + 1) * refield, w)
    Ls = [
        L[..., top, left],
        L[..., top, right],
        L[..., bottom, left],
        L[..., bottom, right],
    ]
    Es = [
        model(Ls[i], noise_level, vec_t=vec_t, model_out_type=model_out_type)
        for i in range(4)
    ]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., : h // 2 * sf, : w // 2 * sf] = Es[0][..., : h // 2 * sf, : w // 2 * sf]
    E[..., : h // 2 * sf, w // 2 * sf : w * sf] = Es[1][
        ..., : h // 2 * sf, (-w + w // 2) * sf :
    ]
    E[..., h // 2 * sf : h * sf, : w // 2 * sf] = Es[2][
        ..., (-h + h // 2) * sf :, : w // 2 * sf
    ]
    E[..., h // 2 * sf : h * sf, w // 2 * sf : w * sf] = Es[3][
        ..., (-h + h // 2) * sf :, (-w + w // 2) * sf :
    ]
    return E


"""
# ---------------------------------------
# split (2)
# ---------------------------------------
"""


def test_split(
    model,
    L,
    refield=32,
    min_size=256,
    sf=1,
    modulo=1,
    noise_level=15,
    vec_t=None,
    model_out_type="pred_xstart",
):
    E = test_split_fn(
        model,
        L,
        refield=refield,
        min_size=min_size,
        sf=sf,
        modulo=modulo,
        noise_level=noise_level,
        vec_t=vec_t,
        model_out_type=model_out_type,
    )
    return E


"""
# ---------------------------------------
# x8 (3)
# ---------------------------------------
"""


def test_x8(
    model, L, modulo=1, noise_level=15, vec_t=None, model_out_type="pred_xstart"
):
    E_list = [
        test_pad(
            model,
            util.augment_img_tensor(L, mode=i),
            modulo=modulo,
            noise_level=noise_level,
            vec_t=vec_t,
            model_out_type=model_out_type,
        )
        for i in range(8)
    ]
    for i in range(len(E_list)):
        if i == 3 or i == 5:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=8 - i)
        else:
            E_list[i] = util.augment_img_tensor(E_list[i], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


"""
# ---------------------------------------
# split and x8 (4)
# ---------------------------------------
"""


def test_split_x8(
    model,
    L,
    refield=32,
    min_size=256,
    sf=1,
    modulo=1,
    noise_level=15,
    vec_t=None,
    model_out_type="pred_xstart",
):
    E_list = [
        test_split_fn(
            model,
            util.augment_img_tensor(L, mode=i),
            refield=refield,
            min_size=min_size,
            sf=sf,
            modulo=modulo,
            noise_level=noise_level,
            vec_t=vec_t,
            model_out_type=model_out_type,
        )
        for i in range(8)
    ]
    for k, i in enumerate(range(len(E_list))):
        if i == 3 or i == 5:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=8 - i)
        else:
            E_list[k] = util.augment_img_tensor(E_list[k], mode=i)
    output_cat = torch.stack(E_list, dim=0)
    E = output_cat.mean(dim=0, keepdim=False)
    return E


# ----------------------------------------
# wrap diffusion model
# ----------------------------------------


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def model_fn(
    x,
    noise_level,
    model_diffusion,
    vec_t=None,
    model_out_type="pred_xstart",
    diffusion=None,
    ddim_sample=False,
    alphas_cumprod=None,
    **model_kwargs,
):

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = torch.div(
        sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
    )  # equivalent noise sigma on image

    # time step corresponding to noise level
    if not torch.is_tensor(vec_t):
        t_step = find_nearest(reduced_alpha_cumprod, (noise_level / 255.0))
        vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)
        # timesteps = torch.linspace(1, 1e-3, num_train_timesteps, device=device)
        # t = timesteps[t_step]
    if not ddim_sample:
        out = diffusion.p_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
        )
    else:
        out = diffusion.ddim_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            eta=0,
        )

    if model_out_type == "pred_x_prev_and_start":
        return out["sample"], out["pred_xstart"]
    elif model_out_type == "pred_x_prev":
        out = out["sample"]
    elif model_out_type == "pred_xstart":
        out = out["pred_xstart"]
    elif model_out_type == "epsilon":
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
    elif model_out_type == "score":
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
        out = -out / beta_prod_t ** (0.5)

    return out


"""
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
# _^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_
# ^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^-^_^
"""


"""
# ---------------------------------------
# print
# ---------------------------------------
"""


# -------------------
# print model
# -------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# -------------------
# print params
# -------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


"""
# ---------------------------------------
# information
# ---------------------------------------
"""


# -------------------
# model inforation
# -------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# -------------------
# params inforation
# -------------------
def info_params(model):
    msg = describe_params(model)
    return msg


"""
# ---------------------------------------
# description
# ---------------------------------------
"""


# ----------------------------------------------
# model name and total number of parameters
# ----------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = "\n"
    msg += "models name: {}".format(model.__class__.__name__) + "\n"
    msg += (
        "Params number: {}".format(sum(map(lambda x: x.numel(), model.parameters())))
        + "\n"
    )
    msg += "Net structure:\n{}".format(str(model)) + "\n"
    return msg


# ----------------------------------------------
# parameters description
# ----------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = "\n"
    msg += (
        " | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}".format(
            "mean", "min", "max", "std", "param_name"
        )
        + "\n"
    )
    for name, param in model.state_dict().items():
        if not "num_batches_tracked" in name:
            v = param.data.clone().float()
            msg += (
                " | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}".format(
                    v.mean(), v.min(), v.max(), v.std(), name
                )
                + "\n"
            )
    return msg


# ----------------------------------------
# load model
# ----------------------------------------


def create_argparser(model_config):
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        diffusion_steps=1000,
        noise_schedule="linear",
        num_head_channels=64,
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,
        num_heads=4,
        num_heads_upsample=-1,
        use_new_attention_order=False,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
        dropout=0.1,
    )
    defaults.update(model_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def grad_and_value(operator, x, x_hat, measurement):
    difference = measurement - operator(x_hat)
    norm = torch.linalg.norm(difference)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
    return norm_grad, norm


if __name__ == "__main__":

    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )

        def forward(self, x):
            x = self.conv(x)
            return x

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    print_model(model)
    print_params(model)
    x = torch.randn((2, 3, 400, 400))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for mode in range(5):
            y = test_mode(model, x, mode)
            print(y.shape)
