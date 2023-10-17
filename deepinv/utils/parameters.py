import numpy as np


def get_DPIR_params(noise_level_img):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    max_iter = 8
    s1 = 49.0 / 255.0
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    lamb = 1 / 0.23
    return lamb, list(sigma_denoiser), list(stepsize), max_iter


def get_GSPnP_params(problem, noise_level_img):
    r"""
    Default parameters for the GSPnP Plug-and-Play algorithm.

    :param str problem: Type of inverse-problem problem to solve. Can be ``deblur``, ``super-resolution``, or ``inpaint``.
    :param float noise_level_img: Noise level of the input image.
    """
    if problem == "deblur":
        max_iter = 500
        sigma_denoiser = 1.8 * noise_level_img
        lamb = 1 / 0.1
    elif problem == "super-resolution":
        max_iter = 500
        sigma_denoiser = 2.0 * noise_level_img
        lamb = 1 / 0.065
    elif problem == "inpaint":
        max_iter = 100
        sigma_denoiser = 10.0 / 255
        lamb = 1 / 0.1
    else:
        raise ValueError("parameters unknown with this degradation")
    stepsize = 1.0
    return lamb, sigma_denoiser, stepsize, max_iter
