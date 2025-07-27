def get_GSPnP_params(problem, noise_level_img):
    r"""
    Default parameters for the GSPnP Plug-and-Play algorithm.

    :param str problem: Type of inverse-problem problem to solve. Can be ``deblur``, ``super-resolution``, or ``inpaint``.
    :param float noise_level_img: Noise level of the input image.
    """
    if problem == "deblur":
        max_iter = 500
        sigma_denoiser = 1.8 * noise_level_img
        lamb = 0.1
    elif problem == "super-resolution":
        max_iter = 500
        sigma_denoiser = 2.0 * noise_level_img
        lamb = 0.065
    elif problem == "inpaint":
        max_iter = 100
        sigma_denoiser = 10.0 / 255
        lamb = 0.1
    else:
        raise ValueError("parameters unknown with this degradation")
    stepsize = 1 / lamb
    return lamb, sigma_denoiser, stepsize, max_iter
