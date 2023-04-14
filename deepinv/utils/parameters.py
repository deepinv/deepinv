import numpy as np


def get_GSPnP_params(problem, noise_level_img, k_index=0):
    if problem == 'deblur' :
        max_iter = 500
        sigma_denoiser = 1.8 * noise_level_img
        if k_index == 8: # Uniform blur
            lamb = 1/0.075
        elif k_index == 9:  # Gaussian blur
            lamb = 1/0.075
        else : # Motion blur
            lamb = 1/0.1
    elif problem == 'super_resolution' :
        max_iter = 500
        sigma_denoiser = 2. * noise_level_img
        lamb = 1/0.065
    elif problem == 'inpaint' :
        max_iter = 100
        sigma_denoiser = 10./255
        lamb = 1/0.1
    else :
        raise ValueError('parameters unknown with this degradation')
    stepsize = 1.
    return lamb, sigma_denoiser, stepsize, max_iter

def get_DPIR_params(noise_level_img):
    max_iter = 8
    s1 = 49.0 / 255.
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(np.float32)
    stepsize = (sigma_denoiser/max(0.01,noise_level_img))**2
    lamb = 1/0.23
    return lamb, list(sigma_denoiser), list(stepsize), max_iter
