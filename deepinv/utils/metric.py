import warnings


def deprecate(msg):
    warnings.warn(msg, DeprecationWarning, stacklevel=1)
    raise NotImplementedError(msg)


def norm(a):
    deprecate("This function is deprecated. Use dinv.metric.functional.norm instead.")


def cal_angle(a, b):
    deprecate("This function is deprecated and will be removed from a future release.")


def cal_psnr(a, b, max_pixel=1.0, normalize=False, mean_batch=True, to_numpy=True):
    deprecate("This function is deprecated. Use dinv.metric.cal_psnr instead.")


def cal_mse(a, b):
    deprecate("This function is deprecated. Use dinv.metric.cal_mse instead.")


def cal_psnr_complex(a, b):
    deprecate(
        "This function is deprecated. Use dinv.metric.PSNR(complex_abs=True) instead."
    )


def complex_abs(data, dim=1, keepdim=True):
    deprecate("This function is deprecated. Use dinv.metric.complex_abs instead.")


def norm_psnr(a, b, complex=False):
    deprecate(
        "This function is deprecated. Use dinv.metric.PSNR(norm_inputs='min_max', complex_abs=complex) instead."
    )
