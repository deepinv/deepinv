from .deal_linearspline import LinearSpline


def get_spline_coefficients(model):
    coeffs_list = []
    for module in model.modules():
        if isinstance(module, LinearSpline):
            coeffs_list.append(module.coefficients)
    return coeffs_list


def get_spline_scaling_factors(model):
    scaling_factors_list = []
    for module in model.modules():
        if isinstance(module, LinearSpline):
            if module.apply_scaling is True:
                scaling_factors_list.append(module.scaling_factors)
    return scaling_factors_list


def get_no_spline_params(model):
    params_list = set(model.parameters())
    no_spline_params_list = (
        params_list
        - set(get_spline_coefficients(model))
        - set(get_spline_scaling_factors(model))
    )
    no_spline_params_list = list(no_spline_params_list)
    return no_spline_params_list


def get_total_tv2(model):
    tv2 = 0.0
    for module in model.modules():
        if isinstance(module, LinearSpline):
            tv2 += module.tv2()
    return tv2
