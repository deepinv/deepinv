import torch


def default_preprocessing(y, physics):
    r"""
    Default preprocessing function for spectral methods.

    The output of the preprocessing function is given by:

    .. math::
        \max(1 - 1/y, -5).

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Instance of the physics modeling the forward matrix.

    :return: The preprocessing function values evaluated at y.
    """
    return torch.max(1 - 1 / y, torch.tensor(-5.0))


def correct_global_phase(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    threshold: float = 1e-5,
    verbose: bool = False,
) -> torch.Tensor:
    r"""
        Corrects the global phase of the reconstructed image.

    .. warning::

        Do not mix the order of the reconstructed and original images since this function modifies x_recon in place.


        The global phase shift is comptued per image and per channel as:

        .. math::
            e^{-i \phi} = \frac{\conj{\hat{x}} \cdot x}{|x|^2},

        where :math:`\conj{\hat{x}}` is the complex conjugate of the reconstructed image, :math:`x` is the reference image, and :math:`|x|^2` is the squared magnitude of the reference image.

        The global phase shift is then applied to the reconstructed image as:

        .. math::
            \hat{x} = \hat{x} \cdot e^{-i \phi},

        for the corresponding image and channel.

        :param torch.Tensor x_recon: Reconstructed image.
        :param torch.Tensor x: Original image.
        :param float threshold: Threshold to determine if the global phase shift is constant. Default is 1e-5.
        :param bool verbose: If True, prints information about the global phase shift. Default is False.

        :return: The corrected image.
    """
    assert x_recon.shape == x.shape, "The shapes of the images should be the same."
    assert (
        len(x_recon.shape) == 4
    ), "The images should be input with shape (N, C, H, W) "

    n_imgs = x_recon.shape[0]
    n_channels = x_recon.shape[1]

    for i in range(n_imgs):
        for j in range(n_channels):
            e_minus_phi = (x_recon[i, j].conj() * x[i, j]) / (x[i, j].abs() ** 2)
            if e_minus_phi.var() < threshold:
                if verbose:
                    print(f"Image {i}, channel {j} has a constant global phase shift.")
            else:
                if verbose:
                    print(f"Image {i}, channel {j} does not have a global phase shift.")
            e_minus_phi = e_minus_phi.mean()
            x_recon[i, j] = x_recon[i, j] * e_minus_phi

    return x_recon


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    r"""
    Compute the cosine similarity between two images.

    The cosine similarity is computed as:

    .. math::
        \text{cosine\_similarity} = \frac{a \cdot b}{\|a\| \cdot \|b\|}.

    The value range is [0,1], higher values indicate higher similarity.
    If one image is a scaled version of the other, i.e., :math:`a = c * b` where :math:`c` is a nonzero complex number, then the cosine similarity will be 1.

    :param torch.Tensor a: First image.
    :param torch.Tensor b: Second image.
    :return: The cosine similarity between the two images."""
    assert a.shape == b.shape
    a = a.flatten()
    b = b.flatten()
    norm_a = torch.sqrt(torch.dot(a.conj(), a).real)
    norm_b = torch.sqrt(torch.dot(b.conj(), b).real)
    return torch.abs(torch.dot(a.conj(), b)) / (norm_a * norm_b)


def spectral_methods(
    y: torch.Tensor,
    physics,
    x=None,
    n_iter=50,
    preprocessing=default_preprocessing,
    lamb=10.0,
    x_true=None,
    log: bool = False,
    log_metric=cosine_similarity,
    early_stop: bool = True,
    rtol: float = 1e-5,
    verbose: bool = False,
):
    r"""
    Utility function for spectral methods.

    This function runs the Spectral Methods algorithm to find the principal eigenvector of the regularized weighted covariance matrix:
    
    .. math::
        \begin{equation*}
        M = \conj{B} \text{diag}(T(y)) B + \lambda I,
        \end{equation*}
    
    where :math:`B` is the linear operator of the phase retrieval class, :math:`T(\cdot)` is a preprocessing function for the measurements, and :math:`I` is the identity matrix of corresponding dimensions. Parameter :math:`\lambda` tunes the strength of regularization.

    To find the principal eigenvector, the function runs power iteration which is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        x_{k+1} &= M x_k \\
        x_{k+1} &= \frac{x_{k+1}}{\|x_{k+1}\|},
        \end{aligned}
        \end{equation*}
  
    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Instance of the physics modeling the forward matrix.
    :param torch.Tensor x: Initial guess for the signals :math:`x_0`.
    :param int n_iter: Number of iterations.
    :param Callable preprocessing: Function to preprocess the measurements. Default is :math:`\max(1 - 1/x, -5)`.
    :param float lamb: Regularization parameter. Default is 10.
    :param bool log: Whether to log the metrics. Default is False.
    :param Callable log_metric: Metric to log. Default is cosine similarity.
    :param bool early_stop: Whether to early stop the iterations. Default is True.
    :param float rtol: Relative tolerance for early stopping. Default is 1e-5.
    :param bool verbose: If True, prints information in case of an early stop. Default is False.

    :return: The estimated signals :math:`x`.
    """
    if x is None:
        # always use randn for initial guess, never use rand!
        x = torch.randn(
            (y.shape[0],) + physics.input_shape,
            dtype=physics.dtype,
            device=physics.device,
        )

    if log is True:
        metrics = []

    #! estimate the norm of x using y
    #! for the i.i.d. case, we have norm(x) = sqrt(sum(y)/A_squared_mean)
    #! for the structured case, when the mean of the squared diagonal elements is 1, we have norm(x) = sqrt(sum(y)), otherwise y gets scaled by the mean to the power of number of layers
    norm_x = torch.sqrt(y.sum())

    x = x.to(torch.cfloat)
    # y should have mean 1
    y = y / torch.mean(y)
    diag_T = preprocessing(y, physics)
    diag_T = diag_T.to(torch.cfloat)
    for i in range(n_iter):
        x_new = physics.B(x)
        x_new = diag_T * x_new
        x_new = physics.B_adjoint(x_new)
        x_new = x_new + lamb * x
        x_new = x_new / torch.linalg.norm(x_new)
        if log:
            metrics.append(log_metric(x_new, x_true))
        if early_stop:
            if torch.linalg.norm(x_new - x) / torch.linalg.norm(x) < rtol:
                if verbose:
                    print(f"Power iteration early stopped at iteration {i}.")
                break
        x = x_new
    #! change the norm of x so that it matches the norm of true x
    x = x * norm_x
    if log:
        return x, metrics
    else:
        return x


def spectral_methods_wrapper(y, physics, n_iter=5000, **kwargs):
    r"""
    Wrapper for spectral methods.

    This function wrapper can be used as the custom_init option when building optimizers.

    :param torch.Tensor y: Measurements.
    :param deepinv.physics.Physics physics: Instance of the physics modeling the forward matrix.
    :param int n_iter: Number of iterations.

    :return: The estimated signals :math:`x` and :math:`z` packed in a dictionary.
    """
    x = spectral_methods(y, physics, n_iter=n_iter, **kwargs)
    z = x.detach().clone()
    return {"est": (x, z)}
