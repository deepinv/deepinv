import torch


def spectral_methods(
    y: torch.Tensor,
    physics,
    x=None,
    n_iter=50,
    preprocessing=lambda y: torch.max(1 - 1 / y, torch.tensor(-5.0)),
    lamb=10.0,
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
    :param deepinv.physics physics: Instance of the physics modeling the forward matrix.
    :param torch.Tensor x: Initial guess for the signals :math:`x_0`.
    :param int n_iter: Number of iterations.
    :param function preprocessing: Function to preprocess the measurements. Default is :math:`\max(1 - 1/x, -5)`.
    :param float lamb: Regularization parameter. Default is 10.

    :return: The estimated signals :math:`x`.
    """
    if x is None:
        x = torch.randn((y.shape[0],) + physics.img_shape, dtype=physics.dtype)
    x = x.to(torch.cfloat)
    x = x / torch.linalg.norm(x)
    # y should have mean 1
    y = y / torch.mean(y)
    diag_T = preprocessing(y)
    diag_T = diag_T.to(torch.cfloat)
    for _ in range(n_iter):
        res = physics.B(x)
        res = diag_T * res
        res = physics.B_adjoint(res)
        x = res + lamb * x
        x = x / torch.linalg.norm(x)
    return x


def correct_global_phase(
    x_hat: torch.Tensor, x: torch.Tensor, verbose=False
) -> torch.Tensor:
    r"""
    Corrects the global phase of the reconstructed image.

    :param torch.Tensor x_hat: Reconstructed image.
    :param torch.Tensor x: Reference image.
    :param bool verbose: If True, prints whether the global phase shift is constant or not.

    :return: The corrected image.
    """
    assert x_hat.shape == x.shape, "The shapes of the images should be the same."
    assert len(x_hat.shape) == 4, "The images should be input with shape (N, C, H, W) "

    n_imgs = x_hat.shape[0]
    n_channels = x_hat.shape[1]

    for i in range(n_imgs):
        for j in range(n_channels):
            e_minus_phi = (x_hat[i, j].conj() * x[i, j]) / (x[i, j].abs() ** 2)
            if verbose:
                if e_minus_phi.var() < 1e-3:
                    print(f"Image {i}, channel {j} has a constant global phase shift.")
                else:
                    print(f"Image {i}, channel {j} does not have a global phase shift.")
            e_minus_phi = e_minus_phi.mean()
            x_hat[i, j] = x_hat[i, j] * e_minus_phi

    return x_hat


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
