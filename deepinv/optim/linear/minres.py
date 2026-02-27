import torch
from torch import Tensor
from deepinv.utils.tensorlist import TensorList
from .utils import dot


def minres(
    A,
    b: Tensor,
    init=None,
    max_iter: int = 1e2,
    tol=1e-5,
    eps=1e-6,
    parallel_dim=0,
    verbose=False,
    precon=lambda x: x.clone(),
):
    """
    Minimal Residual Method for solving symmetric equations.

    Solves :math:`Ax=b` with :math:`A` symmetric using the MINRES algorithm in :cite:t:`paige1975solution`

    The method assumes that :math:`A` is hermite.
    For more details see: https://en.wikipedia.org/wiki/Minimal_residual_method

    Based on https://github.com/cornellius-gp/linear_operator
    Modifications and simplifications for compatibility with deepinverse

    :param Callable A: Linear operator as a callable function.
    :param torch.Tensor b: input tensor of shape (B, ...)
    :param torch.Tensor init: Optional initial guess.
    :param int max_iter: maximum number of MINRES iterations.
    :param float tol: absolute tolerance for stopping the MINRES algorithm.
    :param None, int, list[int] parallel_dim: dimensions to be considered as batch dimensions. If None, all dimensions are considered as batch dimensions.
    :param bool verbose: Output progress information in the console.
    :param Callable precon: preconditioner is a callable function (not tested). Must be positive definite
    :return: (:class:`torch.Tensor`) :math:`x` of shape (B, ...)
    """

    if isinstance(parallel_dim, int):
        parallel_dim = [parallel_dim]
    if parallel_dim is None:
        parallel_dim = []

    if isinstance(b, TensorList):
        dim = [i for i in range(b[0].ndim) if i not in parallel_dim]
    else:
        dim = [i for i in range(b.ndim) if i not in parallel_dim]

    # Rescale b
    b_norm = torch.linalg.vector_norm(b, dim=dim, keepdim=True, ord=2)
    b_is_zero = b_norm < 1e-10
    b_norm = b_norm.masked_fill(b_is_zero, 1)
    b = b / b_norm

    # Create space for matmul product, solution
    if init is not None:
        solution = init / b_norm
    else:
        solution = torch.zeros(b.shape, dtype=b.dtype, device=b.device)

    # Variables for Lanczos terms
    zvec_prev2 = torch.zeros(solution.shape, device=b.device)  # r_(k-1) in wiki
    zvec_prev1 = b - A(solution)  # r_k in wiki
    qvec_prev1 = precon(zvec_prev1)
    alpha_curr = torch.zeros(b.shape, dtype=b.dtype, device=b.device)
    alpha_curr = torch.linalg.vector_norm(alpha_curr, dim=dim, keepdim=True, ord=2)
    beta_prev = torch.abs(dot(zvec_prev1, qvec_prev1, dim=dim).sqrt()).clamp_min(eps)

    # Divide by beta_prev
    zvec_prev1 = zvec_prev1 / beta_prev
    qvec_prev1 = qvec_prev1 / beta_prev

    # Variables for the QR rotation
    # 1) Components of the Givens rotations
    cos_prev2 = torch.ones(alpha_curr.shape, dtype=b.dtype, device=b.device)
    sin_prev2 = torch.zeros(alpha_curr.shape, dtype=b.dtype, device=b.device)
    cos_prev1 = cos_prev2
    sin_prev1 = sin_prev2

    # Variables for the solution updates
    # 1) The "search" vectors of the solution
    # Equivalent to the vectors of Q R^{-1}, where Q is the matrix of Lanczos vectors and
    # R is the QR factor of the tridiagonal Lanczos matrix.
    search_prev2 = torch.zeros_like(solution)
    search_prev1 = torch.zeros_like(solution)
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T b, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    scale_prev = beta_prev

    # Terms for checking for convergence
    solution_norm = torch.linalg.vector_norm(solution, dim=dim, ord=2).unsqueeze(-1)
    search_update_norm = torch.zeros_like(solution_norm)

    # Perform iterations
    flag = True
    for i in range(int(max_iter)):
        # Perform matmul
        prod = A(qvec_prev1)

        # Get next Lanczos terms
        # --> alpha_curr, beta_curr, qvec_curr
        alpha_curr = dot(prod, qvec_prev1, dim=dim)
        prod = prod - alpha_curr * zvec_prev1 - beta_prev * zvec_prev2
        qvec_curr = precon(prod)

        beta_curr = torch.abs(dot(prod, qvec_curr, dim=dim).sqrt()).clamp_min(eps)

        prod = prod / beta_curr
        qvec_curr = qvec_curr / beta_curr

        # Perform JIT-ted update
        ###########################################
        # Start givens rotation
        # Givens rotation from 2 steps ago
        subsub_diag_term = sin_prev2 * beta_prev
        sub_diag_term = cos_prev2 * beta_prev

        # Givens rotation from 1 step ago
        diag_term = alpha_curr * cos_prev1 - sin_prev1 * sub_diag_term
        sub_diag_term = sub_diag_term * cos_prev1 + sin_prev1 * alpha_curr

        # 3) Compute next Givens terms
        radius_curr = torch.sqrt(diag_term * diag_term + beta_curr * beta_curr)
        cos_curr = diag_term / radius_curr
        sin_curr = beta_curr / radius_curr
        # 4) Apply current Givens rotation
        diag_term = diag_term * cos_curr + sin_curr * beta_curr

        # Update the solution
        # --> search_curr, scale_curr solution
        # 1) Apply the latest Givens rotation to the Lanczos-b ( ||b|| e_1 )
        # This is getting the scale terms for the "search" vectors
        scale_curr = -scale_prev * sin_curr
        # 2) Get the new search vector
        search_curr = qvec_prev1 - sub_diag_term * search_prev1
        search_curr = (search_curr - subsub_diag_term * search_prev2) / diag_term

        # 3) Update the solution
        search_update = search_curr * scale_prev * cos_curr
        solution = solution + search_update
        ###########################################

        # Check convergence criterion
        search_update_norm = torch.linalg.vector_norm(
            search_update, dim=dim, ord=2
        ).unsqueeze(-1)
        solution_norm = torch.linalg.vector_norm(solution, dim=dim, ord=2).unsqueeze(-1)
        if (search_update_norm / solution_norm).max().item() < tol:
            if verbose:
                print("MINRES converged at iteration", i + 1)
            flag = False
            break

        # Update terms for next iteration
        # Lanczos terms
        zvec_prev2, zvec_prev1 = zvec_prev1, prod
        qvec_prev1 = qvec_curr
        beta_prev = beta_curr
        # Givens rotations terms
        cos_prev2, cos_prev1 = cos_prev1, cos_curr
        sin_prev2, sin_prev1 = sin_prev1, sin_curr
        # Search vector terms
        search_prev2, search_prev1 = search_prev1, search_curr
        scale_prev = scale_curr

    # For b-s that are close to zero, set them to zero
    solution = solution.masked_fill(b_is_zero, 0)
    if flag and verbose:
        print(f"MINRES did not converge in {i} iterations!")
    return solution * b_norm
