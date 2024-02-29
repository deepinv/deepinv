from deepinv.utils import zeros_like
import torch


def check_conv(X_prev, X, it, crit_conv="residual", thres_conv=1e-3, verbose=False):
    if crit_conv == "residual":
        if isinstance(X_prev, dict):
            X_prev = X_prev["est"][0]
        if isinstance(X, dict):
            X = X["est"][0]
        crit_cur = (X_prev - X).norm() / (X.norm() + 1e-06)
    elif crit_conv == "cost":
        F_prev = X_prev["cost"]
        F = X["cost"]
        crit_cur = (F_prev - F).norm() / (F.norm() + 1e-06)
    else:
        raise ValueError("convergence criteria not implemented")
    if crit_cur < thres_conv:
        if verbose:
            print(
                f"Iteration {it}, current converge crit. = {crit_cur:.2E}, objective = {thres_conv:.2E} \r"
            )
        return True
    else:
        return False


def conjugate_gradient(A, b, max_iter=1e2, tol=1e-5):
    """
    Standard conjugate gradient algorithm.

    It solves the linear system :math:`Ax=b`, where :math:`A` is a (square) linear operator and :math:`b` is a tensor.

    For more details see: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param (callable) A: Linear operator as a callable function, has to be square!
    :param torch.Tensor b: input tensor
    :param int max_iter: maximum number of CG iterations
    :param float tol: absolute tolerance for stopping the CG algorithm.
    :return: torch.Tensor :math:`x` verifying :math:`Ax=b`.

    """

    def dot(s1, s2):
        dot = (s1.conj() * s2).flatten().sum()
        return dot

    x = zeros_like(b)

    r = b
    p = r
    rsold = dot(r, r)

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / dot(p, Ap)
        x = x + p * alpha
        r = r + Ap * (-alpha)
        rsnew = torch.real(dot(r, r))
        # print(rsnew.sqrt())
        if rsnew.sqrt() < tol:
            break
        p = r + p * (rsnew / rsold)
        rsold = rsnew

    return x


def gradient_descent(grad_f, x, step_size=1.0, max_iter=1e2, tol=1e-5):
    """
    Standard gradient descent algorithm`.

    :param callable grad_f: gradient of function to bz minimized as a callable function.
    :param torch.Tensor x: input tensor.
    :param torch.Tensor, float step_size: (constant) step size of the gradient descent algorithm.
    :param int max_iter: maximum number of iterations
    :param float tol: absolute tolerance for stopping the algorithm.
    :return: torch.Tensor :math:`x` minimizing :math:`f(x)`.

    """
    for i in range(int(max_iter)):
        x_prev = x
        x = x - grad_f(x) * step_size
        if check_conv(x_prev, x, i, thres_conv=tol):
            break
    return x
