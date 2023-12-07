from deepinv.utils import zeros_like


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
    Standard conjugate gradient algorithm to solve Ax=b
        see: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    :param A: Linear operator as a callable function, has to be square!
    :param b: input tensor
    :param max_iter: maximum number of CG iterations
    :param tol: absolute tolerance for stopping the CG algorithm.
    :return: torch tensor x verifying Ax=b

    """

    def dot(s1, s2):
        dot = (s1 * s2).flatten().sum()
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
        rsnew = dot(r, r)
        # print(rsnew.sqrt())
        if rsnew.sqrt() < tol:
            break
        p = r + p * (rsnew / rsold)
        rsold = rsnew

    return x


def gradient_descent(grad_f, x, step_size=1.0, max_iter=1e2, tol=1e-5):
    """
    Standard gradient descent algorithm to solve min_x f(x)
    :param grad_f: gradient of function to bz minimized as a callable function.
    :param x: input tensor
    :param step_size: (constant) step size of the gradient descent algorithm.
    :param max_iter: maximum number of iterations
    :param tol: absolute tolerance for stopping the algorithm.
    :return: torch tensor x verifying min_x f(x)

    """
    for i in range(int(max_iter)):
        x_prev = x
        x = x - grad_f(x) * step_size
        if check_conv(x_prev, x, i, thres_conv=tol):
            break
    return x
