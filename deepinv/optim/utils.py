import torch
import torch.nn as nn

def check_conv(x_prev,x,it,crit_conv,verbose=False):
    x_prev = x_prev if type(x_prev) is not tuple else x_prev[0]
    x = x if type(x) is not tuple else x[0]
    crit_cur = (x_prev-x).norm() / (x.norm()+1e-03)
    if verbose:
        print(it, 'crit = ', crit_cur , '\r')
    if crit_conv is not None and crit_cur < crit_conv:
        return True
    else:
        return False


def conjugate_gradient(A, b, max_iter=1e2, tol=1e-5):
    '''
    Standard conjugate gradient algorithm to solve Ax=b
        see: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    :param A: Linear operator as a callable function, has to be square!
    :param b: input tensor
    :param max_iter: maximum number of CG iterations
    :param tol: absolute tolerance for stopping the CG algorithm.
    :return: torch tensor x verifying Ax=b

    '''

    def dot(s1, s2):
        return (s1 * s2).flatten().sum()

    x = torch.zeros_like(b)

    r = b
    p = r
    rsold = dot(r, r)

    for i in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = dot(r, r)
        #print(rsnew.sqrt())
        if rsnew.sqrt() < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def gradient_descent(grad_f, x, step_size=1., max_iter=1e2, tol=1e-5):
    '''
    Standard gradient descent algorithm to solve min_x f(x)
    :param grad_f: gradient of function to bz minimized as a callable function.
    :param x: input tensor
    :param step_size: (constant) step size of the gradient descent algorithm.
    :param max_iter: maximum number of iterations
    :param tol: absolute tolerance for stopping the algorithm.
    :return: torch tensor x verifying min_x f(x)

    '''

    for i in range(int(max_iter)):
        x_prev = x
        x = x - step_size * grad_f(x)
        if check_conv(x_prev, x, i, crit_conv=tol) :
            break
    return x