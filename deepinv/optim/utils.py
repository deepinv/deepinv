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

def create_block_image(x):
    '''
    Concatenates a list of images :math:`x_i` of different shapes :math:`(B,C,H_i,W_i)` into a single image of shape :math:`(B,C, \prod_i H_i, \prod_i W_i)` with diagonal blocks.
    
    :param list x: List of images :math:`x_i` of different shapes :math:`(B,C,H_i,W_i)`.
    '''
    B, C = x[0].shape[0], x[0].shape[1]
    return torch.stack([torch.stack([torch.block_diag(*[el[j,i,:,:] for el in x]) for i in range(C)]) for j in range(B)])

def tuple_from_block_image(x, shapes):
    '''
    From a single image of shape :math:`(B,C, \prod_i H_i, \prod_i W_i)` with diagonal blocks, creates a tuple of images :math:`x_i` of shapes `shapes[i]` .
    
    :param list x: image of shape :math:`(B,C, \prod_i H_i, \prod_i W_i)` with diagonal blocks
    '''
    B, C = x[0].shape[0], x[0].shape[1]
    shapes = [[B,C,0,0]] + shapes
    return tuple([x[:,:,shapes[i][2]:shapes[i+1][2],shapes[i][2]] for i in range(len(shapes)-1)])


def init_anderson_acceleration(x, history_size):
    r"""
    Initialize the Anderson acceleration algorithm.

    :param x: initial iterate.
    :param history_size: size of the histoiry for the Anderson acceleration algorithm.
    :param dtype: dtype of the update.
    :param device: device of the update.
    """
    if isinstance(x, tuple):
        x = create_block_image(x)
    B, N = x.view(x.shape[0],-1).shape
    x_hist = torch.zeros(
        B, history_size, N, dtype=x.dtype, device=x.device
    )  # history of iterates.
    T_hist = torch.zeros(
        B, history_size, N, dtype=x.dtype, device=x.device
    )  # history of T(x_k) with T the fixed point operator.
    H = torch.zeros(
        B,
        history_size + 1,
        history_size + 1,
        dtype=x.dtype,
        device=x.device,
    )  # H in the Anderson acceleration linear system Hp = q .
    H[:, 0, 1:] = H[:, 1:, 0] = 1.0
    q = torch.zeros(
        B, history_size + 1, 1, dtype=x.dtype, device=x.device
    )  # q in the Anderson acceleration linear system Hp = q .
    q[:, 0] = 1
    return x_hist, T_hist, H, q


def anderson_acceleration_step(
    iterator,
    it,
    history_size,
    beta_anderson_acc,
    eps_anderson_acc,
    X_prev,
    TX_prev,
    x_hist,
    T_hist,
    H,
    q,
    cur_data_fidelity,
    cur_prior,
    cur_params,
    *args,
):
    r"""
    Anderson acceleration step.
    
    :param deepinv.optim.optim_iterators.OptimIterator iterator: Fixed-point iterator.
    :param int it: current iteration.
    :param int history_size: size of the histoiry for the Anderson acceleration algorithm.
    :param float beta_anderson_acc: momentum of the Anderson acceleration step. 
    :param float eps_anderson_acc: regularization parameter of the Anderson acceleration step. 
    :param dict X_prev: previous iterate.
    :param dict TX_prev: output of the fixed-point operator evaluated at X_prev
    :param torch.Tensor x_hist: history of last ``history-size`` iterates.
    :param torch.Tensor T_hist: history of T evaluations at the last ``history-size`` iterates, where T is the fixed-point operator.
    :param torch.Tensor H: H in the Anderson acceleration linear system Hp = q .
    :param torch.Tensor q: q in the Anderson acceleration linear system Hp = q .
    :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
    :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
    :param dict cur_params: Dictionary containing the current parameters of the algorithm.
    :param args: arguments for the iterator.
    """
    x_prev = X_prev["iterate"]  # current iterate x
    Tx_prev = TX_prev["iterate"]  # current iterate Tx
    if isinstance(x_prev, tuple):
        x_shapes = [el.shape for el in x_prev]
        x_prev = create_block_image(x_prev)
        Tx_prev = create_block_image(Tx_prev)
    batch_size = x_prev.shape[0]
    x_hist[:, it % history_size] = x_prev.view((batch_size, -1))
    T_hist[:, it % history_size] = Tx_prev.view((batch_size, -1))
    m = min(it + 1, history_size)
    G = T_hist[:, :m] - x_hist[:, :m]
    H[:, 1 : m + 1, 1 : m + 1] = (
        torch.bmm(G, G.transpose(1, 2))
        + eps_anderson_acc
        * torch.eye(m, dtype=x_prev[0].dtype, device=x_prev[0].device)[None]
    )
    p = torch.linalg.solve(H[:, : m + 1, : m + 1], q[:, : m + 1])[
        :, 1 : m + 1, 0
    ]  # solve the linear system H p = q.
    x = (
        beta_anderson_acc * (p[:, None] @ T_hist[:, :m])[:, 0]
        + (1 - beta_anderson_acc) * (p[:, None] @ x_hist[:, :m])[:, 0]
    )
    x = x.view(x_prev.shape)
    if isinstance(x_prev, tuple):
        x = tuple_from_block_image(x.view(x_prev.shape),x_shapes)
    estimate = iterator.get_estimate_from_iterate(
        x, cur_data_fidelity, cur_prior, cur_params, *args
    )
    cost = (
        iterator.cost_fn(estimate, cur_data_fidelity, cur_prior, cur_params, *args)
        if iterator.has_cost
        else None
    )
    return {"iterate": x, "estimate": estimate, "cost": cost}