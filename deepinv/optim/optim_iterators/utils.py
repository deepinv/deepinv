def gradient_descent_step(x, grad, bregman_potential="L2"):
    r"""
    Performs a single step of gradient descent on the Bregman divergence.

    :param torch.Tensor x: Current iterate.
    :param torch.Tensor grad: Gradient of the Bregman divergence.
    :param str bregman_potential: Bregman potential used in the Bregman divergence.
    """
    if bregman_potential == "L2":
        grad_step = x - grad
    elif bregman_potential == "Burg_entropy":
        grad_step = x / (1 + x * grad)
    else:
        raise ValueError(
            f"Gradient Descent with bregman potential {bregman_potential} not implemented"
        )
    return grad_step
