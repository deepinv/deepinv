def gradient_descent_step(x, grad, bregman_potential):
    if bregman_potential == 'L2':
        return x - grad
    elif bregman_potential == 'Burg_entropy':
        return x / (1 + x*grad)
    else: 
        raise ValueError(f'Gradient Descent with bregman potential {bregman_potential} not implemented')
