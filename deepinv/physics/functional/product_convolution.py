import torch
from torch import Tensor
from deepinv.physics.functional import conv2d, conv_transpose2d, multiplier, multiplier_adjoint

def product_convolution(x: Tensor, w: Tensor, h: Tensor) -> Tensor:
    r"""

    Product-convolution operator. 
    Escande, P., & Weiss, P. (2017). Approximation of integral operators using product-convolution expansions. Journal of Mathematical Imaging and Vision, 58, 333-348.

    This forward operator performs

    .. math:: y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (K, c, ...)
    :param torch.Tensor h: Tensor of size (K, c, ...)

    """
    
    K = w.shape[0]
    result = torch.zeros_like(x)
    for k in range(K):
        xk = multiplier(x, w[k])
        result += conv2d(xk, h[k])
        
    return result

def product_convolution_adjoint(x: Tensor, w: Tensor, h: Tensor) -> Tensor:
    r"""

    Product-convolution adjoint operator. 
        
    :param torch.Tensor x: Tensor of size (B, C, ...)
    :param torch.Tensor w: Tensor of size (K, c, ...)
    :param torch.Tensor h: Tensor of size (K, c, ...)
    """
    
    K = w.shape[0]
    result = torch.zeros_like(x)
    for k in range(K):
        xk =  conv_transpose2d(x, h[k])
        result += multiplier_adjoint(xk, w[k])
    
    return result


if __name__ == "main":
    import deepinv as dinv
    import torch
    from deepinv.utils.plotting import plot
    from deepinv.utils.demo import load_url_image, get_image_url
    from deepinv.physics.generator.blur import DiffractionBlurGenerator
    
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url, grayscale=False).to(device)
    x = torch.tensor(x, device=device, dtype=torch.float)
    plot(x)
    
    
    
    
