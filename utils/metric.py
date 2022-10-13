import torch
import fastmri

def abs(x):
    return fastmri.complex_abs(x.squeeze().permute(1,2,0)).detach().cpu().numpy()

def cal_psnr(a, b, max_pixel=1, complex=False, normalize=False):#True
    """Computes the peak signal-to-noise ratio (PSNR)"""
    # a: prediction
    # b: groundtruth


    with torch.no_grad():
        if normalize:
            a = (a - a.min()) / (a.max() - a.min())
            b = (b - b.min()) / (b.max() - b.min())

        if complex:
            a = fastmri.complex_abs(a.permute(0, 2, 3, 1))
            b = fastmri.complex_abs(b.permute(0, 2, 3, 1))
        mse = torch.mean((a - b) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.detach().cpu().numpy() if psnr.device is not 'cpu' else psnr

def cal_mse(a, b):
    """Computes the mean squared error (MSE)"""
    with torch.no_grad():
        mse = torch.mean((a - b) ** 2)
    return mse


def cal_psnr_complex(a, b):
    """
    first permute the dimension, such that the last dimension of the tensor is 2 (real, imag)
    :param a: shape [N,2,H,W]
    :param b: shape [N,2,H,W]
    :return: psnr value
    """
    a = complex_abs(a.permute(0,2,3,1))
    b = complex_abs(b.permute(0,2,3,1))
    return cal_psnr(a,b)

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def norm_psnr(a, b, complex=False):
    return cal_psnr((a - a.min()) / (a.max() - a.min()),
                    (b - b.min()) / (b.max() - b.min()), complex=complex)