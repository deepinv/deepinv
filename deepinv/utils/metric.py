import torch


def norm(a):
    return a.pow(2).sum(dim=3).sum(dim=2).sqrt().unsqueeze(2).unsqueeze(3)


def cal_angle(a, b):
    norm_a = (a * a).flatten().sum().sqrt()
    norm_b = (b * b).flatten().sum().sqrt()
    angle = (a * b).flatten().sum() / (norm_a * norm_b)
    angle = angle.acos() / 3.14159265359
    return angle.detach().cpu().numpy()


def cal_psnr(a, b, max_pixel=1, normalize=False):
    r"""
    Computes the peak signal-to-noise ratio (PSNR)

    If the tensors have size (N, C, H, W), then the PSNR is computed as

    .. math::
        \text{PSNR} = \frac{20}{N} \log_{10} \frac{MAX_I}{\sqrt{\|a- b\|^2_2 / (CHW) }}

    where :math:`MAX_I` is the maximum possible pixel value of the image (e.g. 1.0 for a
    normalized image), and :math:`a` and :math:`b` are the estimate and reference images.

    :param torch.Tensor a: tensor estimate
    :param torch.Tensor b: tensor reference
    :param float max_pixel: maximum pixel value
    :param bool normalize: if ``True``, a is normalized to have the same norm as b.
    """
    with torch.no_grad():
        if type(a) is list or type(a) is tuple:
            a = a[0]
            b = b[0]

        if normalize:
            an = a / norm(a) * norm(b)
        else:
            an = a

        mse = (an - b).pow(2).reshape(an.shape[0], -1).mean(dim=1)
        mse[mse == 0] = 1e-10
        psnr = 20 * torch.log10(max_pixel / mse.sqrt())

    return psnr.mean().detach().cpu().item()


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
    a = complex_abs(a.permute(0, 2, 3, 1))
    b = complex_abs(b.permute(0, 2, 3, 1))
    return cal_psnr(a, b)


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
    return (data**2).sum(dim=-1).sqrt()


def norm_psnr(a, b, complex=False):
    return cal_psnr(
        (a - a.min()) / (a.max() - a.min()),
        (b - b.min()) / (b.max() - b.min()),
        complex=complex,
    )
