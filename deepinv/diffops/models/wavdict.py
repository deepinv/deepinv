import math
import numpy as np
import torch
import torch.nn as nn

from .denoiser import register

try:
    from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
except:
    print('WARNING: pytorch_wavelets not imported')

@register('waveletprior')
class WaveletPrior(nn.Module):
    '''
    Torch implementation of the proximal operator of sparsity in a redundant wavelet dictionary domain (SARA dictionary).

    Minimisation is performed with a dual forward-backward algorithm.

    TODO: raise error if list_wv is larger than 1
    '''

    def __init__(self, y_shape, ths=0.1, max_it=100, conv_crit=1e-3, eps=1e-6, gamma=1., verbose=False,
                 list_wv=['db8'], dtype=torch.FloatTensor, level=3):
        super(WaveletPrior, self).__init__()

        self.dtype = dtype
        self.max_it = max_it
        self.ths = ths
        self.gamma = gamma
        self.eps = eps
        self.list_wv = list_wv
        self.conv_crit = conv_crit
        self.dict = SARA_dict(torch.zeros(y_shape).type(self.dtype), level=level, list_wv=list_wv)
        self.verbose = verbose

    def prox_l1(self, x, ths=0.1):
        return torch.maximum(torch.Tensor([0]).type(x.dtype), x - ths) + torch.minimum(torch.Tensor([0]).type(x.dtype),
                                                                                       x + ths)

    def forward(self, y, ths=None):

        ths_ = self.ths
        if ths is not None:
            ths_ = ths
        v1_low, v1_high = self.dict.Psit(y)  # initialization of L1 dual variable
        v1_high = self.prox_l1(v1_high, ths=ths_)
        x = self.dict.Psi(v1_low, v1_high)

        return x

class WaveletDict(nn.Module):
    '''
    WORK IN PROGRESS
    Torch implementation of the proximal operator of sparsity in a redundant wavelet dictionary domain (SARA dictionary).

    Minimisation is performed with a dual forward-backward algorithm.

    TODO: detail doc + perform tests
    '''

    def __init__(self, y_shape, ths=0.1, max_it=100, conv_crit=1e-3, eps=1e-6, gamma=1., verbose=False,
                 list_wv=['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8'], dtype=torch.FloatTensor, level=3):

        super(WaveletDict, self).__init__()

        self.dtype = dtype
        self.max_it = max_it
        self.ths = ths
        self.gamma = gamma
        self.gamma = gamma
        self.eps = eps
        self.list_wv = list_wv
        self.conv_crit = conv_crit
        self.dict = SARA_dict(torch.zeros(y_shape).type(self.dtype), level=level, list_wv=list_wv)
        self.verbose = verbose

        self.v1_low = None
        self.v1_high = None
        self.r1 = None
        self.vy1 = None
        self.x_ = None

    def prox_l1(self, x, ths=0.1):
        return torch.maximum(torch.Tensor([0]).type(x.dtype), x - ths) + torch.minimum(torch.Tensor([0]).type(x.dtype),
                                                                                       x + ths)

    def forward(self, y, ths=None):  # , A, At, centre, radius, gamma=1.0, crit_conv=1e-5):

        ths_ = self.ths

        if ths is not None:
            ths_ = ths

        self.u_low, self.u_high = self.dict.Psit(y)

        for it in range(self.max_it):

            u_prev = self.u_high.clone()

            x = y - self.dict.Psi(self.u_low, self.u_high)

            Ax_low, Ax_high = self.dict.Psit(x)

            u_low_ = self.u_low + self.gamma * Ax_low
            u_high_ = self.u_high + self.gamma * Ax_high

            self.u_low = u_low_ - self.gamma * self.prox_l1(u_low_ / self.gamma, ths=0. / self.gamma)
            self.u_high = u_high_ - self.gamma * self.prox_l1(u_high_ / self.gamma, ths=ths_ / self.gamma)

            rel_crit = ((self.u_high - u_prev).norm()) / ((self.u_high).norm() + 1e-12)
            if it % 50 == 0:
                print(it, rel_crit)
            if rel_crit < self.conv_crit and it > 10:
                break
        return x


def coef2vec(coef):
    """
    Convert wavelet coefficients to an array-type vector, inverse operation of vec2coef.
    The initial wavelet coefficients are stocked in a list as follows:
        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
    and each element is a 2D array.
    After the conversion, the returned vector is as follows:
    [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ...,cH1.flatten(), cV1.flatten(), cD1.flatten()].
    """
    vec = torch.Tensor([])
    bookkeeping = []
    for ele in coef:
        if type(ele) == tuple:
            bookkeeping.append((np.shape(ele[0])))
            for wavcoef in ele:
                vec = torch.concat((vec, wavcoef.flatten()))
        else:
            bookkeeping.append((np.shape(ele)))
            vec = torch.concat((vec, ele.flatten()))
    return vec, bookkeeping


def vec2coef(vec, bookkeeping):
    """
    Convert an array-type vector to wavelet coefficients, inverse operation of coef2vec.
    The initial vector is stocked in a 1D array as follows:
    [cAn.flatten(), cHn.flatten(), cVn.flatten(), cDn.flatten(), ..., cH1.flatten(), cV1.flatten(), cD1.flatten()].
    After the conversion, the returned wavelet coefficient is in the form of the list as follows:
        [cAn, (cHm, cVn, cDn), ..., (cH1, cV1, cD1)],
    and each element is a 2D array. This list can be passed as the argument in pywt.waverec2.
    """
    ind = 0
    coef = []
    for ele in bookkeeping:
        indnext = math.prod(ele)
        coef.append((torch.reshape(vec[ind:ind + indnext], ele),
                     torch.reshape(vec[ind + indnext:ind + 2 * indnext], ele),
                     torch.reshape(vec[ind + 2 * indnext:ind + 3 * indnext], ele)))
        ind += 3 * indnext

    return coef


def torch2pywt_format(Yl, Yh):
    '''
    Takes as input a torch wavelet element; outputs a list of tensors in the format of pywt (numpy library).
    '''
    Yh_ = [torch.unbind(Yh[-(level + 1)].squeeze(), dim=0) for level in range(len(Yh))]

    return Yl, Yh_


def pywt2torch_format(Yh_):
    '''
    Takes as input a torch wavelet element; outputs a list of tensors in the format of pywt (numpy library).
    '''
    Yh_rev = Yh_[::-1]
    Yh = [torch.stack(Yh_cur).unsqueeze(0) for Yh_cur in Yh_rev]

    if len(Yh[0].shape) == 4:  #  Black and white case
        Yh = [Yh_cur.unsqueeze(0) for Yh_cur in Yh]

    return Yh


def wavedec_asarray(im, wv='db8', level=3):
    xfm = DWTForward(J=level, mode='zero', wave=wv)
    Yl, Yh = xfm(im)
    Yl, Yh_ = torch2pywt_format(Yl, Yh)
    wd, book = coef2vec(Yh_)

    return Yl.flatten(), Yl.shape, wd, book


def waverec_asarray(Yl_flat, Yl_shape, wd, book, wv='db8'):
    wc = vec2coef(wd, book)
    Yl = Yl_flat.reshape(Yl_shape)

    Yh = pywt2torch_format(wc)
    ifm = DWTInverse(mode='zero', wave=wv)
    Y = ifm((Yl, Yh))

    return Y


class SARA_dict(nn.Module):

    def __init__(self, im, level, list_wv=['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']):

        super(SARA_dict, self).__init__()

        self.level = level
        self.list_coeffs_lf = []
        self.list_coeffs = []
        self.list_b = []
        self.list_lfshape = []
        self.list_wv = list_wv

        for wv_cur in self.list_wv:
            low_cur, lf_shape_cur, c_cur, b_cur = wavedec_asarray(im, wv_cur, level=level)
            self.list_coeffs_lf.append(low_cur.shape[0])
            self.list_coeffs.append(len(c_cur))
            self.list_b.append(b_cur)
            self.list_lfshape.append(lf_shape_cur)

        self.list_coeffs_cumsum = np.cumsum(self.list_coeffs)
        self.list_coeffs_lf_cumsum = np.cumsum(self.list_coeffs_lf)

    def Psit(self, x):

        list_tensors_lf = [wavedec_asarray(x, wv_cur, level=self.level)[0] for wv_cur in self.list_wv]
        list_tensors_hf = [wavedec_asarray(x, wv_cur, level=self.level)[2] for wv_cur in self.list_wv]

        out_hf = torch.concat(list_tensors_hf)
        out_lf = torch.concat(list_tensors_lf)

        return out_lf / np.sqrt(len(self.list_wv)), out_hf / np.sqrt(len(self.list_wv))

    def Psi(self, y_lf, y_hf):

        out = waverec_asarray(y_lf[:self.list_coeffs_lf_cumsum[0]], self.list_lfshape[0],
                              y_hf[:self.list_coeffs_cumsum[0]], self.list_b[0], wv=self.list_wv[0])

        for _ in range(len(self.list_coeffs_cumsum) - 1):
            out = out + waverec_asarray(y_lf[self.list_coeffs_lf_cumsum[_]:self.list_coeffs_lf_cumsum[_ + 1]],
                                        self.list_lfshape[_ + 1],
                                        y_hf[self.list_coeffs_cumsum[_]:self.list_coeffs_cumsum[_ + 1]],
                                        self.list_b[_ + 1], wv=self.list_wv[_ + 1])

        return out / np.sqrt(len(self.list_wv))
