import numpy as np
import torch
import torch.nn as nn

from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)


class WaveletDict(nn.Module):
    '''
    Torch implementation of the proximal operator of sparsity in a redundant wavelet dictionary domain (SARA dictionary).

    Minimisation is performed with a dual forward-backward algorithm.

    TODO: detail doc + perform tests + warm restart
    '''

    def __init__(self, y_shape, ths=0.1, max_it=100, conv_crit=1e-3, eps=1e-6, gamma=1., verbose=False,
                 list_wv=['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']):

        super(WaveletDict, self).__init__()

        self.max_it = max_it
        self.ths = ths
        self.gamma = gamma
        self.eps = eps
        self.list_wv = list_wv
        self.conv_crit = conv_crit
        self.dict = SARA_dict(torch.zeros(y_shape), level=3, list_wv=list_wv)
        self.verbose = verbose

    def prox_l1(self, x, ths=0.1):
        return torch.maximum(torch.Tensor([0]), x - ths) + torch.minimum(torch.Tensor([0]), x + ths)

    def forward(self, y):

        # Some inits
        v1 = 0. * self.dict.Psit(y)  # initialization of L1 dual variable
        r1 = torch.clone(v1)
        vy1 = torch.clone(v1)
        x_ = torch.zeros_like(y)

        for _ in range(self.max_it):

            x_old = torch.clone(x_)

            v_up = self.dict.Psi(v1)
            x_ = torch.maximum(x_ - self.gamma * (v_up + x_ - y), torch.Tensor([0.]))  # Projection on [0, +\infty)
            prev_xsol = 2. * x_ - x_old
            r1 = self.dict.Psit(prev_xsol)
            v1 = v1 + 0.5 * r1 - self.prox_l1(v1 + 0.5 * r1, ths=0.5 * self.ths)  # weights on ths

            rel_err = torch.linalg.norm(x_ - x_old) / torch.linalg.norm(x_old + self.eps)
            if rel_err < self.conv_crit:
                break

            if self.verbose:
                if _ % 50 == 0:
                    print('Iter ', str(_), ' rel crit = ', rel_err)

        if self.verbose:
            print('Converged after ', str(_), ' iterations; relative err = ', rel_err)

        return x_


def coef2vec(coef, Nc, Nx, Ny):
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
    bookkeeping.append((Nc, Nx, Ny))
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
    ind = bookkeeping[0][0] * bookkeeping[0][1] * bookkeeping[0][2]
    coef = [torch.reshape(vec[:ind], bookkeeping[0])]
    for ele in bookkeeping[1:-1]:
        indnext = ele[0] * ele[1] * ele[2]
        coef.append((torch.reshape(vec[ind:ind + indnext], ele),
                     torch.reshape(vec[ind + indnext:ind + 2 * indnext], ele),
                     torch.reshape(vec[ind + 2 * indnext:ind + 3 * indnext], ele)))
        ind += 3 * indnext

    return coef


def torch2pywt_format(Yl, Yh):
    '''
    Takes as input a torch wavelet element; outputs a list of tensors in the format of pywt (numpy library).
    '''
    Yl_ = Yl.squeeze()
    Yh_ = [torch.unbind(Yh[-(level + 1)].squeeze(), dim=0) for level in range(len(Yh))]
    wd_ = [Yl_] + Yh_

    return wd_


def pywt2torch_format(wd_):
    '''
    Takes as input a torch wavelet element; outputs a list of tensors in the format of pywt (numpy library).
    '''
    Yl = wd_[0].unsqueeze(0)
    Yh_rev = wd_[1:][::-1]
    Yh = [torch.stack(Yh_cur).unsqueeze(0) for Yh_cur in Yh_rev]

    return Yl, Yh


def wavedec_asarray(im, wv='db8', level=3):
    xfm = DWTForward(J=level, mode='zero', wave=wv)
    Yl, Yh = xfm(im)
    wd_ = torch2pywt_format(Yl, Yh)
    wd, book = coef2vec(wd_, im.shape[-3], im.shape[-2], im.shape[-1])

    return wd, book


def waverec_asarray(wd, book, wv='db8'):
    wc = vec2coef(wd, book)
    Yl, Yh = pywt2torch_format(wc)
    ifm = DWTInverse(mode='zero', wave=wv)
    Y = ifm((Yl, Yh))

    return Y


class SARA_dict(nn.Module):

    def __init__(self, im, level, list_wv=['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8']):

        super(SARA_dict, self).__init__()

        self.level = level
        self.list_coeffs = []
        self.list_b = []
        self.list_wv = list_wv

        for wv_cur in self.list_wv:
            c_cur, b_cur = wavedec_asarray(im, wv_cur, level=level)
            self.list_coeffs.append(len(c_cur))
            self.list_b.append(b_cur)

        # Dirac basis
        c_cur, b_cur = im.flatten(), im.shape
        self.list_coeffs.append(len(c_cur))
        self.list_b.append(b_cur)

        self.list_coeffs_cumsum = np.cumsum(self.list_coeffs)

    def Psit(self, x):

        list_tensors = [wavedec_asarray(x, wv_cur, level=self.level)[0] for wv_cur in self.list_wv]
        list_tensors.append(x.flatten())
        out = torch.concat(list_tensors)

        return out / np.sqrt(len(self.list_wv) + 1)

    def Psi(self, y):

        out = waverec_asarray(y[:self.list_coeffs_cumsum[0]], self.list_b[0], wv=self.list_wv[0])
        for _ in range(len(self.list_coeffs_cumsum) - 2):
            out = out + waverec_asarray(y[self.list_coeffs_cumsum[_]:self.list_coeffs_cumsum[_ + 1]],
                                        self.list_b[_ + 1], wv=self.list_wv[_ + 1])
        out = out + np.reshape(y[self.list_coeffs_cumsum[-2]:], self.list_b[-1])

        return out / np.sqrt(len(self.list_wv) + 1)

