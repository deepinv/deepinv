import torch
from deepinv.physics import LinearPhysics

speed_of_light = 3e8


class NearFieldMIMO(LinearPhysics):
    r"""
    Near field MIMO radar

    The near field model is expressed as

    .. math:

        y = Ax


    :param torch.Tensor tx_coords: coordinates of ..
    
    |sep|

    :Examples:

        >>> from deepinv.physics import Inpainting
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
        >>> mask = torch.zeros(1, 3, 3) # Define empty mask
        >>> mask[:, 2, :] = 1 # Keeping last line only
        >>> physics = Inpainting(mask=mask, img_size=x.shape[1:])
        >>> physics(x)
        tensor([[[[ 0.0000, -0.0000, -0.0000],
                  [ 0.0000, -0.0000, -0.0000],
                  [ 0.4033,  0.8380, -0.7193]]]])

    """

    def __init__(
        self,
        tx_coords,  # Nt 3
        rx_coords,  # Nr 3
        freqs,  # Nf
        v=speed_of_light,
        path_loss=True,
        sigma=0,
        tx_requires_grad=False,
        rx_requires_grad=False,
        wave_numbers_requires_grad=False,
    ):

        super().__init__()
        self.noise_model = dinv.physics.GaussianNoise(sigma=sigma)
        self.v = v
        self.tx_coords = torch.nn.Parameter(
            tx_coords, requires_grad=tx_requires_grad
        )  # Nt 3
        self.rx_coords = torch.nn.Parameter(
            rx_coords, requires_grad=rx_requires_grad
        )  # Nr 3
        self.wave_numbers = torch.nn.Parameter(
            torch.pi * 2 * freqs / v, requires_grad=wave_numbers_requires_grad
        )  # Nf

        self.path_loss = path_loss

    def _get_mat(
        self, scene_coords, tx_coords, rx_coords, wave_numbers, path_loss=None
    ):
        r"""

        something
        """
        if path_loss == None:
            path_loss = self.path_loss

        #             : Ns Nf Nt Nr Np (NumSampels NumFreqs NumTransmitters NumReceivers NumPoints)
        # scene coords: Ns Np 3 --> Ns _  _  _  Np 3
        # wavenumbers:  Nf      --> _  Nf _  _  _
        # tx_coords:    Nr 3    --> _  _  Nt _  _  3
        # rx_coords:    Nt 3    --> _  _  _  Nr _  3

        dT = torch.linalg.norm(
            scene_coords[:, None, None, None, ...]
            - tx_coords[None, None, :, None, None, :],
            dim=-1,
        )
        dR = torch.linalg.norm(
            scene_coords[:, None, None, None, ...]
            - rx_coords[None, None, None, :, None, :],
            dim=-1,
        )

        # returned tensor will be of shape (Ns Nf Nt Nr Np)
        if path_loss:
            return torch.exp(
                -1j * wave_numbers[None, :, None, None, None] * (dT + dR)
            ) / (dT * dR)
        else:
            return torch.exp(-1j * wave_numbers[None, :, None, None, None] * (dT + dR))

    def A(self, x):
        Amat = self._get_mat(
            scene_coords=x[1],
            tx_coords=self.tx_coords,
            rx_coords=self.rx_coords,
            wave_numbers=self.wave_numbers,
        )

        return torch.einsum("NFTRP,NP->NFTR", Amat, x[0])  # NCFTR

    def A_adjoint(self, y):
        Amat = self._get_mat(
            scene_coords=y[1],
            tx_coords=self.tx_coords,
            rx_coords=self.rx_coords,
            wave_numbers=self.wave_numbers,
        )

        return torch.einsum("NFTRP,NFTR->NP", Amat.conj(), y[0])

    def A_dagger(self, y, method="backprojection"):
        if method == "Back Projection":
            return self.BP(y=y)
        elif method == "Kirchhoff Migration":
            return self.KM(y=y)

    def BP(self, y):
        Amat = self._get_mat(
            scene_coords=y[1],
            tx_coords=self.tx_coords,
            rx_coords=self.rx_coords,
            wave_numbers=self.wave_numbers,
            path_loss=False,
        )

        return torch.einsum("NFTRP,NFTR->NP", Amat.conj(), y[0])

    def KM(self, y):
        # Current implementation is for planar arrays
        scene_coords = y[1]

        rt = (
            scene_coords[:, None, None, None, ...]
            - self.tx_coords[None, None, :, None, None, :]
        )
        dt = torch.linalg.norm(rt, dim=-1)
        ut = rt / (dt[..., None])

        rr = (
            scene_coords[:, None, None, None, ...]
            - self.rx_coords[None, None, None, :, None, :]
        )
        dr = torch.linalg.norm(rr, dim=-1)
        ur = rr / (dr[..., None])

        time_derivative_op = (
            1j * self.wave_numbers[None, :, None, None, None] * (dr + dt)
        )

        term_1 = 4 * ut[..., -1] * ur[..., -1] / (dr * dt)

        term_par_1 = (1 / self.v**2) * (time_derivative_op**2)
        term_par_2 = (1 / self.v) * (1 / dr + 1 / dt) * time_derivative_op
        term_par_3 = 1 / (dr * dt)

        w = (
            term_1
            * (term_par_1 + term_par_2 + term_par_3)
            * torch.exp(1j * self.wave_numbers[None, :, None, None, None] * (dt + dr))
        )

        return torch.einsum("NFTRP,NFTR->NP", w, y[0])


class NearFieldFMCWMonoSAR(LinearPhysics):
    r""" 
    Near field full 
    
    """
    def __init__(
        self, dx, Nx, dy, Ny, rmin, rmax, freqs, v=speed_of_light, sigma=0  # Nt 3  # Nf
    ):
        super().__init__()

        self.v = v
        self.dx = dx
        self.dy = dy

        self.Nx = Nx
        self.Ny = Ny

        self.range_res = v / (2 * (freqs[-1] - freqs[0]))
        self.range_axis = torch.arange(len(freqs)) * self.range_res

        idx_min, idx_max = self.r2idx(rmin=rmin, rmax=rmax)

        self.idx_min = idx_min
        self.idx_max = idx_max

        self.k = 2 * torch.pi * (freqs) / self.v
        kx, ky, kz = self._get_kxyz(Nx=self.Nx, Ny=self.Ny, k=self.k[idx_min:idx_max])
        self.range_use = self.range_axis[idx_min:idx_max]

        self.kx = torch.nn.Parameter(kx, requires_grad=False)
        self.ky = torch.nn.Parameter(ky, requires_grad=False)
        self.kz = torch.nn.Parameter(kz, requires_grad=False)

        self.rma_kernel = torch.nn.Parameter(
            self._get_rma_kernel(R=self.range_use, kx=self.kx, ky=self.ky, kz=self.kz),
            requires_grad=False,
        )
        self.kernel_non_zero = self.rma_kernel.abs() > 0

    def F(self, sig):
        return torch.fft.fftn(sig, dim=[0, 1], s=[self.Nx, self.Ny], norm="ortho")

    def FH(self, sig):
        return torch.fft.ifftn(sig, dim=[0, 1], s=[self.Nx, self.Ny], norm="ortho")

    def r2idx(self, rmin, rmax):
        idx_min = torch.floor(rmin / self.range_res).to(int)
        idx_max = torch.ceil(rmax / self.range_res).to(int)
        return idx_min, idx_max

    def prep(self, time_domain_echo):
        y = torch.fft.fftn(time_domain_echo, dim=-1)
        return y[:, :, self.idx_min : self.idx_max]

    def _get_kxyz(self, Nx, Ny, k):

        wx = 2 * torch.pi / self.dx
        wy = 2 * torch.pi / self.dy

        kx = torch.linspace(-(wx / 2), (wx / 2), Nx)[:, None, None]
        ky = torch.linspace(-(wy / 2), (wy / 2), Ny)[None, :, None]
        kz = (2 * k[None, None, ...]) ** 2 - kx**2 - ky**2
        kz[kz < 0] = 0
        kz = torch.sqrt(kz)
        return kx, ky, kz

    def _get_rma_kernel(self, R, kx, ky, kz):

        phaseFactor0 = torch.exp(-1j * R[None, None, :] * kz)
        phaseFactor0[(kx**2 + ky**2) > (2 * kz) ** 2] = 0
        phaseFactor = kz * phaseFactor0
        phaseFactor = torch.fft.fftshift(phaseFactor, dim=[0, 1])
        return phaseFactor

    def A_dagger(self, y):
        return self.FH(self.F(y) * self.rma_kernel)

    def A(self, x):
        X = self.F(x)
        Y = X * 0
        Y[self.kernel_non_zero] = (
            X[self.kernel_non_zero] / self.rma_kernel[self.kernel_non_zero]
        )
        return self.FH(Y)

    def A_adjoint(self, y):
        Y = self.F(y)
        X = Y * 0
        X[self.kernel_non_zero] = Y[self.kernel_non_zero] / (
            self.rma_kernel[self.kernel_non_zero].conj()
        )
        return self.FH(X)
