from __future__ import annotations
from .forward import LinearPhysics
import torch
import math


class SPECT(LinearPhysics):
    def __init__(
        self,
        attenuation: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attenuation = attenuation

    def A(
        self,
        x: torch.Tensor,
        voxel_size: torch.Tensor,
        attenuation: torch.Tensor | None = None,
        fwhm_data_mm: float | tuple = 4.0,
    ) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(
                f"Input volume must have 1 channel, got {x.shape[1]} channels"
            )
        self.update_parameters(attenuation=attenuation)
        attenuation = self.attenuation

        n_x, n_y, n_z = x[-3:-1]
        n_view = x.spacing[-2]
        d_y = voxel_size[-2]
        if attenuation == None:
            attenuation = 0.014 * torch.ones(n_x, n_y, n_z)

        v = torch.zeros(n_x, n_z, n_view)

        if fwhm_data_mm.shape == 2:
            p = fwhm_data_mm.unsqueeze(-1).unsqueeze(-1)
            p[-2] = n_y
            p[-1] = n_view
            # FWHM(d) = (D/L)·(L + d + b)
            p[:, :, j, l] = gaussian2d(px, pz, sigma=sigma_of_depth(j, l))

        for l in range(n_view):
            theta_l = 2 * math.pi * l / n_view
            x_tilde = x.rotate(theta_l)
            att_tilde = attenuation.rotate(theta_l)
            j = x_tilde[-2] // d_y
            for j in range(n_y):
                mu_bar = torch.exp(
                    -d_y
                    * (
                        0.5 * att_tilde
                        + att_tilde.flip(1).cumsum(1).flip(1)
                        - att_tilde
                    )
                )
                x_tilde[:, j, :] *= mu_bar[:, j, :]
                v[:, :, l] += torch.nn.functional.conv2d(
                    x_tilde[:, j, :], p[:, :, j, l]
                )
