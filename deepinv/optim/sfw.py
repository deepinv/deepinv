
import torch
import torch.nn.functional as F
import deepinv as dinv



class SFW(torch.nn.Module):
    """
    Sliding Frank-Wolfe algorithm for off-grid spikes recovery.
    BLASSO: \min_{m \in \mathcal{M}([0,1]^2)} 
    \frac{1}{2}\left\|A(m)-y\right\|_2^2 + \lambda \|m\|_{\mathrm{TV}},
    
    recovery measure: m = \sum_{k=1}^{K} a_k \delta_{x_k},
               with  `a_k \in \mathbb{R}` and `x_k \in [0,1]^2`.
    """
    def __init__(
        self,
        lambda_reg: float,
        max_iter: int = 10,
        lasso_max_iter: int = 200,
        sliding_max_iter: int = 100,
    ):
        super().__init__()

        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.lasso_max_iter = lasso_max_iter
        self.sliding_max_iter = sliding_max_iter

    
    def _certificate_kernel(
        self,
        physics,
        device,
        dtype,
    ) -> torch.Tensor:
       
        Nx = physics.Nx
        Ny = physics.Ny
        sigma = physics.sigma_psf

        # Pixel centers on [0,1]^2
        xc = (
            torch.arange(
                Nx,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) / Nx

        yc = (
            torch.arange(
                Ny,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) / Ny

        Y, X = torch.meshgrid(
            yc,
            xc,
            indexing="ij",
        )

        # Same fixed normalization constant as SpikePhysics
       
        x_ref = 0.5 * (X.min() + X.max())
        y_ref = 0.5 * (Y.min() + Y.max())

        kernel_ref = torch.exp(
            -(
                (X - x_ref) ** 2
                + (Y - y_ref) ** 2
            )
            / (2 * sigma**2)
        )

        C_sigma = kernel_ref.sum()

       
        dx = torch.arange(
            -(Nx - 1),
            Nx,
            device=device,
            dtype=dtype,
        ) / Nx

        dy = torch.arange(
            -(Ny - 1),
            Ny,
            device=device,
            dtype=dtype,
        ) / Ny

        DY, DX = torch.meshgrid(
            dy,
            dx,
            indexing="ij",
        )

        kernel = torch.exp(
            -(DX**2 + DY**2)
            / (2 * sigma**2)
        ) / C_sigma

       
        return kernel[None, None, :, :]

    

    def _certificate(
        self,
        residual: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the SFW certificate on the pixel grid.

        \eta^{[k]} = \frac{1}{\lambda}  h_\sigma * \left(y - A(m^{[k]})\right).

        Parameters
        ----------
        residual : torch.Tensor
            Residual y - A(m), with shape (1, 1, Ny, Nx).

        kernel : torch.Tensor
            Gaussian kernel used for the adjoint convolution.

        Returns
        -------
        torch.Tensor
            Certificate with shape (1, 1, Ny, Nx).
    
        """

        if residual.ndim != 4:
            raise ValueError(
                "residual must have shape (1, 1, Ny, Nx)."
            )

        pad_y = (kernel.shape[-2] - 1) // 2
        pad_x = (kernel.shape[-1] - 1) // 2

        certificate = F.conv2d(
            residual,
            kernel,
            padding=(pad_y, pad_x),
        )
        return certificate / self.lambda_reg


    
    def _support_estimation(
    self,
    certificate: torch.Tensor,
    physics,
    ):
        """
        Estimate the new support point from the certificate.
        
        It consists to select the point is the center of the pixel where the absolute
        value of the certificate is maximal.

        Parameters
        ----------
        certificate : torch.Tensor
            Certificate with shape (1, 1, Ny, Nx).

        physics : SpikePhysics
            Forward physics containing Nx and Ny.

        Returns
        -------
        position : torch.Tensor
            selected position with shape (2,), in [0,1]^2.

        eta_value : torch.Tensor
            Signed certificate value at the selected position.

        eta_max : torch.Tensor
            Maximum absolute certificate value.
        """

        # --------------------------------------------------
        # Maximum of the absolute certificate
        # --------------------------------------------------
        abs_certificate = torch.abs(
        certificate[0, 0]
        )

        flat_index = torch.argmax(
            abs_certificate
        )
        # --------------------------------------------------
        # Flat index -> pixel indices
        # --------------------------------------------------
        iy = flat_index // physics.Nx
        ix = flat_index % physics.Nx

        # --------------------------------------------------
        # Pixel center coordinates on [0,1]^2
        # --------------------------------------------------
        x = (
            ix.to(certificate.dtype) + 0.5
        ) / physics.Nx

        y = (
            iy.to(certificate.dtype) + 0.5
        ) / physics.Ny

        position = torch.stack((x, y))

        # --------------------------------------------------
        # Certificate values at the selected pixel
        # --------------------------------------------------
        eta_value = certificate[0, 0, iy, ix]

        eta_max = torch.abs(eta_value)

        return position, eta_value, eta_max
        

    

    def _lasso_step(
        self,
        y: torch.Tensor,
        physics,
        positions: torch.Tensor,
    ) -> torch.Tensor:
       
        K = positions.shape[0]

        # --------------------------------------------------
        # Images A(delta_xj) of all current support points
        # --------------------------------------------------
        atoms = []

        for j in range(K):

            atom = physics.A(
                positions[j:j + 1]
            )[0, 0]

            atoms.append(atom)

        atoms = torch.stack(
            atoms,
            dim=0,
        )

        # atoms.shape = (K, Ny, Nx)

        def A_amplitudes(amplitudes, **kwargs):

            if amplitudes.ndim == 1:
                amplitudes = amplitudes.unsqueeze(0)

            image = torch.einsum(
                "bk,kij->bij",
                amplitudes,
                atoms,
            )

            return image[:, None, :, :]

        # --------------------------------------------------
        # Adjoint B_X^*
        # --------------------------------------------------
        def A_adjoint_amplitudes(image, **kwargs):

            return torch.einsum(
                "bij,kij->bk",
                image[:, 0],
                atoms,
            )

        amplitude_physics = dinv.physics.LinearPhysics(
            A=A_amplitudes,
            A_adjoint=A_adjoint_amplitudes,
            device=y.device,
        )

        # --------------------------------------------------
        # Lipschitz constant
        # L = ||B_X||^2
        #   = lambda_max(B_X^* B_X)
        # --------------------------------------------------
        atoms_flat = atoms.reshape(
            K,
            -1,
        )

        gram = atoms_flat @ atoms_flat.T

        L = torch.linalg.eigvalsh(
            gram
        ).max()

        stepsize = 0.99 / L.item()

        # --------------------------------------------------
        # DeepInv FISTA
        # --------------------------------------------------
        fista = dinv.optim.FISTA(
            data_fidelity=dinv.optim.L2(),
            prior=dinv.optim.L1Prior(),
            lambda_reg=self.lambda_reg,
            stepsize=stepsize,
            max_iter=self.lasso_max_iter,
        )

        amplitudes = fista(
            y, amplitude_physics, )

        return amplitudes.squeeze(0).detach()



    def _sliding_step(
        self,
        y: torch.Tensor,
        physics,
        positions: torch.Tensor,
        amplitudes: torch.Tensor,
    ):
    
        # -------------------------------------------------
        # Initialization from the LASSO step
        # --------------------------------------------------
        positions_opt = (
            positions
            .detach()
            .clone()
            .requires_grad_(True)
        )

        amplitudes_opt = (
            amplitudes
            .detach()
            .clone()
            .requires_grad_(True)
        )

        optimizer = torch.optim.Adam(
            [
                amplitudes_opt,
                positions_opt,
            ],
            lr=1e-3,
        )

        losses = []
        previous_loss = float("inf")

        # --------------------------------------------------
        # Joint optimization
        # --------------------------------------------------
        for _ in range(self.sliding_max_iter):

            optimizer.zero_grad()

            prediction = physics.A(
                [
                    positions_opt,
                    amplitudes_opt,
                ]
            )

            data_term = 0.5 * torch.sum(
                (prediction - y) ** 2
            )

            regularization = (
                self.lambda_reg
                * torch.abs(amplitudes_opt).sum()
            )

            loss = data_term + regularization

            loss.backward()

            optimizer.step()

            # Projection inside [0,1]^2
            with torch.no_grad():
                positions_opt.clamp_(
                    min=0.0,
                    max=1.0,
                )

            loss_value = loss.detach().item()
            losses.append(loss_value)

            # Convergence criterion
            if abs(previous_loss - loss_value) < 1e-6:
                break

            previous_loss = loss_value
        
        print(f"Sliding final loss: {loss_value:.6e}")

        return (
            positions_opt.detach(),
            amplitudes_opt.detach(),
        )



    
    def forward(
        self,
        y: torch.Tensor,
        physics,
    ):

        device = y.device
        dtype = y.dtype

       
        certificate_kernel = (
            self._certificate_kernel(
                physics,
                device=device,
                dtype=dtype,
            )
        )

        # --------------------------------------------------
        # Initial measure m^0 = 0
        # --------------------------------------------------
        positions = torch.empty(
            (0, 2),
            device=device,
            dtype=dtype,
        )

        amplitudes = torch.empty(
            (0,),
            device=device,
            dtype=dtype,
        )

        # ==================================================
        # SFW iterations
        # ==================================================
        for _ in range(self.max_iter):

            # Current prediction A(m^k)
            prediction = physics.A(
                [
                    positions,
                    amplitudes,
                ]
            )

            # Residual: r^k = y - A(m^k)
           
            residual = ( y - prediction )

            # Certificate: eta^k = (1/lambda) A^* r^k
            certificate = self._certificate(
                residual,
                certificate_kernel,
            )

            # Estimate new support point
          
            (
                new_position,
                eta_value,
                eta_max,
            ) = self._support_estimation(  certificate, physics,)

            # Stopping criterion:||eta^k||_infinity <= 1
            if (
                eta_max.item()
                <= 1.0 
            ):
                break

            # Add new point to the support
            positions = torch.cat(
                [
                    positions,
                    new_position[None, :],
                ],
                dim=0,
            )

            # LASSO: optimize amplitudes at fixed positions
         
            amplitudes = self._lasso_step(
                y,
                physics,
                positions,
            )

            # Sliding:jointly optimize positions and amplitudes
            (
                positions,
                amplitudes,
            ) = self._sliding_step(
                y,
                physics,
                positions,
                amplitudes,
            )
            
            # Remove negligible atoms
            keep = torch.abs(amplitudes) > 1e-2
            positions = positions[keep]
            amplitudes = amplitudes[keep]

        return positions, amplitudes