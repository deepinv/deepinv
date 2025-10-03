#### Ultrasound passive cavitation beamforming operator
from deepinv.physics.forward import LinearPhysics
import numpy as np
import torch
import torch.nn.functional as F

###################################################################
###################################################################

#################### Passive Cavitation Class #####################

class Cavitation(LinearPhysics):
    def __init__(self, img_size, n_sensors, batch_size, dtype, x_lim, y_lim, z_lim, c_speed, fs, device = "cpu", **kwargs):
        ### img_size (C x H x W) C: time samples H: depth azimuthal pixels, W: Lateral pixels
        super().__init__(**kwargs)
        self.img_size = img_size
        self.n_sensors = n_sensors
        self.batch_size = batch_size
        self.x_lim  = x_lim
        self.z_lim = z_lim
        self.device = device
        self.dtype = dtype
        self.norm = (self.img_size[1] * self.img_size[2])
        self.c_speed = c_speed
        self.fs = fs

        # ===========================
        # Image grid computation
        # ===========================
        x_grid = np.linspace(x_lim[0], x_lim[1], img_size[2])
        z_grid = np.linspace(z_lim[0], z_lim[1], img_size[1])
        X, Z = np.meshgrid(x_grid, z_grid)
        Y = np.full_like(X, y_lim)

        # ===========================
        # Sensor grid computation
        # ===========================
        self.center_sensor = n_sensors // 2
        z_transducer  = z_lim[0]  - 8*abs((z_grid[1] - z_grid[0]))
        pos_sens_x = torch.linspace(x_lim[0], x_lim[1], n_sensors, device=device, dtype=dtype)
        pos_sens_y = torch.full((n_sensors,), y_lim, dtype=dtype, device=device)
        pos_sens_z = torch.full((n_sensors,), z_transducer, dtype=dtype, device=device)


        X_vec = torch.tensor(X.ravel(), dtype=dtype, device=device)
        Z_vec = torch.tensor(Z.ravel(), dtype=dtype, device=device)
        Y_vec = torch.tensor(Y.ravel(), dtype=dtype, device=device)

        # ===========================
        # Distance and delays
        # ===========================
        dx = X_vec[:, None] - pos_sens_x[None, :]
        dz = Z_vec[:, None] - pos_sens_z[None, :]
        dy = Y_vec[:, None] - pos_sens_y[None, :]

        distances = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2).T  # (n_sensors, n_pixels_x*n_pixels_z)
        time_delays = distances / self.c_speed
        sample_delays = torch.clamp(torch.round(time_delays * self.fs), min=1).to(torch.int64)

        # ===========================
        # Expanded grid (aperture)
        # ===========================
        pitch = x_grid[1] - x_grid[0]
        self.n_pixels_x_exp = self.img_size[2] + 2 * self.center_sensor
        n_pixels_exp = self.n_pixels_x_exp * self.img_size[1]
        x_lim_exp = x_lim + np.array([-self.center_sensor, self.center_sensor]) * pitch
        x_grid_exp = np.linspace(x_lim_exp[0], x_lim_exp[1], self.n_pixels_x_exp)
        X2, Z2 = np.meshgrid(x_grid_exp, z_grid)
        Y2 = np.full_like(X2, y_lim)

        X2_vec = torch.tensor(X2.ravel(), dtype=dtype, device=device)
        Z2_vec = torch.tensor(Z2.ravel(), dtype=dtype, device=device)
        Y2_vec = torch.tensor(Y2.ravel(), dtype=dtype, device=device)

        dx2 = X2_vec[:, None] - pos_sens_x[None, :]
        dz2 = Z2_vec[:, None] - pos_sens_z[None, :]
        dy2 = Y2_vec[:, None] - pos_sens_y[None, :]

        distances_exp = torch.sqrt(dx2 ** 2 + dy2 ** 2 + dz2 ** 2).T  # (n_sensors, n_pixels_exp)
        time_delays_exp = distances_exp / self.c_speed
        sample_delays_exp = torch.clamp(torch.round(time_delays_exp * self.fs), min=1).to(torch.int64)


        # ===========================
        # Kernel construction
        # ===========================
        raw_kernel = (sample_delays_exp[self.center_sensor - 1, :]).reshape(self.img_size[1], self.n_pixels_x_exp)
        samples = torch.arange(1, self.img_size[0] + 1, dtype=torch.int64, device=device)
        lin_kernel = (raw_kernel.flatten()[:, None] == samples[None, :]).float()
        lin_kernel = lin_kernel.reshape(self.img_size[1], self.n_pixels_x_exp, self.img_size[0])

        circ_kernel = torch.roll(lin_kernel, shifts=1, dims=2)
        circ_kernel_flip = torch.flip(torch.flip(circ_kernel, dims=[0]), dims=[1])
        self.circ_kernel_flip = circ_kernel_flip.permute(2, 0, 1)  # (T, Z, X)
        self.circ_kernel_adj = torch.conj(torch.flip(self.circ_kernel_flip, dims=[0, 1, 2]))

    def A(self, x, **kwargs):
        #### x is of size b x C x H x W (batch x time x depth x width)
        #### y is of size b x 1 x C x S (batch x 1 x time x sensors)
        y = None
        my_image = torch.cat([x, x], dim=1)  # (batch, C, H, W_exp)
        for con in range(self.img_size[1]):
            image_slice = my_image[:, :, con, :]
            kernel_slice = self.circ_kernel_flip[:, -(con + 1), :]
            image_slice = image_slice.unsqueeze(1)  # (batch, 1, 2*C, X)
            kernel_slice = kernel_slice.unsqueeze(0).unsqueeze(0)  # (1, 1, C, X_exp)
            kernel_slice = torch.flip(kernel_slice, dims=[-2, -1])
            pad_h = kernel_slice.shape[-2] - 1  # C - 1
            pad_w = kernel_slice.shape[-1] - 1  # X_exp - 1
            padding = (pad_w, pad_w, pad_h, pad_h)
            image_slice = F.pad(image_slice, padding)
            conv_out = F.conv2d(image_slice, kernel_slice)
            conv_out = conv_out.squeeze(1)  # (batch, H_out, W_out)
            conv_out = conv_out / self.norm
            if y is None:
                y = conv_out
            else:
                y += conv_out
        self.y_big = y.clone()
        center_result_x = round(y.shape[2] / 2)
        y = y[:, self.img_size[0]: 2 * self.img_size[0], center_result_x - self.center_sensor: center_result_x + self.center_sensor]
        return y

    def A_adjoint(self, y, **kwargs):
        Y_padded = torch.zeros((self.batch_size, 3 * self.img_size[0] - 1, self.img_size[2] + self.n_pixels_x_exp - 1), device=self.device )
        center_result_x = round(Y_padded.shape[2] / 2)
        Y_padded[0, self.img_size[0]: 2 * self.img_size[0], center_result_x - self.center_sensor: center_result_x + self.center_sensor] = y
        my_image_est = torch.zeros((self.batch_size, 2 * self.img_size[0], self.img_size[1] , self.img_size[2]), device=self.device )  # (batch, 2*C, H, W)
        for con in range(self.img_size[1]):
            kernel_slice_adj = self.circ_kernel_adj[:, con, :]  # (C, W_exp)
            kernel_slice_adj = kernel_slice_adj.unsqueeze(0).unsqueeze(0)  # (1,1,H_k,W_k)
            kernel_slice_adj = torch.flip(kernel_slice_adj, dims=[-2, -1])
            y_in = Y_padded.unsqueeze(0)  # (1,1,H,W)
            conv_out = F.conv2d(y_in, kernel_slice_adj)  # (1,1,H_valid,W_valid)
            my_image_est[:, :, con, :] = conv_out.squeeze(0).squeeze(0) / self.norm

        x = my_image_est[:, :self.img_size[0], :, :] + my_image_est[:, self.img_size[0]: 2 * self.img_size[0], :, :]
        return x
