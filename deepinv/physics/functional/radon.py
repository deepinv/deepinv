import torch
from torch import nn
import torch.nn.functional as F

if torch.__version__ > "1.2.0":
    affine_grid = lambda theta, size: F.affine_grid(theta, size, align_corners=True)
    grid_sample = lambda input, grid, mode="bilinear": F.grid_sample(
        input, grid, align_corners=True, mode=mode
    )
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample


def fan_beam_grid(theta, image_size, fan_parameters, dtype=torch.float, device="cpu"):
    scale_factor = 2.0 / (image_size * fan_parameters["pixel_spacing"])
    n_detector_pixels = fan_parameters["n_detector_pixels"]
    source_radius = fan_parameters["source_radius"] * scale_factor
    detector_radius = fan_parameters["detector_radius"] * scale_factor
    detector_spacing = fan_parameters["detector_spacing"] * scale_factor
    detector_length = detector_spacing * (n_detector_pixels - 1)

    R = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=dtype,
        device=device,
    )
    base_grid = affine_grid(R, torch.Size([1, 1, n_detector_pixels, image_size]))
    x_vals = base_grid[0, 0, :, 0]
    dist_scaling = (
        0.5
        * detector_length
        * (x_vals + source_radius)
        / (source_radius + detector_radius)
    )
    base_grid[:, :, :, 1] *= dist_scaling[None, None, :]
    base_grid = base_grid.reshape(-1, 2)
    rot_matrix = torch.tensor(
        [[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]],
        dtype=dtype,
        device=device,
    )
    base_grid = base_grid @ rot_matrix.T
    base_grid = base_grid.reshape(1, n_detector_pixels, image_size, 2).transpose(1, 2)
    return base_grid


# constants
PI = 4 * torch.ones(1).atan()
SQRT2 = (2 * torch.ones(1)).sqrt()


def fftfreq(n):
    val = 1.0 / n
    results = torch.zeros(n)
    N = (n - 1) // 2 + 1
    p1 = torch.arange(0, N)
    results[:N] = p1
    p2 = torch.arange(-(n // 2), 0)
    results[N:] = p2
    return results * val


def deg2rad(x):
    return x * 4 * torch.ones(1, device=x.device, dtype=x.dtype).atan() / 180


class AbstractFilter(nn.Module):
    def __init__(self, device="cpu", dtype=torch.float):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x):
        input_size = x.shape[2]
        projection_size_padded = max(
            64, int(2 ** (2 * torch.tensor(input_size)).float().log2().ceil())
        )
        pad_width = projection_size_padded - input_size
        padded_tensor = F.pad(x, (0, 0, 0, pad_width))
        f = self._get_fourier_filter(padded_tensor.shape[2]).to(x.device)
        fourier_filter = self.create_filter(f)
        fourier_filter = fourier_filter.unsqueeze(-2)

        projection = (
            torch.view_as_real(torch.fft.fft(padded_tensor.transpose(2, 3))).transpose(
                2, 3
            )
            * fourier_filter
        )
        result = torch.view_as_real(
            torch.fft.ifft(torch.view_as_complex(projection).transpose(2, 3))
        )[..., 0]
        result = result.transpose(2, 3)[:, :, :input_size, :]

        return result

    def _get_fourier_filter(self, size):
        n = torch.cat(
            [torch.arange(1, size / 2 + 1, 2), torch.arange(size / 2 - 1, 0, -2)]
        )

        f = torch.zeros(size, dtype=self.dtype, device=self.device)
        f[0] = 0.25
        f[1::2] = -1 / (PI * n) ** 2

        fourier_filter = torch.view_as_real(torch.fft.fft(f, dim=-1))
        fourier_filter[:, 1] = fourier_filter[:, 0]

        return 2 * fourier_filter

    def create_filter(self, f):
        raise NotImplementedError


class RampFilter(AbstractFilter):
    def __init__(self, **kwargs):
        super(RampFilter, self).__init__(**kwargs)

    def create_filter(self, f):
        return f


class Radon(nn.Module):
    r"""
    Sparse Radon transform operator.


    :param int in_size: the size of the input image. If None, the size is inferred from the input image.
    :param torch.Tensor theta: the angles at which the Radon transform is computed. Default is ``torch.arange(180)``.
    :param bool circle: if ``True``, the input image is assumed to be a circle. Default is ``False``.
    :param bool parallel_computation: if ``True``, all projections are performed in parallel. Requires more memory but is faster on GPUs.
    :param bool fan_beam: If ``True``, use fan beam geometry, if ``False`` use parallel beam
    :param dict fan_parameters: Only used if fan_beam is ``True``. Contains the parameters defining the scanning geometry. The dict should contain the keys:

        - "pixel_spacing" defining the distance between two pixels in the image, default: 0.5 / (in_size)

        - "source_radius" distance between the x-ray source and the rotation axis (middle of the image), default: 57.5

        - "detector_radius" distance between the x-ray detector and the rotation axis (middle of the image), default: 57.5

        - "n_detector_pixels" number of pixels of the detector, default: 258

        - "detector_spacing" distance between two pixels on the detector, default: 0.077

        The default values are adapted from the geometry in `https://doi.org/10.5281/zenodo.8307932 <https://doi.org/10.5281/zenodo.8307932>`_,
        where pixel spacing, source and detector radius and detector spacing are given in cm.
        Note that a to small value of n_detector_pixels*detector_spacing can lead to severe circular artifacts in any reconstruction.
    :param torch.dtype dtype: the data type of the output. Default is torch.float.
    :param str, torch.device device: the device of the output. Default is torch.device('cpu').
    """

    def __init__(
        self,
        in_size=None,
        theta=None,
        circle=False,
        parallel_computation=True,
        fan_beam=False,
        fan_parameters=None,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.circle = circle
        self.theta = theta
        if theta is None:
            self.theta = torch.arange(180)
        self.dtype = dtype
        self.parallel_computation = parallel_computation
        self.fan_beam = fan_beam
        self.fan_parameters = fan_parameters
        if fan_beam:
            if self.fan_parameters is None:
                self.fan_parameters = {}
            if not "pixel_spacing" in self.fan_parameters.keys():
                assert (
                    not in_size is None
                ), "Either input size or pixel spacing have to be given"
                self.fan_parameters["pixel_spacing"] = 0.5 / in_size
            if not "source_radius" in self.fan_parameters.keys():
                self.fan_parameters["source_radius"] = 57.5
            if not "detector_radius" in self.fan_parameters.keys():
                self.fan_parameters["detector_radius"] = 57.5
            if not "n_detector_pixels" in self.fan_parameters.keys():
                self.fan_parameters["n_detector_pixels"] = 258
            if not "detector_spacing" in self.fan_parameters.keys():
                self.fan_parameters["detector_spacing"] = 0.077
        self.all_grids = None
        if in_size is not None:
            self.all_grids = self._create_grids(self.theta, in_size, circle).to(device)
            if self.parallel_computation:
                self.all_grids_par = torch.cat(
                    [self.all_grids[i] for i in range(len(self.theta))], 2
                )

    def forward(self, x):
        r"""

        :param torch.Tensor x: the input image.
        """
        N, C, W, H = x.shape
        assert W == H, "Input image must be square"

        if (
            self.all_grids is None
        ):  # if in_size was not given, we have to create the grid online.
            self.all_grids = self._create_grids(
                self.theta, W, self.circle, device=x.device
            )
            if self.parallel_computation:
                self.all_grids_par = torch.cat(
                    [self.all_grids[i] for i in range(len(self.theta))], 2
                )

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            x = F.pad(x, (pad_width[0], pad_width[1], pad_width[0], pad_width[1]))

        N, C, W, _ = x.shape

        if self.parallel_computation:
            rotated_par = grid_sample(
                x, self.all_grids_par.repeat(N, 1, 1, 1).to(x.device)
            )
            out = (
                rotated_par.sum(2).reshape(N, C, len(self.theta), -1).transpose(-2, -1)
            )
        else:
            out = torch.zeros(
                N,
                C,
                self.all_grids[0].shape[-2],
                len(self.theta),
                device=x.device,
                dtype=self.dtype,
            )

            for i in range(len(self.theta)):
                rotated = grid_sample(
                    x, self.all_grids[i].repeat(N, 1, 1, 1).to(x.device)
                )
                out[..., i] = rotated.sum(2)
        return out

    def _create_grids(self, angles, grid_size, circle, device="cpu"):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for theta in angles:
            theta = deg2rad(theta)
            if self.fan_beam:
                all_grids.append(
                    fan_beam_grid(
                        theta,
                        grid_size,
                        self.fan_parameters,
                        dtype=self.dtype,
                        device=device,
                    )
                )
            else:
                R = torch.tensor(
                    [[[theta.cos(), theta.sin(), 0], [-theta.sin(), theta.cos(), 0]]],
                    dtype=self.dtype,
                    device=device,
                )
                all_grids.append(
                    affine_grid(R, torch.Size([1, 1, grid_size, grid_size]))
                )
        return torch.stack(all_grids)


class IRadon(nn.Module):
    r"""
    Inverse sparse Radon transform operator.


    :param int in_size: the size of the input image. If None, the size is inferred from the input image.
    :param torch.Tensor theta: the angles at which the Radon transform is computed. Default is torch.arange(180).
    :param bool circle: if True, the input image is assumed to be a circle. Default is False.
    :param use_filter: if True, the ramp filter is applied to the input image. Default is True.
    :param int out_size: the size of the output image. If None, the size is the same as the input image.
    :param bool parallel_computation: if True, all projections are performed in parallel. Requires more memory but is faster on GPUs.
    :param torch.dtype dtype: the data type of the output. Default is torch.float.
    :param str, torch.device device: the device of the output. Default is torch.device('cpu').
    """

    def __init__(
        self,
        in_size=None,
        theta=None,
        circle=False,
        use_filter=True,
        out_size=None,
        parallel_computation=True,
        dtype=torch.float,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.circle = circle
        self.device = device
        self.theta = theta if theta is not None else torch.arange(180).to(self.device)
        self.out_size = out_size
        self.in_size = in_size
        self.parallel_computation = parallel_computation
        self.dtype = dtype
        self.ygrid, self.xgrid, self.all_grids = None, None, None
        if in_size is not None:
            self.ygrid, self.xgrid = self._create_yxgrid(in_size, circle)
            self.all_grids = self._create_grids(self.theta, in_size, circle).to(
                self.device
            )
            if self.parallel_computation:
                self.all_grids_par = torch.cat(
                    [self.all_grids[i] for i in range(len(self.theta))], 2
                )
        self.filter = (
            RampFilter(dtype=self.dtype, device=self.device)
            if use_filter
            else lambda x: x
        )

    def forward(self, x, filtering=True):
        r"""

        :param torch.Tensor x: the input image.
        :param bool filtering: if True, the ramp filter is applied to the input image. Default is True.
        """
        it_size = x.shape[2]
        ch_size = x.shape[1]

        if self.in_size is None:
            self.in_size = (
                int((it_size / SQRT2).floor()) if not self.circle else it_size
            )
        # if None in [self.ygrid, self.xgrid, self.all_grids]:
        if self.ygrid is None or self.xgrid is None or self.all_grids is None:
            self.ygrid, self.xgrid = self._create_yxgrid(self.in_size, self.circle)
            self.all_grids = self._create_grids(self.theta, self.in_size, self.circle)
            if self.parallel_computation:
                self.all_grids_par = torch.cat(
                    [self.all_grids[i] for i in range(len(self.theta))], 2
                )

        x = self.filter(x) if filtering else x

        if self.parallel_computation:
            reco = grid_sample(x, self.all_grids_par.repeat(x.shape[0], 1, 1, 1))
            reco = reco.reshape(x.shape[0], ch_size, it_size, len(self.theta), it_size)
            reco = reco.sum(-2)
        else:
            reco = torch.zeros(
                x.shape[0],
                ch_size,
                it_size,
                it_size,
                device=self.device,
                dtype=self.dtype,
            )
            for i_theta in range(len(self.theta)):
                reco += grid_sample(
                    x, self.all_grids[i_theta].repeat(reco.shape[0], 1, 1, 1)
                )

        if not self.circle:
            W = self.in_size
            diagonal = it_size
            pad = int(torch.tensor(diagonal - W, dtype=torch.float).ceil())
            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center
            pad_width = (pad_before, pad - pad_before)
            reco = F.pad(
                reco, (-pad_width[0], -pad_width[1], -pad_width[0], -pad_width[1])
            )

        if self.circle:
            reconstruction_circle = (self.xgrid**2 + self.ygrid**2) <= 1
            reconstruction_circle = reconstruction_circle.repeat(
                x.shape[0], ch_size, 1, 1
            )
            reco[~reconstruction_circle] = 0.0

        reco = reco * PI.item() / (2 * len(self.theta))

        if self.out_size is not None:
            pad = (self.out_size - self.in_size) // 2
            reco = F.pad(reco, (pad, pad, pad, pad))

        return reco

    def _create_yxgrid(self, in_size, circle):
        if not circle:
            in_size = int((SQRT2 * in_size).ceil())
        unitrange = torch.linspace(-1, 1, in_size, dtype=self.dtype, device=self.device)
        return torch.meshgrid(unitrange, unitrange, indexing="ij")

    def _XYtoT(self, theta):
        T = self.xgrid * (deg2rad(theta)).cos() - self.ygrid * (deg2rad(theta)).sin()
        return T

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())
        all_grids = []
        for i_theta in range(len(angles)):
            X = (
                torch.ones(grid_size, dtype=self.dtype, device=self.device)
                .view(-1, 1)
                .repeat(1, grid_size)
                * i_theta
                * 2.0
                / (len(angles) - 1)
                - 1.0
            )
            Y = self._XYtoT(angles[i_theta])
            all_grids.append(
                torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1).unsqueeze(0)
            )
        return torch.stack(all_grids)
