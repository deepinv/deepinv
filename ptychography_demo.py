# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import deepinv as dinv
import matplotlib.pyplot as plt
import torch
import numpy as np
from functools import partial
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot



device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# %%
size = 128

url = get_image_url("CBSD_0010.png")
image = load_url_image(url, grayscale=False, img_size=(size, size))

# image = plt.imread('985-128x128.jpg')
n_img = 10*10
image = image[:, 0, ...].unsqueeze(1)  # Take only one channel
print(image.shape)
plot([image], figsize=(10, 10))


# %%
phase = image / image.max() * np.pi  # between 0 and pi
input = torch.exp(1j * phase.to(torch.complex64)).to(device)

# %%
# Initialize forward operators
def shift(x, x_shift, y_shift, pad_zeros=True):
    x = torch.roll(x, (x_shift, y_shift), dims=(-2, -1))

    if pad_zeros:
        if x_shift < 0:
            x[..., x_shift:, :] = 0
        elif x_shift > 0:
            x[..., 0:x_shift, :] = 0
        if y_shift < 0:
            x[..., :, y_shift:] = 0
        elif y_shift > 0:
            x[..., :, 0:y_shift] = 0
    return x


def get_overlap_img(probe, shifts):
    overlap_img = torch.zeros_like(probe, dtype=torch.float32)
    for (x_shift, y_shift) in shifts:
        overlap_img += torch.abs(shift(probe, x_shift, y_shift))**2
    return overlap_img

# %%
class PtychographyLinearOperator(dinv.physics.LinearPhysics):
    r"""
        Linear forward operator for the Ptychography phase retrival class, corresponding to the operator

        .. math::

            B_l(x) = F(p_l \times x) \in \mathbb{C}^{n \times n}.

        where :math:`F` is the 2D Fourier transform and :math:`p_l` is a probe function shifted by :math:`l`.
        The probe operator is applied element-wise on the input image :math:`x \in \mathbb{C}^{n \times n}`.

        :param tuple in_shape: shape of the input image.
        :param probe: probe function viewed as a 2D tensor.
        :param str probe_type: type of the probe, used if probe is not provided.
        :param int probe_radius: radius of the probe, used if probe is not provided.
        :param array_like shifts: shifts of the probe.
        :param int fov: field of view, used if shifts is not provided to compute it.
        :param int n_img: number of images, used if shifts not provided, equal to the number of shifts otherwise. Must be a perfect square.
        :param device: cpu or gpu.

            """

    def __init__(self, in_shape=None, probe=None, shifts=None,
                 probe_type=None, probe_radius=None,  # probe parameters
                 fov=None, n_img: int = 25, device="cpu", **kwargs):

        super().__init__(**kwargs)

        self.device = device

        if probe is not None:
            self.probe = probe
            self.in_shape = in_shape if in_shape is not None else probe.shape
        else:
            self.in_shape = in_shape
            self.probe_type = probe_type
            self.probe_radius = probe_radius
            self.probe = self.construct_probe(type=probe_type, probe_radius=probe_radius)

        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            self.fov = fov
            probe_radius = 0
            self.shifts = self.generate_shifts(size=in_shape, n_img=n_img, probe_radius=probe_radius, fov=fov)

        self.probe = self.probe / get_overlap_img(self.probe, self.shifts).mean().sqrt()

    def A(self, x, **kwargs):
        op_fft2 = partial(torch.fft.fft2, norm="ortho")
        f = lambda x, x_shift, y_shift: op_fft2(self.probe * shift(x, x_shift, y_shift))
        return torch.cat([f(x, x_shift, y_shift) for (x_shift, y_shift) in self.shifts], dim=1)

    def A_adjoint(self, y, **kwargs):
        op_ifft2 = partial(torch.fft.ifft2, norm="ortho")
        g = lambda y, x_shift, y_shift: shift(self.probe*op_ifft2(y), x_shift, y_shift)
        return torch.cat([g(y, x_shift, y_shift) for (x_shift, y_shift) in self.shifts], dim=1).sum(dim=1, keepdim=True)

    def construct_probe(self, type='disk', probe_radius=10):
        if type == 'disk' or type is None:
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            probe = torch.zeros(self.in_shape, device=self.device)
            probe[torch.sqrt((X - self.in_shape[0] // 2) ** 2 + (Y - self.in_shape[1] // 2) ** 2)
                  < probe_radius] = 1
        else:
            raise NotImplementedError(f'Probe type {type} not implemented')
        return probe

    def generate_shifts(self, size, n_img, probe_radius=10, fov=None):
        if fov is None:
            start_shift = -( size // 2  - probe_radius)
            end_shift = (size // 2  - probe_radius)
        else:
            start_shift = - fov // 2
            end_shift = fov // 2

        assert int(np.sqrt(n_img)) ** 2 == n_img, "n_img needs to be a perfect square"
        side_n_img = int(np.sqrt(n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        y_shifts, x_shifts = np.meshgrid(shifts, shifts, indexing='ij')
        return np.concatenate([x_shifts.reshape(n_img, 1), y_shifts.reshape(n_img, 1)], axis=1)

class Ptychography(dinv.physics.PhaseRetrieval):
    r"""
    Ptychography forward operator, corresponding to the operator

    .. math::

        \forw{x}_l = |F(p_l \times x)|^2.

    where :math:`F` is the 2D Fourier transform and :math:`p_l` is a probe function shifted by :math:`l`.
    The probe operator is applied element-wise on the input image :math:`x`.

    :param tuple in_shape: shape of the input image.
    :param probe: probe function viewed as a 2D tensor.
    :param str probe_type: type of the probe, used if probe is not provided.
    :param int probe_radius: radius of the probe, used if probe is not provided.
    :param array_like shifts: shifts of the probe.
    :param int fov: field of view, used if shifts is not provided to compute it.
    :param int n_img: number of images, used if shifts not provided, equal to the number of shifts otherwise. Must be a perfect square.
    :param device: cpu or gpu.

    """
    def __init__(self, in_shape=None, probe=None, shifts=None,
                 probe_type=None, probe_radius=None,  # probe parameters
                 fov=None, n_img: int = 25, device="cpu", **kwargs):

        B = PtychographyLinearOperator(in_shape=in_shape, probe=probe,
                                            shifts=shifts, probe_type=probe_type,
                                            probe_radius=probe_radius, fov=fov, n_img=n_img, device=device)
        self.probe = B.probe
        self.shifts = B.shifts
        self.device = device

        super().__init__(B, **kwargs)


physics = Ptychography(in_shape=(size, size), shifts=None, n_img=n_img, probe_type='disk', probe_radius=30, fov=170, device=device)

# x0 = torch.randn(1, 1, size, size, dtype=torch.complex64, device=device)
# norm_operator = physics.B.compute_norm(x0=x0)


overlap_img = get_overlap_img(physics.B.probe, physics.B.shifts).cpu()

# %%
# Plot probe
probe = physics.probe.cpu()
plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
plt.imshow(torch.abs(probe))
plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(torch.angle(probe))
# plt.colorbar()
plt.show()

# %%
# y = ptycho_fwd.apply(input)
y = physics(input)
print(y.shape)
# example_img = np.fft.fftshift(y[0, 50, :, :].cpu())
# # plt.imshow(np.log(np.abs(example_img)))
# plt.colorbar()
# plt.show()

# %%
# Gradient descent to find the original image
x_est = torch.randn(1, 1, size, size, dtype=torch.complex64, device=device)
x_est.requires_grad = True

n_iter = 200
lr = 0.1
for i in range(n_iter):
    # loss = torch.sum(torch.abs(ptycho_fwd.apply(x_est) - y))
    loss = torch.sum(torch.abs(physics(x_est) - y))
    loss.backward()
    with torch.no_grad():
        x_est -= lr * x_est.grad
        x_est.grad.zero_()
    if i % 10 == 0:
        print(f'Iter {i}, loss: {loss.item()}')

# %%
# Display the result
final_est = x_est.detach().cpu().squeeze()

# Make the mean phase equal to 0
average_phase = torch.angle(torch.mean(final_est))
final_est = torch.exp(1j * (torch.angle(final_est) - average_phase))

plt.imshow(torch.angle(final_est), cmap='gray')
plt.axis('off')  # Turn off axis labels
plt.colorbar()
plt.show()

