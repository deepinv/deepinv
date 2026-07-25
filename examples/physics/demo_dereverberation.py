r"""
Reverberation operator for audio dereverberation.
===================================================

In this example we show how to use the :class:`deepinv.physics.Reverberation` forward
model together with :class:`deepinv.physics.generator.RIRGenerator` to simulate
single-channel audio dereverberation problems, following
:footcite:t:`lemercier2023diffusion`.

Recording a dry (anechoic) audio signal :math:`x` in a room adds reverberation to
it: every wall reflection produces a delayed and attenuated copy of the signal,
so that the recorded signal :math:`y` can be modeled as the convolution of
:math:`x` with the room impulse response (RIR) :math:`h` of the room, plus some
measurement noise :math:`n`

.. math::

    y = h * x + n,

where :math:`*` denotes a causal 1D convolution (see
:class:`deepinv.physics.Reverberation`). Recovering :math:`x` from :math:`y` (and
possibly :math:`h`) is the *dereverberation* problem.

RIRs can be simulated with the room acoustics simulator
`pyroomacoustics <https://github.com/LCAV/pyroomacoustics>`_
:footcite:t:`scheibler2018pyroomacoustics`, wrapped here by
:class:`deepinv.physics.generator.RIRGenerator`, which samples a random shoebox room
(dimensions, reverberation time :math:`T_{60}`, microphone/source positions) at every
call.

.. note::

    This example requires the optional dependencies `pyroomacoustics` (RIR
    simulation) and `sounddevice` (audio playback when running this example as a
    script), which can be installed with ``pip install deepinv[audio]`` or
    ``pip install pyroomacoustics sounddevice``.

"""

import torch
import matplotlib.pyplot as plt
from IPython.display import Audio, display

import deepinv as dinv

device = dinv.utils.get_device()

fs = 16000  # sampling frequency (Hz)
duration = 5.0  # signal duration (s)
T = int(duration * fs)

# %%
# Simulate a room impulse response
# ---------------------------------
#
# We use :class:`deepinv.physics.generator.RIRGenerator` to simulate the RIR of a
# random shoebox room with a reverberation time :math:`T_{60}` sampled between 0.4s
# and 1.0s.

rir_generator = dinv.physics.generator.RIRGenerator(
    filter_length=fs, fs=fs, t60_range=(0.4, 1.0), device=device
)
rir_params = rir_generator.step(batch_size=1, seed=0)
h = rir_params["filter"]

# %%
# Define the forward model
# -------------------------
#
# The :class:`deepinv.physics.Reverberation` operator convolves a dry signal with the
# simulated RIR, and we add a small amount of Gaussian measurement noise to the
# reverberant signal.

physics = dinv.physics.Reverberation(
    filter=h,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.1),
)

# %%
# Generate a toy dry signal and the associated measurement
# ----------------------------------------------------------
#
# We build a toy "speech-like" dry signal made of a few short chirps separated by
# silences, and apply the reverberation operator to it.

t = torch.arange(T, device=device) / fs
x = torch.zeros(1, 1, T, device=device)
for start, stop, f0, f1 in [(0.05, 0.25, 300, 900), (0.45, 0.65, 500, 200)]:
    mask = (t >= start) & (t < stop)
    local_t = t[mask] - start
    chirp = torch.sin(
        2 * torch.pi * (f0 * local_t + (f1 - f0) / (2 * (stop - start)) * local_t**2)
    )
    window = torch.hann_window(mask.sum().item(), device=device)
    x[0, 0, mask] = chirp * window

y = physics(x)

# %%
# A simple dereverberation baseline
# -----------------------------------
#
# As a simple dereverberation baseline (assuming the RIR :math:`h` is known), we
# compute the least-squares pseudo-inverse reconstruction of the reverberation
# operator, obtained with a few iterations of conjugate gradient.
#
# .. tip::
#
#     For a more advanced reconstruction, using a learned prior over clean speech
#     signals, see :class:`deepinv.sampling.DPS`, which implements the diffusion
#     posterior sampling method proposed in :footcite:t:`lemercier2023diffusion` (this
#     requires a pretrained diffusion model over the signals of interest).

x_hat = physics.A_dagger(y)

# %%
# Plot the results
# ------------------

fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=False)
axs[0].plot(t.cpu(), x[0, 0].cpu())
axs[0].set_title("Dry signal $x$")
axs[1].plot(torch.arange(h.shape[-1]).cpu() / fs, h[0, 0].cpu())
axs[1].set_title("Simulated room impulse response $h$")
axs[2].plot(t.cpu(), y[0, 0].cpu())
axs[2].set_title("Reverberant + noisy measurement $y$")
axs[3].plot(t.cpu(), x_hat[0, 0].detach().cpu())
axs[3].set_title(r"Dereverberation baseline $\hat{x}=A^{\dagger}y$")
for ax in axs:
    ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

# %%
# Listen to the signals
# ------------------------
#
# We can also listen to the dry, reverberant and dereverberated signals: in a
# notebook, this displays a playable widget for each signal, while running this
# example as a script also plays them out loud through the default audio device
# (using `sounddevice <https://python-sounddevice.readthedocs.io>`_).

try:
    import sounddevice as sd
except (ImportError, OSError):
    # sounddevice not installed, or its PortAudio backend is unavailable
    sd = None


def play(signal):
    if sd is None:
        return
    try:
        sd.play(signal, fs)
        sd.wait()
    except Exception:
        pass  # no audio output device available, e.g. when building the docs


signals = {
    "Dry signal x": x[0, 0].cpu().numpy(),
    "Reverberant + noisy measurement y": y[0, 0].cpu().numpy(),
    "Dereverberation baseline x_hat": x_hat[0, 0].detach().cpu().numpy(),
}
for name, signal in signals.items():
    print(name)
    display(Audio(signal, rate=fs))
    play(signal.clip(-1.0, 1.0))

# %%
# :References:
#
# .. footbibliography::
