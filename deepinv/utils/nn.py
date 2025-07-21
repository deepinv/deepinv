import os
import subprocess
from warnings import warn
import numpy as np
import torch


def _get_freer_gpu_torch(hide_warnings=False):
    """
    Uses PyTorch's Runtime API to select GPU with most of free VRAM.

    :return tuple (device, idx, mem): cuda device,  index and its free memory in GB; all
    `None, None, None` are returned if no cuda devices were detected
    """
    if torch.cuda.device_count() == 0:
        if not hide_warnings:
            warn("Couldn't find free GPU")
        return None, None, None

    idx, mem = max(
        (
            (d, torch.cuda.mem_get_info(d)[0] / 1048576)
            for d in range(torch.cuda.device_count())
        ),
        key=lambda x: x[1],
    )
    device = torch.device(f"cuda:{idx}")
    return device, idx, mem


def _get_freer_gpu_system(hide_warnings=False):
    """
    Uses Nvidia driver `nvidia-smi` to select GPU with most of free VRAM.
    This is faster than using PyTorch's API, but may be unreliable if some environment
    variables are not set.

    :param boolean hide_warnings: if True environment variables are not checked and
    all warning messages are supressed (default `False`)
    :return tuple (device, idx, mem): cuda device,  index and its free memory in GB:
    """

    env = os.environ.copy()  # do not to change environment in any case
    if (
        "CUDA_DEVICE_ORDER" not in env.keys()
        or env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        if not hide_warnings:
            warn(
                "Your attempt to choose GPU was not reliable because the environment variable "
                "'CUDA_DEVICE_ORDER' either is not set or is not equal to 'PCI_BUS_ID' (see also the doc). "
                "Before loading PyTorch you can set the variable in your shell using the command: "
                "'export CUDA_DEVICE_ORDER=PCI_BUS_ID' or use IPython's 'magic' command "
                "'%env CUDA_DEVICE_ORDER=PCI_BUS_ID'."
            )

    pipeline = "nvidia-smi -q -d Memory | grep -A5 GPU | grep Free"
    if os.name == "posix":
        shell = True
    else:
        pipeline = ["bash", "-c", pipeline]
        shell = False
    proc = subprocess.run(pipeline, shell=shell, capture_output=True, text=True)
    stdout = proc.stdout
    lines = stdout.splitlines()
    memory_available = [int(line.split()[2]) for line in lines]
    idx, mem = np.argmax(memory_available), np.max(memory_available)
    device = torch.device(f"cuda:{idx}")

    return device, idx, mem


def get_freer_gpu(verbose=True, use_torch_api=True, hide_warnings=False):
    """
    Returns the GPU device with the most free memory.

    Use in conjunction with ``torch.cuda.is_available()``.

    If `use_torch_api=True` then attempts to select GPU using only torch commands, otherwise
    uses system driver to detect GPUs (via `nvidia-smi` command). The first method may be slower
    but is more reliable as the former depends on environment settings.
    If system method is chosen and fails, the call falls back to using torch commands and a warning
    is printed. If no CUDA devices are detected, then `None` is returned.

    :param bool verbose: print selected GPU index and memory
    :param bool use_torch_api: use torch commands if True, or Nvidia driver otherwise
    :param bool hide_warnings: supress all warnings for all methods
    :return torch.device device: selected cuda device

    .. warning::
        GPU indices in ``nvidia-smi`` may not match those in PyTorch if in your environment ``CUDA_DEVICE_ORDER``
        is not set to ``PCI_BUS_ID``:
        https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150
        If the variable is not set or has different value, the call to will print a warning
        (if not supressed with `hide_warnings=True`) but will not change the device.

    """
    if use_torch_api:
        device, idx, mem = _get_freer_gpu_torch(hide_warnings)
    else:
        try:
            device, idx, mem = _get_freer_gpu_system(hide_warnings)
        except:
            warn(
                "an exception occured when selecting GPU using nvidia-driver (nvidia-smi) "
                "falling back to direct PyTorch's runtime API"
            )
            device, idx, mem = _get_freer_gpu_torch(hide_warnings=False)
    if verbose:
        print(f"Selected GPU {idx} with {mem} MiB free memory")

    return device
