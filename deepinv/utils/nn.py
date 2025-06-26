import os
import subprocess
from warnings import warn
import numpy as np
import torch

_warn_msg_cuda_order = (
    "Your attempt to choose GPU was not reliable because the environment variable "
    "'CUDA_DEVICE_ORDER' either is not set or not equal to 'PCI_BUS_ID' (see also the doc). "
    "Before loading Pytorch you can set the variable in your shell using the command: "
    "'export CUDA_DEVICE_ORDER=PCI_BUS_ID' "
    "or use IPython's 'magic' command '%env CUDA_DEVICE_ORDER=PCI_BUS_ID'."
)


def get_freer_gpu(verbose=True):
    """
    Returns the GPU device with the most free memory.

    Use in conjunction with ``torch.cuda.is_available()``.
    Attempts to use ``nvidia-smi`` with ``bash``, if these don't exist then uses torch commands to get free memory.

    :param bool verbose: print selected GPU index and memory
    :return torch.device device: selected torch cuda device

    .. warning::
        GPU indices in ``nvidia-smi`` may not match those in Pytorch if in your environment ``CUDA_DEVICE_ORDER``
        is not set to ``PCI_BUS_ID``. If the above variable is not set, the call will generate a warning but will
        still return a device. We strongly recommend setting this variable as above prior to loading Pytorch's
        runtime API.

    """
    try:
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

        env = os.environ.copy()
        if "CUDA_DEVICE_ORDER" not in env or env["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
            warn(_warn_msg_cuda_order)
    except:
        if torch.cuda.device_count() == 0:
            warn("Couldn't find free GPU")
            return torch.device(f"cuda")

        else:
            # Note this is slower and will return slightly different values to nvidia-smi
            idx, mem = max(
                (
                    (d, torch.cuda.mem_get_info(d)[0] / 1048576)
                    for d in range(torch.cuda.device_count())
                ),
                key=lambda x: x[1],
            )
            device = torch.device(f"cuda:{idx}")

    if verbose:
        print(f"Selected GPU {idx} with {mem} MiB free memory ")

    return device
