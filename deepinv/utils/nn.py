import os
from warnings import warn
import numpy as np
import torch


def get_freer_gpu(verbose=True):
    """
    Returns the GPU device with the most free memory.

    Use in conjunction with ``torch.cuda.is_available()``.
    Attempts to use ``nvidia-smi`` with ``bash``, if these don't exist then uses torch commands to get free memory.

    :param bool verbose: print selected GPU index and memory
    :return torch.device device: selected torch cuda device.
    """
    try:
        os.system(
            "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"
            if os.name == "posix"
            else 'bash -c "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"'
        )
        memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        idx, mem = np.argmax(memory_available), np.max(memory_available)
        device = torch.device(f"cuda:{idx}")

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
