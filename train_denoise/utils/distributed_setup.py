import os
import subprocess
import socket

import torch
import torch.distributed as dist


def setup_distributed(seed):
    # number of nodes / node ID
    n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    node_id = int(os.environ["SLURM_NODEID"])

    # local rank on the current node / global rank
    local_rank = int(os.environ["SLURM_LOCALID"])
    global_rank = int(os.environ["SLURM_PROCID"])

    # number of processes / GPUs per node
    world_size = int(os.environ["SLURM_NTASKS"])
    n_gpu_per_node = world_size // n_nodes

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    master_addr = hostnames.split()[0].decode("utf-8")

    # set environment variables for 'env://'
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(29500)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node = n_nodes > 1
    multi_gpu = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if multi_gpu:
        print("Initializing PyTorch distributed ...")
        dist.init_process_group(
            init_method="env://",
            backend="nccl",
        )

    # Set the global random seed from pytorch to ensure reproducibility of the example
    torch.manual_seed(seed=seed)

    return device, global_rank
