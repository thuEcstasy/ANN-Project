"""
Helpers for distributed training.
"""

import io
import os
import socket
import jittor as jt
import blobfile as bf

import torch.distributed as dist

# Change this to reflect your cluster layout.

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)
    
    dist.init_process_group(backend=backend, init_method="env://")

    jt.cuda(1)


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    return jt.load(path)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    pass


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
