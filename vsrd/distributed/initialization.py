import os
import socket

import torch


def init_process_group(backend, port):

    from mpi4py import MPI

    world_size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    hostname = socket.gethostname()
    address = socket.gethostbyname(hostname)
    env_vars = dict(MASTER_ADDR=address, MASTER_PORT=str(port))

    MPI.COMM_WORLD.barrier()

    env_vars = MPI.COMM_WORLD.bcast(env_vars, root=0)

    MPI.COMM_WORLD.barrier()

    env_vars.update(WORLD_SIZE=str(world_size), RANK=str(rank))
    os.environ.update(env_vars)

    torch.distributed.init_process_group(backend=backend)
