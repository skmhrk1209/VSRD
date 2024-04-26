import contextlib

import torch
import torch.nn as nn
import tqdm as taqaddum

from .. import utils


def average_tensors(tensors):
    world_size = torch.distributed.get_world_size()
    for tensor in tensors:
        torch.distributed.all_reduce(tensor)
        tensor /= world_size


def average_gradients(parameters):
    world_size = torch.distributed.get_world_size()
    for parameter in parameters:
        if parameter.requires_grad and parameter.grad is not None:
            torch.distributed.all_reduce(parameter.grad)
            parameter.grad /= world_size


def broadcast_tensors(tensors, src_rank=0):
    for tensor in tensors:
        torch.distributed.broadcast(tensor, src_rank)


def broadcast_gradients(parameters, src_rank=0):
    for parameter in parameters:
        if parameter.requires_grad and parameter.grad is not None:
            torch.distributed.broadcast(parameter.grad, src_rank)


def get_device_id(num_devices_per_process=1, device_id_offset=0):
    num_local_devices = torch.cuda.device_count()
    num_local_processes = num_local_devices // num_devices_per_process
    local_rank = torch.distributed.get_rank() % num_local_processes
    device_id = local_rank * num_devices_per_process + device_id_offset
    return device_id


def get_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def get_sampler(loader):
    sampler = loader.batch_sampler.sampler if loader.batch_sampler is not None else loader.sampler
    return sampler if isinstance(sampler, torch.utils.data.DistributedSampler) else None


def get_logger(*args, rank=0, **kwargs):
    logger = utils.get_logger(*args, **kwargs)
    logger.addFilter(lambda _: torch.distributed.get_rank() == rank)
    return logger


def tqdm(iterable, *args, **kwargs):
    return taqaddum.tqdm(iterable, *args, **kwargs) if not torch.distributed.get_rank() else iterable


@contextlib.contextmanager
def barrier():
    torch.distributed.barrier()
    try:
        yield
    finally:
        torch.distributed.barrier()
