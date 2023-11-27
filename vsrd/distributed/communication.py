import io
import pickle

import torch
import torch.nn as nn


def object_to_tensor(object, device):
    bytes = io.BytesIO()
    pickle.dump(object, bytes)
    storage = torch.ByteStorage.from_buffer(bytes.getvalue())
    tensor = torch.ByteTensor(storage).to(device)
    return tensor


def tensor_to_object(tensor):
    bytes = tensor.detach().cpu().numpy().tobytes()
    object = pickle.load(io.BytesIO(bytes))
    return object


def broadcast(object, src_rank=0, **kwargs):
    if isinstance(object, torch.Tensor):
        torch.distributed.broadcast(object, src=src_rank, **kwargs)
    else:
        device = torch.cuda.current_device()
        if torch.distributed.get_rank() != src_rank:
            shape = torch.empty(1, dtype=torch.long, device=device)
        else:
            tensor = object_to_tensor(object, device=device)
            shape = torch.tensor(tensor.shape, dtype=torch.long, device=device)
        torch.distributed.broadcast(shape, src=src_rank, **kwargs)
        if torch.distributed.get_rank() != src_rank:
            tensor = torch.empty(shape, dtype=torch.uint8, device=device)
        torch.distributed.broadcast(tensor, src=src_rank, **kwargs)
        if torch.distributed.get_rank() != src_rank:
            object = tensor_to_object(tensor)
    return object


def all_gather(object, **kwargs):

    def all_gather_impl(tensor, **kwargs):
        world_size = torch.distributed.get_world_size()
        shape = tensor.new_tensor(tensor.shape, dtype=torch.long)
        shapes = [torch.empty_like(shape) for _ in range(world_size)]
        torch.distributed.all_gather(shapes, shape, **kwargs)
        if torch.all(torch.stack(shapes) == shapes[0]):
            tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(tensors, tensor, **kwargs)
        else:
            max_shape = torch.max(torch.stack(shapes, dim=0), dim=0).values
            padding = sum((
                (0, max_size - size) for size, max_size
                in reversed(list(zip(shape, max_shape)))
            ), ())
            padded_tensor = nn.functional.pad(tensor, padding)
            padded_tensors = [torch.empty_like(padded_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(padded_tensors, padded_tensor, **kwargs)
            def slice(tensor, shape):
                assert tensor.ndim == len(shape)
                for dim, size in enumerate(shape):
                    tensor = torch.narrow(tensor, dim, 0, size)
                return tensor
            tensors = list(map(slice, padded_tensors, shapes))
        return tensors

    if isinstance(object, torch.Tensor):
        objects = all_gather_impl(object, **kwargs)
    else:
        device = torch.cuda.current_device()
        tensor = object_to_tensor(object, device=device)
        tensors = all_gather_impl(tensor, **kwargs)
        objects = list(map(tensor_to_object, tensors))

    return objects
