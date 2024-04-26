import inspect

import torch
import torch.nn as nn

from . import utils


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(
        self,
        module,
        device_id_offset=0,
        num_devices_per_process=1,
        convert_sync_batch_norm=True,
        sync_batch_norm_group_size=None,
        sync_batch_norm_num_groups=None,
        **kwargs,
    ):
        if convert_sync_batch_norm:
            process_group = None
            if sync_batch_norm_group_size or sync_batch_norm_num_groups:
                assert (
                    sync_batch_norm_group_size and not sync_batch_norm_num_groups or
                    not sync_batch_norm_group_size and sync_batch_norm_num_groups
                )
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                sync_batch_norm_group_size = sync_batch_norm_group_size or world_size // sync_batch_norm_num_groups
                assert not world_size % sync_batch_norm_group_size
                process_groups = list(map(list, torch.split(torch.arange(world_size), sync_batch_norm_group_size)))
                process_groups = list(map(torch.distributed.new_group, process_groups))
                process_group = process_groups[rank // sync_batch_norm_group_size]
            module = nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group=process_group)

        device_id = utils.get_device_id(num_devices_per_process, device_id_offset)

        super().__init__(module.to(device_id), [device_id], **kwargs)

    @classmethod
    def reinitialize(cls, instance, **kwargs):
        signature = inspect.signature(super().__init__)
        _, *parameters = signature.parameters.items()
        return cls(**dict({
            name: getattr(instance, name, parameter.default)
            for name, parameter in parameters
        }, **kwargs))
