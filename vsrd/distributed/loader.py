import torch


class DistributedDataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, *args, sampler=None, batch_sampler=None, **kwargs):
        if not sampler and not batch_sampler:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        super().__init__(dataset, *args, sampler=sampler, batch_sampler=batch_sampler, **kwargs)
