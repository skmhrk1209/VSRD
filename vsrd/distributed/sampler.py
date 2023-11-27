import operator

import torch


class DistributedSampler(torch.utils.data.DistributedSampler):
    """ Wrapper class over `torch.utils.data.Sampler` for distributed training

    References:
        - [Code](https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py#L681)
    """

    class IndexDataset(torch.utils.data.Dataset):

        def __init__(self, sampler):
            self.indices = list(sampler)

        def __getitem__(self, index):
            return self.indices[index]

        def __len__(self):
            return len(self.indices)

    def __init__(self, sampler, *args, **kwargs):
        dataset = __class__.IndexDataset(sampler)
        super().__init__(dataset, *args, **kwargs)
        self.sampler = sampler

    def __iter__(self):
        indices = super().__iter__()
        dataset = __class__.IndexDataset(self.sampler)
        return iter(operator.itemgetter(*indices)(dataset))
