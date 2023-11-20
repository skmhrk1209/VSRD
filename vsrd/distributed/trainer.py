import contextlib

import torch.nn as nn

from . import utils


class DistributedTrainer(contextlib.ContextDecorator):

    def __init__(self, models, optimizer, closure=None):
        self.models = models
        self.optimizer = optimizer
        self.closure = closure

    def __enter__(self):
        self.optimizer.zero_grad()
        for model in self.models:
            if not isinstance(model, nn.parallel.DistributedDataParallel):
                utils.broadcast_tensors(model.buffers())
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for model in self.models:
            if not isinstance(model, nn.parallel.DistributedDataParallel):
                utils.average_gradients(model.parameters())
        self.optimizer.step(self.closure)
