# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import os
import time
import logging
import operator
import importlib
import itertools
import functools
import contextlib
import collections

import torch
import torch.nn as nn
import numpy as np


class Dict(dict):

    def getattr(self, key):
        super().__getattr__(key)

    def setattr(self, key, value):
        super().__setattr__(key, value)

    def delattr(self, key):
        super().__delattr__(key)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)

    @classmethod
    def apply(cls, dictionary):
        return apply(lambda element: cls(element) if isinstance(element, dict) else element, dictionary)


class DefaultDict(collections.defaultdict):

    def getattr(self, key):
        super().__getattr__(key)

    def setattr(self, key, value):
        super().__setattr__(key, value)

    def delattr(self, key):
        super().__delattr__(key)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dictionary):
        self.__dict__.update(dictionary)

    @classmethod
    def apply(cls, dictionary):
        return apply(lambda element: cls(element) if isinstance(element, dict) else element, dictionary)


class StatMeter(Dict):

    def update(self, **items):
        for key, value in items.items():
            stat = self.get(key, Dict(mean=value, variance=0, count=0))
            mean = (stat.mean * stat.count + value) / (stat.count + 1)
            variance = ((stat.mean ** 2 + stat.variance) * stat.count + value ** 2) / (stat.count + 1) - mean ** 2
            count = stat.count + 1
            self[key] = Dict(mean=mean, variance=variance, count=count)

    def means(self):
        for stat in self.values():
            yield stat.mean

    def variances(self):
        for stat in self.values():
            yield stat.variance

    def counts(self):
        for stat in self.values():
            yield stat.count


class SMAMeter(Dict):

    def update(self, **items):
        for key, value in items.items():
            stat = self.get(key, Dict(mean=value, count=0))
            mean = (stat.mean * stat.count + value) / (stat.count + 1)
            count = stat.count + 1
            self[key] = Dict(mean=mean, count=count)

    def means(self):
        for stat in self.values():
            yield stat.mean

    def counts(self):
        for stat in self.values():
            yield stat.count


class EMAMeter(Dict):

    def __init__(self, *args, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.setattr("momentum", momentum)

    def update(self, **items):
        for key, value in items.items():
            stat = self.get(key, Dict(mean=value, count=0))
            mean = stat.mean * self.momentum + value * (1 - self.momentum)
            count = stat.count + 1
            self[key] = Dict(mean=mean, count=count)

    def means(self):
        for stat in self.values():
            yield stat.mean

    def counts(self):
        for stat in self.values():
            yield stat.count


class RuntimeMeter(EMAMeter):

    def means(self):
        min_count = min(self.counts())
        for stat in self.values():
            yield stat.mean * stat.count / min_count


class ProgressMeter(RuntimeMeter):

    def __init__(self, num_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setattr("num_steps", num_steps)

    def progress(self):
        min_count = min(self.counts())
        return min_count / self.num_steps

    def elapsed_seconds(self):
        sum_mean = sum(self.means())
        return sum_mean * self.num_steps * self.progress()

    def arrival_seconds(self):
        sum_mean = sum(self.means())
        return sum_mean * self.num_steps * (1.0 - self.progress())


class StopWatch(object):

    def __init__(self):
        self.stack = []

    def start(self):
        self.stack.append(time.time())

    def stop(self):
        return time.time() - self.stack.pop()

    def restart(self):
        value = self.stop()
        self.start()
        return value


class Saver(object):

    def __init__(self, dirname):
        self.dirname = dirname

    def save(self, filename, **kwargs):
        os.makedirs(self.dirname, exist_ok=True)
        torch.save(kwargs, os.path.join(self.dirname, filename))


class ModeSwitcher(contextlib.ContextDecorator):

    def __init__(self, mode, *models):
        self.mode = mode
        self.models = models
        self.modes = {}

    def __enter__(self):
        for model in self.models:
            self.modes[model] = model.training
            model.train(self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        for model in self.models:
            model.train(self.modes.pop(model))
        assert not self.modes


class TrainSwitcher(ModeSwitcher):

    def __init__(self, *models):
        super().__init__(True, *models)


class EvalSwitcher(ModeSwitcher):

    def __init__(self, *models):
        super().__init__(False, *models)


class NormFreezer(contextlib.ContextDecorator):

    def __init__(self, mode, *models):
        self.mode = mode
        self.models = models
        self.modes = {}

    def __enter__(self):
        for model in self.models:
            for module in model.modules():
                if isinstance(module, (
                    nn.modules.batchnorm._BatchNorm,
                    nn.modules.instancenorm._InstanceNorm,
                )):
                    self.modes[module] = module.training
                    module.train(not self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        for model in self.models:
            for module in model.modules():
                if isinstance(module, (
                    nn.modules.batchnorm._BatchNorm,
                    nn.modules.instancenorm._InstanceNorm,
                )):
                    module.train(self.modes.pop(module))
        assert not self.modes


class ParameterFreezer(contextlib.ContextDecorator):

    def __init__(self, mode, *parameters):
        self.mode = mode
        self.parameters = parameters
        self.modes = {}

    def __enter__(self):
        for parameter in self.parameters:
            self.modes[parameter] = parameter.requires_grad
            parameter.requires_grad_(not self.mode)

    def __exit__(self, exception_type, exception_value, traceback):
        for parameter in self.parameters:
            parameter.requires_grad_(self.modes.pop(parameter))
        assert not self.modes


class RandomStateRestorer(contextlib.ContextDecorator):

    def __init__(self, mode=True):
        self.mode = mode

    def __enter__(self):
        self.rng_state = torch.get_rng_state()

    def __exit__(self, exception_type, exception_value, traceback):
        self.mode and torch.set_rng_state(self.rng_state)


class AveragedModel(torch.optim.swa_utils.AveragedModel):

    def __init__(self, *args, average_buffers=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.average_buffers = average_buffers

    @torch.no_grad()
    def update_parameters(self, model):
        super().update_parameters(model)
        for avaraged_buffer, model_buffer in zip(self.module.buffers(), model.buffers()):
            model_buffer = model_buffer.to(avaraged_buffer)
            num_averaged = self.n_averaged.to(avaraged_buffer)
            if self.average_buffers and self.n_averaged > 1:
                model_buffer = self.avg_fn(avaraged_buffer, model_buffer, num_averaged)
            avaraged_buffer.copy_(model_buffer, non_blocking=True)


class EMAModel(AveragedModel):

    def __init__(self, *args, momentum=0.9999, **kwargs):
        self.momentum = momentum
        super().__init__(*args, avg_fn=self.average, **kwargs)

    def average(self, averaged_parameter, model_parameter, num_averaged):
        # NOTE: `torch.lerp` does not support integer tensors
        # return torch.lerp(model_parameter, averaged_parameter, self.momentum)
        return averaged_parameter * self.momentum + model_parameter * (1.0 - self.momentum)


def import_function(name):
    module_name, function_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


def import_module(config, globals=None, locals=None):
    if isinstance(config, dict) and "function" in config:
        function = import_function(config.function)
        args = list(map(functools.partial(import_module, globals=globals, locals=locals), config.get("args", [])))
        kwargs = dict(zip(
            config.get("kwargs", {}).keys(),
            map(functools.partial(import_module, globals=globals, locals=locals), config.get("kwargs", {}).values()),
        ))
        return function(*args, **kwargs)
    if isinstance(config, (list, tuple, set)):
        return type(config)(map(functools.partial(import_module, globals=globals, locals=locals), config))
    if isinstance(config, dict):
        return type(config)(zip(config.keys(), map(functools.partial(import_module, globals=globals, locals=locals), config.values())))
    if isinstance(config, str) and config.split(":", 1)[0] == "eval":
        return eval(config.split(":", 1)[1], globals, locals)
    return config


def apply(function, element):
    if isinstance(element, (list, tuple, set)):
        element = type(element)(map(functools.partial(apply, function), element))
    if isinstance(element, dict):
        element = type(element)(zip(element.keys(), map(functools.partial(apply, function), element.values())))
    element = function(element)
    return element


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            iterator = iter(iterable)
            item = next(iterator)
        yield item


def pairwise(iterable):
    prevs, nexts = itertools.tee(iterable)
    next(nexts, None)
    return zip(prevs, nexts)


def pairwise_longest(iterable, fillvalue=None):
    prevs, nexts = itertools.tee(iterable)
    next(nexts, None)
    return itertools.zip_longest(prevs, nexts, fillvalue=fillvalue)


def compose(function, *functions):
    return lambda *args, **kwargs: (
        compose(*functions)(function(*args, **kwargs))
        if functions else function(*args, **kwargs)
    )


def multimap(functions, *iterables):
    return map(lambda f, *x: f(*x), functions, *iterables)


def map_innermost(function, sequence, classes=(list, tuple, set)):
    return type(sequence)(map(functools.partial(map_innermost, function, classes=classes), sequence)) if isinstance(sequence, classes) else function(sequence)


def to(element, *args, **kwargs):
    return apply(lambda element: element.to(*args, **kwargs) if isinstance(element, torch.Tensor) else element, element)


def tensor_args(dtype=None, device=None):
    def decorator(function):
        def wrapper(*args, **kwargs):
            args = tuple(map(functools.partial(torch.as_tensor, dtype=dtype, device=device), args))
            kwargs = dict(zip(kwargs.keys(), map(functools.partial(torch.as_tensor, dtype=dtype, device=device), kwargs.values())))
            return function(*args, **kwargs)
        return wrapper
    return decorator


def multi_dim(reducer):
    def multi_dim_reducer(inputs, dims, **kwargs):
        for dim in reversed(sorted(dims)):
            inputs = reducer(inputs, dim=dim, **kwargs)
        return inputs
    return multi_dim_reducer


def unsqueeze(inputs, *dims):
    return functools.reduce(torch.unsqueeze, dims, inputs)


def unsqueeze_as(source, target, start_dim=0):
    matched_dims = []
    matched_dim = start_dim
    for source_size in source.shape:
        matched_dim += target.shape[matched_dim:].index(source_size)
        matched_dims.append(matched_dim)
        matched_dim += 1
    shape = [
        target_size if dim in matched_dims else 1
        for dim, target_size in enumerate(target.shape)
    ]
    source = source.reshape(*shape)
    return source


def reversed_pad(inputs, padding, *args, **kwargs):
    padding += (0,) * (inputs.ndim * 2 - len(padding))
    padding = sum(reversed(list(itertools.islice(pairwise(padding), None, None, 2))), ())
    outputs = nn.functional.pad(inputs, padding, *args, **kwargs)
    return outputs


def linear_map(inputs, in_min=None, in_max=None, out_min=0.0, out_max=1.0):
    in_min = multi_dim(compose(torch.min, operator.itemgetter(0)))(inputs, dims=range(1, inputs.ndim), keepdim=True) if in_min is None else in_min
    in_max = multi_dim(compose(torch.max, operator.itemgetter(0)))(inputs, dims=range(1, inputs.ndim), keepdim=True) if in_max is None else in_max
    @tensor_args(dtype=inputs.dtype, device=inputs.device)
    def linear_map_impl(inputs, in_min, in_max, out_min, out_max):
        return out_min + (out_max - out_min) * (inputs - in_min) / (in_max - in_min)
    outputs = linear_map_impl(inputs, in_min, in_max, out_min, out_max)
    return outputs


def log_map(inputs, in_min=None, in_max=None, out_min=0.0, out_max=1.0):
    in_min = multi_dim(compose(torch.min, operator.itemgetter(0)))(inputs, dims=range(1, inputs.ndim), keepdim=True) if in_min is None else in_min
    in_max = multi_dim(compose(torch.max, operator.itemgetter(0)))(inputs, dims=range(1, inputs.ndim), keepdim=True) if in_max is None else in_max
    @tensor_args(dtype=inputs.dtype, device=inputs.device)
    def log_map_impl(inputs, in_min, in_max, out_min, out_max):
        return torch.log(torch.exp(out_min) + (torch.exp(out_max) - torch.exp(out_min)) * (inputs - in_min) / (in_max - in_min))
    outputs = log_map_impl(inputs, in_min, in_max, out_min, out_max)
    return outputs


def fuse_post_norm(module):

    added_chidren = []

    for (prev_name, prev_module), (next_name, next_module) in pairwise_longest(list(module.named_children()), (None, None)):

        if isinstance(prev_module, (nn.modules.conv._ConvNd, nn.Linear)):

            if isinstance(next_module, (
                nn.modules.batchnorm._BatchNorm,
                nn.modules.instancenorm._InstanceNorm,
            )):
                if not next_module.track_running_stats: continue

                def fuse_post_norm_impl(module, norm):

                    def compute_fused_weight(module, norm):
                        module_weight = module.weight
                        if isinstance(module, nn.modules.conv._ConvNd):
                            if module.transposed:
                                module_weight = module_weight.transpose(0, 1)
                        module_weight = module_weight * unsqueeze_as((
                            norm.weight *
                            torch.rsqrt(norm.running_var + norm.eps)
                        ), module_weight, start_dim=0)
                        if isinstance(module, nn.modules.conv._ConvNd):
                            if module.transposed:
                                module_weight = module_weight.transpose(0, 1)
                        return module_weight

                    def compute_fused_bias(module, norm):
                        module_bias = module.bias
                        if module_bias is None:
                            module_bias = module.weight.new_zeros(module.out_channels)
                        module_bias = norm.bias + (
                            (module_bias - norm.running_mean) *
                            norm.weight *
                            torch.rsqrt(norm.running_var + norm.eps)
                        )
                        return module_bias

                    module_weight = compute_fused_weight(module, norm)
                    module_bias = compute_fused_bias(module, norm)
                    module.weight = nn.Parameter(module_weight)
                    module.bias = nn.Parameter(module_bias)

                fuse_post_norm_impl(prev_module, next_module)

                # NOTE: replace normalizations with identity functions instead of removing
                added_chidren.append((next_name, nn.Identity()))

        module.add_module(prev_name, fuse_post_norm(prev_module))

    for name, child in added_chidren:
        module.add_module(name, child)

    return module


def convert_batch_norm_to_group_norm(module, num_groups=None, group_size=None, **kwargs):
    assert num_groups and not group_size or not num_groups and group_size
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_groups = num_groups or module.num_features // group_size
        assert not module.num_features % num_groups
        device, = set(tensor.device for tensor in module.state_dict().values())
        group_norm = nn.GroupNorm(num_groups, module.num_features, **kwargs).to(device)
        group_norm.weight = module.weight
        group_norm.bias = module.bias
        return group_norm
    for name, child in module.named_children():
        module.add_module(name, convert_batch_norm_to_group_norm(child, num_groups=num_groups, group_size=group_size, **kwargs))
    return module


def convert_group_norm_to_batch_norm(module, **kwargs):
    if isinstance(module, nn.GroupNorm):
        device, = set(tensor.device for tensor in module.state_dict().values())
        batch_norm = nn.BatchNorm2d(module.num_channels, **kwargs).to(device)
        batch_norm.weight = module.weight
        batch_norm.bias = module.bias
        return batch_norm
    for name, child in module.named_children():
        module.add_module(name, convert_group_norm_to_batch_norm(child, **kwargs))
    return module


def apply_spectral_norm(module, **kwargs):
    if isinstance(module, (nn.modules.conv._ConvNd, nn.Linear)):
        return nn.utils.spectral_norm(module, **kwargs)
    for name, child in module.named_children():
        module.add_module(name, apply_spectral_norm(child, **kwargs))
    return module


def vectorize(function):

    def vectorized(*args, **kwargs):

        def unstack(value):
            if isinstance(value, torch.Tensor):
                values = torch.unbind(value, dim=0)
            else:
                values = [value] * batch_size
            return values

        def stack(values):
            if all(isinstance(value, torch.Tensor) for value in values):
                value = torch.stack(values, dim=0)
            else:
                value, *_ = values
            return value

        batch_size, = {
            *(value.shape[0] for value in args if isinstance(value, torch.Tensor)),
            *(value.shape[0] for value in kwargs.values() if isinstance(value, torch.Tensor)),
        }

        args = list(map(unstack, args))
        kwargs = [list(zip(*map(unstack, item))) for item in kwargs.items()]

        values = [
            function(*list(args or []), **dict(kwargs or {}))
            for args, kwargs in itertools.zip_longest(zip(*args), zip(*kwargs))
        ]

        value_type, = set(map(type, values))

        if isinstance(value_type, (list, tuple, set)):
            values = value_type(map(stack, zip(*values)))
        elif isinstance(value_type, dict):
            values = [value.items() for value in values]
            values = value_type((tuple(map(stack, zip(*value))) for value in zip(*values)))
        else:
            values = stack(values)

        return values

    return vectorized


def unvectorize(function):

    def unvectorized(*args, **kwargs):

        def unsqueeze(value):
            if isinstance(value, torch.Tensor):
                value = torch.unsqueeze(value, dim=0)
            return value

        def squeeze(value):
            if isinstance(value, torch.Tensor):
                value = torch.squeeze(value, dim=0)
            return value

        args = list(map(unsqueeze, args))
        kwargs = dict(zip(kwargs.keys(), map(unsqueeze, kwargs.values())))

        values = function(*args, **kwargs)

        if isinstance(values, (list, tuple, set)):
            values = type(values)(map(squeeze, values))
        elif isinstance(values, dict):
            values = type(values)(zip(values.keys(), map(squeeze, values.values())))
        else:
            values = squeeze(values)

        return values

    return unvectorized


def torch_function(function):

    def wrapper(*args, **kwargs):

        def torch_to_numpy(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return value

        def numpy_to_torch(value):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            return value

        return apply(numpy_to_torch, function(
            *apply(torch_to_numpy, args),
            **apply(torch_to_numpy, kwargs),
        ))

    return wrapper


def numpy_function(function):

    def wrapper(*args, **kwargs):

        def torch_to_numpy(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return value

        def numpy_to_torch(value):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            return value

        return apply(torch_to_numpy, function(
            *apply(numpy_to_torch, args),
            **apply(numpy_to_torch, kwargs),
        ))

    return wrapper


def collate_nested_dicts(inputs):

    def collate_dicts(inputs):
        if isinstance(inputs, list) and all(isinstance(input, dict) for input in inputs):
            keys = functools.reduce(operator.and_, map(set, inputs))
            inputs = {key: list(map(operator.itemgetter(key), inputs)) for key in keys}
        return inputs

    while True:
        outputs = apply(collate_dicts, inputs)
        if outputs == inputs: break
        inputs = outputs

    def stack_if_possible(inputs):
        if isinstance(inputs, list) and all(isinstance(input, torch.Tensor) for input in inputs):
            if len(set(map(torch.Tensor.size, inputs))) == 1:
                inputs = torch.stack(inputs, dim=0)
        return inputs

    outputs = apply(stack_if_possible, outputs)

    return outputs


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # add a stream handler by dafault
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    format = "%(levelname)s: %(asctime)s: %(message)s"
    formatter = logging.Formatter(format)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
