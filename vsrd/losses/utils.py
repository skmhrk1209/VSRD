# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import torch


def reduced(loss_function):
    def wrapper(*args, reduction="mean", **kwargs):
        losses = loss_function(*args, **kwargs)
        if reduction == "none":
            return losses
        if reduction == "mean":
            return torch.mean(losses)
        if reduction == "sum":
            return torch.sum(losses)
        else:
            raise ValueError(f"`reduction` argument should be 'none'|'mean'|'sum', but got {reduction}.")
    return wrapper
