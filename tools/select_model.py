import os
import re
import json
import glob
import shutil
import operator
import argparse

import torch
import tqdm


def main(args):

    regex = re.compile(r"(\w+)(<|<=|==|!=|>=|>)([+-]?(?:\d+\.?\d*|\.\d+))")
    operations = {
        "<" : operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">" : operator.gt,
    }

    constraints = []
    for constraint in args.constraints:
        match = regex.match(constraint)
        name, operation, value = match.groups()
        constraints.append(
            lambda metrics, name=name, operation=operation, value=value:
            operations[operation](metrics[name], float(value))
        )

    best_metrics = None
    best_filename = None
    filenames = glob.glob(os.path.join(args.dirname, "**", "*.pt"), recursive=True)
    for filename in tqdm.tqdm(filenames):
        if os.path.basename(filename) == "model.pt": continue
        checkpoint = torch.load(filename, map_location="cpu")
        if not all(metric in checkpoint["metrics"] for metric in args.metrics): continue
        if not all(constraint(checkpoint["metrics"]) for constraint in constraints): continue
        if not (not args.epoch_range or checkpoint["epoch"] in range(*args.epoch_range)): continue
        metrics = operator.itemgetter(*args.metrics)(checkpoint["metrics"])
        metrics = metrics if isinstance(metrics, tuple) else (metrics,)
        mean = lambda iterable: sum(iterable) / len(iterable)
        if not best_metrics or (operator.gt if args.maximum else operator.lt)(mean(metrics), mean(best_metrics)):
            best_metrics = metrics
            best_filename = filename

    shutil.copy(best_filename, os.path.join(args.dirname, "model.pt"))
    print(f"{best_filename} -> {os.path.join(args.dirname, 'model.pt')}: {json.dumps(dict(zip(args.metrics, best_metrics)), indent=4)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TorchDet3D: Best Model Selector")
    parser.add_argument("--dirname", type=str)
    parser.add_argument("--metrics", type=str, nargs="+", default=[])
    parser.add_argument("--constraints", type=str, nargs="+", default=[])
    parser.add_argument("--epoch_range", type=int, nargs="+", default=[])
    parser.add_argument("--maximum", action="store_true")
    args = parser.parse_args()

    main(args)
