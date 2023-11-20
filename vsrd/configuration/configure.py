# ================================================================
# Copyright 2022 SenseTime. All Rights Reserved.
# @author Hiroki Sakuma <sakuma@sensetime.jp>
# ================================================================

import os
import shutil
import argparse

from . import configurator


def main(args):

    cached_root = os.path.join(".cache", os.path.basename(os.path.normpath(args.root)))
    assert not os.path.exists(cached_root)

    shutil.copytree(args.root, cached_root)

    try:

        if args.gather:
            configurator.Configurator.gather(args.root, verbose=args.verbose)

        if args.scatter:
            configurator.Configurator.scatter(args.root, verbose=args.verbose)

    except Exception as error:

        print(f"{type(error)}: {error}, so restore the cached directory.")

        shutil.rmtree(args.root)
        shutil.move(cached_root, args.root)

    else:

        shutil.rmtree(cached_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="JSON Configurator")
    parser.add_argument("root", type=str)
    parser.add_argument("--gather", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    main(parser.parse_args())
