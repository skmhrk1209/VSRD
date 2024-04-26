import os
import glob
import json
import functools


class Configurator(object):

    @staticmethod
    def gather(root, verbose=False):

        def gather_impl(root):

            def extract_default(*configs):

                if all(isinstance(config, dict) for config in configs):

                    default_configs = {}
                    for key in list(configs[0]):
                        if all(key in config for config in configs[1:]):
                            default_config = extract_default(*(config[key] for config in configs))
                            if default_config is not None:
                                default_configs[key] = default_config
                                for config in configs:
                                    if isinstance(default_config, dict):
                                        not config[key] and config.pop(key)
                                    else:
                                        config.pop(key)

                    return default_configs or None

                else:

                    return configs[0] if all(config == configs[0] for config in configs[1:]) else None

            filenames = sorted(glob.glob(os.path.join(root, "*")))
            dirnames = list(filter(os.path.isdir, filenames))

            if not dirnames: return root

            dirnames = list(map(gather_impl, dirnames))
            filenames = [os.path.join(dirname, "config.json") for dirname in dirnames]

            configs = []
            for filename in filenames:
                with open(filename) as file:
                    config = json.load(file)
                    configs.append(config)

            default_config = extract_default(*configs)
            default_config = default_config or {}

            filename = os.path.join(root, "config.json")
            assert not os.path.exists(filename)

            with open(filename, "w") as file:
                json.dump(default_config, file, indent=4, sort_keys=True)

            verbose and print(f"Created the default config in {root}.")

            for dirname, filename, config in zip(dirnames, filenames, configs):

                with open(filename, "w") as file:
                    json.dump(config, file, indent=4, sort_keys=True)

                verbose and print(f"Overwrited the existing config in {dirname}.")

            return root

        verbose and print(f"\nConfiguring {root}...\n")
        gather_impl(root)

    @staticmethod
    def scatter(root, verbose=False):

        def scatter_impl(root):

            filenames = sorted(glob.glob(os.path.join(root, "*")))
            dirnames = list(filter(os.path.isdir, filenames))

            if not dirnames: return root

            filename = os.path.join(root, "config.json")

            with open(filename) as file:
                default_config = json.load(file)

            os.remove(filename)

            verbose and print(f"Deleted the default config in {root}.")

            filenames = [os.path.join(dirname, "config.json") for dirname in dirnames]

            configs = []
            for filename in filenames:
                with open(filename) as file:
                    config = json.load(file)
                    configs.append(config)

            configs = [__class__.merge(config, default_config) for config in configs]

            for dirname, filename, config in zip(dirnames, filenames, configs):

                with open(filename, "w") as file:
                    json.dump(config, file, indent=4, sort_keys=True)

                verbose and print(f"Overwrited existing config in {dirname}.")

            dirnames = list(map(scatter_impl, dirnames))

            return root

        verbose and print(f"\nUnconfiguring {root}...\n")
        scatter_impl(root)

    @staticmethod
    def load(filename):

        assert os.path.exists(filename)

        def load_impl(filename):

            if not os.path.exists(filename): return []

            dirname = os.path.dirname(os.path.dirname(filename))
            configs = load_impl(os.path.join(dirname, "config.json"))

            with open(filename) as file:
                config = json.load(file)
                configs.append(config)

            return configs

        configs = load_impl(filename)
        config = __class__.merge(*configs)

        return config

    @staticmethod
    def merge(*configs):

        def merge_impl(source_config, target_config):

            if not all(isinstance(config, dict) for config in (source_config, target_config)):
                assert source_config == target_config
                return source_config

            config = {}

            for key in source_config:
                if key not in target_config:
                    config[key] = source_config[key]
                else:
                    config[key] = merge_impl(source_config[key], target_config[key])

            for key in target_config:
                if key not in source_config:
                    config[key] = target_config[key]

            return config

        config = functools.reduce(merge_impl, configs, {})

        return config
