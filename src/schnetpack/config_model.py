import json
import importlib


class Hyperparameters:

    default_parameters = {}

    def __init__(self):
        self.config = {}

    def get_default_config(self):
        return self.default_parameters.copy()

    def get_config(self):
        return self.config.copy()

    def get_dump_dict(self, in_dict=None):
        """
        Prepare hyper-parameters for dumping to json file.
        Args:
            in_dict (dic): hyper-parameters

        """
        if in_dict is None:
            in_dict = self.config.copy()
        dump_dict = {}
        for key, value in in_dict.items():
            if isinstance(value, type) and issubclass(value, Hyperparameters):
                dump_dict[key] = dict(class_name=value.__name__, module=value.__module__)
            elif isinstance(value, Hyperparameters):
                dump_dict[key] = dict(name=value.__class__.__name__, module=value.__class__.__module__,
                params=self.get_dump_dict(value.config))
            else:
                dump_dict[key] = value

        return dump_dict

    def dump_config(self):
        """
        Dump the hyper-parameters to a json file.

        """
        dict_to_dump = self.get_dump_dict()
        with open('./test', 'w') as file:
            json.dump(dict_to_dump, file)

    def load_config_json(self, fp):
        """
        Load a json configuration as dict.
        Args:
            fp (str): path to json file

        """
        with open(fp, 'r') as file:
            config = json.load(file)
        return config

    def update_config(self, config):
        self.config = self.get_default_config()
        for key, value in config.items():
            if type(value) == dict:
                module_class = importlib.import_module(value['module'] + '.' + value['name'])
                if 'params' in value.keys():
                    self.config.update(dict(key=module_class(value['params'])))
                else:
                    self.config.update(dict(key=module_class))
            else:
                self.config.update(dict(key=value))

    def load_config(self, fp):
        config = self.load_config_json(fp)
        self.update_config(config)

    def _create_config(self, **kwargs):
        self.config = kwargs
        self.config.update(self.default_parameters)

    def _create_attributes(self, **kwargs):
        self._create_config(**kwargs)
        for param, value in self.config.items():
            if param in self.default_parameters.keys():
                setattr(self, param, value)
            else:
                print('Invalid parameter "{}" will not be used!')