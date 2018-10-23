import json
import importlib


class Hyperparameters:

    default_parameters = {}

    def __init__(self, **kwargs):
        self._create_attributes(**kwargs)

    # INIT
    def create_config(self, **kwargs):
        """
        Create self.config attribute from kwargs and default parameters.

        """
        set_config = kwargs
        self.config = self.default_parameters.copy()
        for key, value in set_config.items():
            if key in self.config.keys():
                self.config[key] = value
            else:
                print('Invalid parameter "{}" will not be used!'.format(key))

    def _create_attributes(self, **kwargs):
        """
        Create attributes from kwargs.

        """
        self.create_config(**kwargs)
        for param, value in self.config.items():
            setattr(self, param, value)

    # GET
    def get_default_config(self):
        return self.default_parameters.copy()

    def get_config(self):
        return self.config.copy()

    # DUMP
    def dump_config(self, fp):
        """
        Dump the hyper-parameters to a json file.

        Args:
            fp (str): path to file

        """
        dict_to_dump = self.get_dump_dict()
        with open(fp, 'w') as file:
            json.dump(dict_to_dump, file)

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
            if isinstance(value, type):
                dump_dict[key] = dict(class_name=value.__name__, module=value.__module__)
            elif isinstance(value, Hyperparameters):
                dump_dict[key] = dict(class_name=value.__class__.__name__, module=value.__class__.__module__,
                params=self.get_dump_dict(value.config))
            else:
                dump_dict[key] = value
        return dump_dict

    # LOAD
    def from_json(self, fp):
        with open(fp) as file:
            json_config = json.load(file)
        return type(self)(**self.from_dict(json_config))

    def from_dict(self, config):
        """
        Create self.config from default parameters and config settings.

        Args:
            config (dict): hyper-parameter configuration

        """
        new_config = {}
        for key, value in config.items():
            if type(value) == dict:
                module = importlib.import_module(value['module'])
                module_class = getattr(module, value['class_name'])
                if 'params' in value.keys():
                    new_config[key] = module_class(**self.from_dict(value['params']))
                else:
                    new_config[key] = module_class
            else:
                new_config[key] = value
        return new_config