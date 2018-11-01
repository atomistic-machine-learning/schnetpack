import json
import importlib


class Hyperparameters:


    def __init__(self, config):
        """
        Args:
            config (dict): configuration of __init__
        """
        self.config = {}
        self.default_config = {}
        self.get_configs(config)

    def get_configs(self, config):
        config = config.copy()
        del config['self']
        self.config = config
        defaults = self.__init__.__defaults__
        if defaults is None:
            defaults = ()
        varnames = self.__init__.__code__.co_varnames[1:]
        n_defaults = len(defaults)
        n_vars = len(varnames)
        defaults = (None,) * (n_vars - n_defaults) + defaults
        self.default_config = dict(zip(varnames, defaults))

    # GET
    def get_default_config(self):
        return self.default_config.copy()

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

    def dump_default_config(self, fp):
        """
        Dump the hyper-parameters to a json file.

        Args:
            fp (str): path to file

        """
        dict_to_dump = self.get_dump_dict(self.default_config)
        with open(fp, 'w') as file:
            json.dump(dict_to_dump, file)

    def get_dump_dict(self, in_dict=None):
        """
        Prepare hyper-parameters for dumping to json file.

        Args:
            in_dict (dic): hyper-parameters

        """
        if in_dict is None:
            in_dict = self.config
        in_dict = in_dict.copy()
        dump_dict = {}
        for key, value in in_dict.items():
            dump_dict[key] = self._dictify_object(value)
        return dump_dict

    def _dictify_object(self, obj):
        """
        Transform object to be a part of the dump dict.

        Args:
            obj: can be a class, an instance of Hyperparameters, a List or any primitive datatype

        Returns:
            a dictionary that can be dumped to a JSON-file
        """
        if type(obj) == list:
            return [dict(self._dictify_object(element)) for element in obj]
        if isinstance(obj, type):
            return dict(class_name=obj.__name__, module=obj.__module__)
        if isinstance(obj, Hyperparameters):
            return dict(class_name=obj.__class__.__name__, module=obj.__class__.__module__,
                        params=self.get_dump_dict(obj.config))
        return obj

    # LOAD
    @classmethod
    def from_json(cls, fp):
        """
        Load configuration of a Model from a JSON config file.

        Args:
            fp (str): path to config JSON

        Returns:
            instance of model with loaded configuration

        """
        with open(fp) as file:
            json_config = json.load(file)
        return cls(**cls.from_dict(json_config))

    @classmethod
    def from_dict(cls, config):
        """
        Create self.config from default parameters and config settings.

        Args:
            config (dict): hyper-parameter configuration

        """
        new_config = {}
        for key, value in config.items():
            new_config[key] = cls._restore_object(value)
        return new_config

    @classmethod
    def _restore_object(cls, obj):
        """
        Restore an object that has been transforemd with self._dictify_object.

        Args:
            obj: part of the config dictionary

        Returns:
            the restored object
        """
        if type(obj) == list:
            print(obj)
            return [cls._restore_object(element) for element in obj]
        if type(obj) == dict and 'module' in obj.keys():
            module = importlib.import_module(obj['module'])
            module_class = getattr(module, obj['class_name'])
            if 'params' in obj.keys():
                return module_class(**cls.from_dict(obj['params']))
            else:
                return module_class
        return obj