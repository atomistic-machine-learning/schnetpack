import pytest
import os
from schnetpack.config_model import Hyperparameters


class Dummy(object):
    def __init__(self, dummy_a):
        self.dummy_a = dummy_a

class SubNetwork(Hyperparameters):

    def __init__(self, s1=1, s2=2, s3=3):
        Hyperparameters.__init__(self, locals())

    def __eq__(self, other):
        for attr in ['s1', 's2', 's3']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

subnetwork = SubNetwork()

class Network(Hyperparameters):

    def __init__(self, a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork):
        Hyperparameters.__init__(self, locals())

    def __eq__(self, other):
        for attr in ['a', 'b', 'c', 'd', 'dummy']:
            if getattr(self, attr) != getattr(other, attr):
                return False
            if self.sub != other.sub:
                return False
        return True

net = Network(c=13, d=14)
default_net = Network()
default_config = dict(a=1, b=2, c=3, d=4, dummy=Dummy, sub=subnetwork)
updated_config = dict(a=1, b=2, c=13, d=14, dummy=Dummy, sub=subnetwork)

class TestHyperparams(object):

    def test_init(self):
        assert net.a == 1
        assert net.b == 2
        assert net.c == 13
        assert net.d == 14

    def test_config_update(self):
        assert net.config == updated_config

    def test_empty_config_update(self):
        assert default_net.config == default_config

    def test_empty_config_init(self):
        for key in default_config.keys():
            assert getattr(default_net, key) == default_config[key]

    def test_dump_and_load(self):
        net.dump_config('config_file')
        net2 = Network().from_json('config_file')
        for attribute in ['a', 'b', 'c', 'd', 'config', 'default_parameters', 'dummy']:
            assert getattr(net, attribute) == getattr(net2, attribute)

    @classmethod
    def teardown_class(cls):
        """
        Remove artifacts that have been created during testing.
        """
        if os.path.exists('config_file'):
            os.remove('config_file')





